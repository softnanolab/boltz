"""Persistent, single-GPU protein inference using Boltz's stock prediction path.

Only orchestration and distogram export live here. The model, feature generation,
CUDA kernels, diffusion sampler and structure writer are unchanged.
"""

import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import seed_everything
from rdkit import Chem

from boltz.data import const
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzWriter
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
    process_inputs,
)
from boltz.model.models.boltz2 import Boltz2


def distogram_payload(model, batch, prediction):
    """Export unpadded protein logits and the model's own bin/token metadata."""

    def array(tensor):
        return tensor.detach().cpu().numpy()

    if len(batch["record"]) != 1:
        raise ValueError("distogram export requires one record per batch")
    keep = array(batch["token_pad_mask"][0]).astype(bool)
    if np.any(array(batch["mol_type"][0])[keep] != const.chain_type_ids["PROTEIN"]):
        raise ValueError("distogram export supports protein-only inputs")
    raw = prediction["pdistogram"]
    if raw.ndim != 5 or raw.shape[0] != 1 or raw.shape[-2] != 1:
        raise ValueError(f"expected [1, tokens, tokens, 1, bins], got {raw.shape}")
    logits = array(raw[0, :, :, 0].float())[keep][:, keep]
    if logits.shape[-1] != model.num_bins or not np.isfinite(logits).all():
        raise ValueError("invalid distogram logits or bin count")
    types = array(batch["res_type"][0])[keep].argmax(-1)
    names = np.asarray([const.tokens[int(i)] for i in types])
    return {
        "schema_version": np.int32(1),
        "logits": logits,
        "bin_edges_a": np.linspace(model.min_dist, model.max_dist, model.num_bins - 1),
        "asym_id": array(batch["asym_id"][0])[keep],
        "residue_index": array(batch["residue_index"][0])[keep],
        "token_index": array(batch["token_index"][0])[keep],
        "residue_name": names,
        "representative_atom": np.asarray([const.res_to_disto_atom[str(n)] for n in names]),
        "boundary_convention": np.asarray("lower_bin_includes_edge"),
    }


class Boltz2Predictor:
    """Load once; reset the RNG before each target's preprocessing and prediction.

    The per-target seed is independent of model initialization and target order.
    It does not reproduce the CLI's RNG stream, which seeds before model loading.
    Output directories must be fresh to avoid upstream's processed-input cache.
    """

    def __init__(self, checkpoint: Path, mol_dir: Path):
        if not torch.cuda.is_available():
            raise RuntimeError("Boltz2Predictor requires a CUDA GPU")
        self.mol_dir = Path(mol_dir)
        torch.set_float32_matmul_precision("highest")
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        for key in ("CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"):
            os.environ.setdefault(key, "1")
        self.model = (
            Boltz2.load_from_checkpoint(
                checkpoint,
                strict=True,
                map_location="cpu",
                predict_args={},
                diffusion_process_args=asdict(Boltz2DiffusionParams()),
                ema=False,
                use_kernels=True,
                pairformer_args=asdict(PairformerArgsV2()),
                msa_args=asdict(MSAModuleArgs()),
                steering_args=asdict(BoltzSteeringParams()),
            )
            .eval()
            .cuda()
        )

    def predict(
        self,
        input_path: Path,
        out_dir: Path,
        *,
        seed: int,
        recycling_steps: int,
        sampling_steps: int,
        diffusion_samples: int,
        max_parallel_samples: int,
        step_scale: float,
        max_msa_seqs: int,
    ):
        """Write stock PDB/confidence outputs and return the learned distogram."""
        input_path, out_dir = Path(input_path), Path(out_dir)
        if out_dir.exists():
            raise FileExistsError(f"prediction requires a fresh output directory: {out_dir}")
        seed_everything(seed)
        process_inputs(
            data=[input_path],
            out_dir=out_dir,
            ccd_path=self.mol_dir.parent / "ccd.pkl",
            mol_dir=self.mol_dir,
            msa_server_url="",
            msa_pairing_strategy="greedy",
            max_msa_seqs=max_msa_seqs,
            use_msa_server=False,
            boltz2=True,
            preprocessing_threads=1,
        )
        processed = out_dir / "processed"
        manifest = Manifest.load(processed / "manifest.json")
        if [record.id for record in manifest.records] != [input_path.stem]:
            raise ValueError(f"Boltz could not process {input_path}")
        data = Boltz2InferenceDataModule(
            manifest=manifest,
            target_dir=processed / "structures",
            msa_dir=processed / "msa",
            mol_dir=self.mol_dir,
            num_workers=0,
            constraints_dir=processed / "constraints",
            template_dir=processed / "templates",
            extra_mols_dir=processed / "mols",
        )
        self.model.predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
            "keys_dict_out": ["pdistogram"],
        }
        self.model.structure_module.step_scale = step_scale
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            batch = next(iter(data.predict_dataloader()))
            batch = data.transfer_batch_to_device(batch, self.model.device, 0)
            prediction = self.model.predict_step(batch, 0)
            if prediction.get("exception", False):
                raise RuntimeError("Boltz prediction failed (CUDA out of memory)")
            payload = distogram_payload(self.model, batch, prediction)
            writer = BoltzWriter(
                data_dir=processed / "structures",
                output_dir=out_dir / "predictions",
                output_format="pdb",
                boltz2=True,
            )
            writer.write_on_batch_end(None, self.model, prediction, [], batch, 0, 0)
        return payload
