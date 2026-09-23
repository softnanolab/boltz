"""The export must preserve logits and describe exactly the unpadded tokens."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boltz.data import const
from boltz.ecstasy import distogram_payload


def test_export_preserves_logits_and_masks_both_axes():
    model = SimpleNamespace(min_dist=2.0, max_dist=22.0, num_bins=64)
    residues = torch.zeros(1, 3, len(const.tokens))
    residues[0, 0, const.token_ids["ALA"]] = 1
    residues[0, 1, const.token_ids["GLY"]] = 1
    batch = {
        "record": [None],
        "token_pad_mask": torch.tensor([[1, 1, 0]]),
        "mol_type": torch.zeros(1, 3),
        "res_type": residues,
        "asym_id": torch.tensor([[0, 1, 0]]),
        "residue_index": torch.tensor([[0, 0, 0]]),
        "token_index": torch.tensor([[0, 1, 2]]),
    }
    logits = torch.arange(3 * 3 * 64).reshape(1, 3, 3, 1, 64).float()
    payload = distogram_payload(model, batch, {"pdistogram": logits})
    np.testing.assert_array_equal(payload["logits"], logits[0, :2, :2, 0])
    np.testing.assert_array_equal(payload["residue_name"], ["ALA", "GLY"])
    np.testing.assert_array_equal(payload["representative_atom"], ["CB", "CA"])
    np.testing.assert_array_equal(payload["asym_id"], [0, 1])
    np.testing.assert_array_equal(payload["bin_edges_a"], np.linspace(2, 22, 63))
    batch["mol_type"][0, 0] = const.chain_type_ids["DNA"]
    with pytest.raises(ValueError, match="protein-only"):
        distogram_payload(model, batch, {"pdistogram": logits})
