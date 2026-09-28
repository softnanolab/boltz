"""Exercise the real CPU sampling loop without a checkpoint or GPU kernels."""

# ruff: noqa: CPY001, INP001
from __future__ import annotations

import math

import pytest
import torch

from boltz.model.modules import diffusionv2


class _TinyScoreModel(torch.nn.Module):
    def __init__(self, **_kwargs: object) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))


def _sampler(
    monkeypatch: pytest.MonkeyPatch, particle_count: int
) -> tuple[diffusionv2.AtomDiffusion, list[int]]:
    monkeypatch.setattr(diffusionv2, "DiffusionModule", _TinyScoreModel)
    sampler = diffusionv2.AtomDiffusion(
        score_model_args={"token_s": 1}, num_sampling_steps=3, step_scale=1.0
    ).eval()
    batch_sizes = []
    seen = 0

    def denoise(
        coords: torch.Tensor, _sigma: float, network_condition_kwargs: dict[str, object]
    ) -> torch.Tensor:
        nonlocal seen
        batch_size = coords.shape[0]
        assert network_condition_kwargs["multiplicity"] == batch_size
        batch_sizes.append(batch_size)
        # Give every particle a distinct answer each step. The real sampler
        # must scatter these answers back into their original rows.
        first = seen % particle_count + 1
        seen += batch_size
        labels = torch.arange(first, first + batch_size, dtype=coords.dtype)
        return labels[:, None, None].expand_as(coords).clone()

    monkeypatch.setattr(sampler, "preconditioned_network_forward", denoise)
    return sampler, batch_sizes


def _steering(**overrides: object) -> dict[str, object]:
    return {
        "fk_steering": False,
        "physical_guidance_update": False,
        "contact_guidance_update": False,
        **overrides,
    }


@pytest.mark.parametrize(
    ("samples", "cap"), [(32, 5), (32, 8), (4, 1), (1, None), (1, 5), (3, 5)]
)
def test_sample_caps_each_denoiser_batch_and_retains_every_sample(
    monkeypatch: pytest.MonkeyPatch, samples: int, cap: int | None
) -> None:
    """Enforce the memory cap without dropping or mixing output samples."""
    sampler, batch_sizes = _sampler(monkeypatch, samples)

    result = sampler.sample(
        atom_mask=torch.ones((1, 4)),
        multiplicity=samples,
        max_parallel_samples=cap,
        steering_args=_steering(),
    )

    limit = samples if cap is None else cap
    assert max(batch_sizes) <= limit
    batches_per_step = math.ceil(samples / limit)
    assert len(batch_sizes) == sampler.num_sampling_steps * batches_per_step
    for start in range(0, len(batch_sizes), batches_per_step):
        assert sum(batch_sizes[start : start + batches_per_step]) == samples
    coords = result["sample_atom_coords"]
    assert coords.shape == (samples, 4, 3)
    expected = torch.arange(1, samples + 1, dtype=coords.dtype)[:, None, None]
    torch.testing.assert_close(coords, expected.expand_as(coords))


def test_sample_caps_expanded_steering_particles_before_final_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cap denoising particles and keep one output per requested sample."""
    samples, particles, cap = 3, 4, 5
    sampler, batch_sizes = _sampler(monkeypatch, samples * particles)
    # Exercise particle expansion/resampling without evaluating physical energy.
    monkeypatch.setattr(diffusionv2, "get_potentials", lambda *_args, **_kwargs: [])

    result = sampler.sample(
        atom_mask=torch.ones((1, 4)),
        multiplicity=samples,
        max_parallel_samples=cap,
        steering_args=_steering(
            fk_steering=True,
            num_particles=particles,
            fk_resampling_interval=1,
            fk_lambda=1.0,
        ),
    )

    assert batch_sizes == [5, 5, 2] * sampler.num_sampling_steps
    coords = result["sample_atom_coords"]
    assert coords.shape == (samples, 4, 3)
    assert torch.isfinite(coords).all()
    labels = coords[:, 0, 0]
    lower = torch.arange(samples) * particles + 1
    assert torch.all(labels >= lower - 1e-5)
    assert torch.all(labels <= lower + particles - 1 + 1e-5)
    torch.testing.assert_close(coords, labels[:, None, None].expand_as(coords))


@pytest.mark.parametrize("cap", [0, -1])
def test_sample_rejects_nonpositive_parallel_cap(
    monkeypatch: pytest.MonkeyPatch, cap: int
) -> None:
    """Reject invalid caps before calling the network."""
    sampler, batch_sizes = _sampler(monkeypatch, 1)

    with pytest.raises(ValueError, match="max_parallel_samples must be positive"):
        sampler.sample(
            atom_mask=torch.ones((1, 4)),
            multiplicity=1,
            max_parallel_samples=cap,
            steering_args=_steering(),
        )

    assert batch_sizes == []
