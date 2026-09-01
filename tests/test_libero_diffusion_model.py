"""Unit contracts for the compact LIBERO diffusion policy."""

from __future__ import annotations

import pytest
import torch

from tools.libero_diffusion_model import (
    ByteInstructionEncoder,
    CompactDiffusionPolicy,
    DiffusionPolicy,
    DiffusionPolicyConfig,
)


def _batch(batch_size: int = 2, device: str = "cpu"):
    config = DiffusionPolicyConfig()
    condition = {
        "visual": torch.randn(
            batch_size, config.obs_horizon, config.visual_dim, device=device
        ),
        "state": torch.randn(
            batch_size, config.obs_horizon, config.state_dim, device=device
        ),
        "instructions": ["put the cup on the plate", "open the drawer"][:batch_size],
    }
    return {
        "condition": condition,
        "actions": torch.randn(
            batch_size, config.action_horizon, config.action_dim, device=device
        ).clamp(-1, 1),
    }


def test_default_config_instruction_encoder_and_architecture_contract():
    config = DiffusionPolicyConfig()
    assert (config.visual_dim, config.state_dim, config.obs_horizon) == (512, 8, 2)
    assert (config.action_dim, config.action_horizon) == (7, 8)

    encoder = ByteInstructionEncoder()
    tokens = torch.tensor(
        [[99, 117, 112, 256, 256], [111, 112, 101, 110, 256]], dtype=torch.long
    )
    lengths = torch.tensor([3, 4], dtype=torch.long)
    encoded = encoder(tokens, lengths)

    assert encoded.shape == (2, 128)
    assert encoded.device.type == "cpu"
    assert encoder.padding_index == 256
    assert encoder.embedding.embedding_dim == 32
    assert encoder.gru.hidden_size == 128
    assert encoder.gru.num_layers == 1

    model = CompactDiffusionPolicy()
    assert DiffusionPolicy is CompactDiffusionPolicy
    assert model.scheduler.config.num_train_timesteps == 50
    assert model.scheduler.config.beta_schedule == "squaredcos_cap_v2"
    assert model.scheduler.config.prediction_type == "epsilon"
    assert len(model.residual_blocks) == 4


def test_instruction_encoder_ignores_values_beyond_lengths():
    encoder = ByteInstructionEncoder().eval()
    lengths = torch.tensor([3], dtype=torch.long)
    padded = torch.tensor([[99, 117, 112, 256, 256]], dtype=torch.long)
    arbitrary_tail = torch.tensor([[99, 117, 112, 7, 11]], dtype=torch.long)

    assert torch.equal(
        encoder(padded, lengths),
        encoder(arbitrary_tail, lengths),
    )


def test_training_loss_is_scalar_and_sample_has_action_block_shape_cpu():
    model = CompactDiffusionPolicy().eval()
    batch = _batch()

    loss = model.training_loss(batch)
    sampled = model.sample_actions(
        batch["condition"],
        generator=torch.Generator(device="cpu").manual_seed(7),
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert sampled.shape == (2, 8, 7)
    assert sampled.device.type == "cpu"
    assert torch.all(sampled >= -1) and torch.all(sampled <= 1)
    assert model.scheduler.num_inference_steps == 10


def test_language_conditions_change_sampled_actions():
    torch.manual_seed(12)
    model = CompactDiffusionPolicy().eval()
    condition = _batch(batch_size=1)["condition"]
    place = {**condition, "instructions": ["place the red cup"]}
    opened = {**condition, "instructions": ["open the cabinet"]}

    place_actions = model.sample_actions(
        place,
        generator=torch.Generator(device="cpu").manual_seed(91),
    )
    open_actions = model.sample_actions(
        opened,
        generator=torch.Generator(device="cpu").manual_seed(91),
    )

    assert not torch.equal(place_actions, open_actions)


def test_sampling_is_deterministic_for_equal_generator_seeds():
    model = CompactDiffusionPolicy().eval()
    condition = _batch(batch_size=1)["condition"]

    first = model.sample_actions(
        condition,
        generator=torch.Generator(device="cpu").manual_seed(123),
    )
    second = model.sample_actions(
        condition,
        generator=torch.Generator(device="cpu").manual_seed(123),
    )

    assert torch.equal(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_autocast_smoke():
    model = CompactDiffusionPolicy().cuda().eval()
    batch = _batch(device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model.training_loss(batch)

    assert torch.isfinite(loss.float())
