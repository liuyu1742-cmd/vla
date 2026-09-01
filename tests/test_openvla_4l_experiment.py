from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.openvla_4l_experiment import (
    is_trainable,
    reduce_language_layers,
    selected_layer_indices,
    select_keyframe_steps,
    success_rate,
)


def test_four_layer_selection_is_deterministic():
    assert selected_layer_indices(32, 4) == (0, 1, 2, 3)


@pytest.mark.parametrize("keep_layers", (0, 33))
def test_layer_selection_rejects_invalid_depth(keep_layers):
    with pytest.raises(ValueError, match="within the source depth"):
        selected_layer_indices(32, keep_layers)


def test_last_layer_only_selects_layer_three_and_lm_head():
    assert is_trainable(
        "language_model.model.layers.3.self_attn.q_proj.weight",
        "last_layer_only",
        3,
    )
    assert is_trainable("language_model.lm_head.weight", "last_layer_only", 3)
    assert not is_trainable(
        "language_model.model.layers.2.self_attn.q_proj.weight",
        "last_layer_only",
        3,
    )


def test_frozen_vision_excludes_only_vision_backbone():
    assert not is_trainable(
        "vision_backbone.featurizer.blocks.0.attn.qkv.weight",
        "frozen_vision",
        3,
    )
    assert is_trainable("projector.fc1.weight", "frozen_vision", 3)
    assert is_trainable(
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "frozen_vision",
        3,
    )


def test_reduce_language_layers_updates_both_configs():
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(layers=list(range(8))),
            config=SimpleNamespace(num_hidden_layers=8),
        ),
        config=SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=8),
        ),
    )
    metadata = reduce_language_layers(model, 4)
    assert model.language_model.model.layers == [0, 1, 2, 3]
    assert model.language_model.config.num_hidden_layers == 4
    assert model.config.text_config.num_hidden_layers == 4
    assert metadata == {
        "source_layers": 8,
        "kept_layers": 4,
        "selected_layer_indices": [0, 1, 2, 3],
    }


def test_keyframes_include_start_middle_and_terminal():
    assert select_keyframe_steps([{"step": index} for index in range(9)]) == (0, 4, 8)


def test_success_rate_uses_episode_booleans():
    assert success_rate([{"success": True}, {"success": False}]) == 50.0
    with pytest.raises(ValueError, match="at least one episode"):
        success_rate([])

