from tools.profile_openvla_finetune_modes import (
    classify_parameter,
    summarize_named_parameters,
)


def test_full_mode_trains_every_parameter():
    assert classify_parameter("vision_backbone.layer.weight", "full")


def test_frozen_vision_excludes_vision_parameters():
    assert not classify_parameter("vision_backbone.layer.weight", "frozen_vision")
    assert classify_parameter("language_model.layers.0.weight", "frozen_vision")


def test_last_layer_only_is_narrow():
    assert classify_parameter("language_model.layers.31.weight", "last_layer_only")
    assert classify_parameter("language_model.lm_head.weight", "last_layer_only")
    assert not classify_parameter("language_model.layers.30.weight", "last_layer_only")


def test_summary_counts_selected_parameters():
    named = [
        ("vision_backbone.layer.weight", 10),
        ("language_model.layers.30.weight", 20),
        ("language_model.layers.31.weight", 30),
        ("language_model.lm_head.weight", 40),
    ]

    row = summarize_named_parameters(named, "last_layer_only")

    assert row["total_parameters"] == 100
    assert row["trainable_parameters"] == 70

