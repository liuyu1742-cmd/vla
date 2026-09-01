from tools.profile_openvla_finetune_modes_v2 import (
    action_text,
    classify_parameter,
    summarize_named_parameters,
)


def test_exact_project_modes():
    assert classify_parameter("vision_backbone.layer.weight", "full")
    assert not classify_parameter("vision_backbone.layer.weight", "frozen_vision")
    assert classify_parameter("language_model.layers.31.weight", "last_layer_only")
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


def test_action_text_is_dependency_free():
    class Tokenizer:
        vocab_size = 32000

        @staticmethod
        def decode(ids):
            return ",".join(str(value) for value in ids)

    assert len(action_text(Tokenizer(), [0.0] * 7).split(",")) == 7

