from pathlib import Path

import pytest


def test_render_variant_bddl_changes_only_language_interest_and_goal():
    from tools.experiment_4_2_5_novel_objects import (
        NOVEL_VARIANTS,
        render_variant_bddl,
    )

    source = (
        "  (:language Put the bowl on the stove)\n"
        "  (:obj_of_interest\n"
        "    akita_black_bowl_1\n"
        "    flat_stove_1_cook_region\n"
        "  )\n"
        "  (:goal\n"
        "    (And (On akita_black_bowl_1 flat_stove_1_cook_region))\n"
        "  )\n"
    )

    rendered = render_variant_bddl(source, NOVEL_VARIANTS["wine_bottle_on_stove"])

    assert "(:language Put the wine bottle on the stove)" in rendered
    assert "    wine_bottle_1\n    flat_stove_1_cook_region" in rendered
    assert "(And (On wine_bottle_1 flat_stove_1_cook_region))" in rendered
    assert "(:language Put the bowl on the stove)" not in rendered


def test_render_variant_bddl_rejects_ambiguous_source():
    from tools.experiment_4_2_5_novel_objects import (
        NOVEL_VARIANTS,
        render_variant_bddl,
    )

    with pytest.raises(ValueError, match="exactly once"):
        render_variant_bddl(
            "(:language Put the bowl on the stove)\n"
            "(:language Put the bowl on the stove)\n",
            NOVEL_VARIANTS["wine_bottle_on_stove"],
        )


def test_summarize_episodes_reports_success_rate_and_weighted_latency():
    from tools.experiment_4_2_5_novel_objects import summarize_episodes

    summary = summarize_episodes(
        [
            {
                "success": True,
                "decision_steps": 2,
                "mean_inference_seconds": 0.4,
            },
            {
                "success": False,
                "decision_steps": 1,
                "mean_inference_seconds": 0.7,
            },
        ]
    )

    assert summary["episodes"] == 2
    assert summary["successes"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["mean_inference_seconds"] == pytest.approx(0.5)


def test_variant_sources_exist_in_local_libero_tree():
    from tools.experiment_4_2_5_novel_objects import NOVEL_VARIANTS

    for variant in NOVEL_VARIANTS.values():
        assert Path(variant.source_bddl).is_file()
