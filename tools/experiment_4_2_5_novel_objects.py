"""Evaluate task-level object substitutions for experiment 4.2.5.

Each variant preserves an official LIBERO-Goal scene and its fixed initial
states, but changes the manipulated object or articulated sub-part in the
language, object-of-interest block, and goal predicate. OpenVLA acts directly;
no expert action, recovery controller, or simulator-state policy is used.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.openvla_libero_goal_pure_eval import (
    LIBERO_ROOT,
    REPO_ROOT,
    _configure_libero,
    _environment_for_task,
    _libero_image,
    _load_model,
    _predict_openvla_action,
    _set_seed,
)


GOAL_BDDL_ROOT = (
    LIBERO_ROOT / "libero" / "libero" / "bddl_files" / "libero_goal"
)


@dataclass(frozen=True)
class NovelVariant:
    variant_id: str
    source_task_id: int
    source_task: str
    source_object: str
    novel_task: str
    novel_object: str
    source_bddl: Path
    replacements: tuple[tuple[str, str], ...]


NOVEL_VARIANTS = {
    "wine_bottle_on_stove": NovelVariant(
        variant_id="wine_bottle_on_stove",
        source_task_id=1,
        source_task="put the bowl on the stove",
        source_object="bowl",
        novel_task="put the wine bottle on the stove",
        novel_object="wine bottle",
        source_bddl=GOAL_BDDL_ROOT / "put_the_bowl_on_the_stove.bddl",
        replacements=(
            (
                "(:language Put the bowl on the stove)",
                "(:language Put the wine bottle on the stove)",
            ),
            (
                "    akita_black_bowl_1\n    flat_stove_1_cook_region",
                "    wine_bottle_1\n    flat_stove_1_cook_region",
            ),
            (
                "(And (On akita_black_bowl_1 flat_stove_1_cook_region))",
                "(And (On wine_bottle_1 flat_stove_1_cook_region))",
            ),
        ),
    ),
    "bowl_to_stove_front": NovelVariant(
        variant_id="bowl_to_stove_front",
        source_task_id=5,
        source_task="push the plate to the front of the stove",
        source_object="plate",
        novel_task="push the bowl to the front of the stove",
        novel_object="bowl",
        source_bddl=GOAL_BDDL_ROOT / "push_the_plate_to_the_front_of_the_stove.bddl",
        replacements=(
            (
                "(:language Push the plate to the front of the stove)",
                "(:language Push the bowl to the front of the stove)",
            ),
            (
                "    plate_1\n    main_table_stove_front_region",
                "    akita_black_bowl_1\n    main_table_stove_front_region",
            ),
            (
                "(And (On plate_1 main_table_stove_front_region))",
                "(And (On akita_black_bowl_1 main_table_stove_front_region))",
            ),
        ),
    ),
    "open_top_drawer": NovelVariant(
        variant_id="open_top_drawer",
        source_task_id=0,
        source_task="open the middle drawer of the cabinet",
        source_object="middle drawer",
        novel_task="open the top drawer of the cabinet",
        novel_object="top drawer",
        source_bddl=GOAL_BDDL_ROOT / "open_the_middle_drawer_of_the_cabinet.bddl",
        replacements=(
            (
                "(:language Open the middle layer of the drawer)",
                "(:language Open the top drawer of the cabinet)",
            ),
            (
                "    wooden_cabinet_1_middle_region",
                "    wooden_cabinet_1_top_region",
            ),
            (
                "(And (Open wooden_cabinet_1_middle_region))",
                "(And (Open wooden_cabinet_1_top_region))",
            ),
        ),
    ),
}


def render_variant_bddl(source_text: str, variant: NovelVariant) -> str:
    """Apply audited, exact-once substitutions to an official BDDL task."""
    rendered = source_text
    for old, new in variant.replacements:
        count = rendered.count(old)
        if count != 1:
            raise ValueError(
                f"{variant.variant_id}: expected replacement source exactly once, "
                f"found {count}: {old!r}"
            )
        rendered = rendered.replace(old, new, 1)
    return rendered


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("episodes must not be empty")
    successes = sum(bool(row["success"]) for row in episodes)
    timed_steps = sum(int(row["decision_steps"]) for row in episodes)
    weighted_seconds = sum(
        float(row["mean_inference_seconds"]) * int(row["decision_steps"])
        for row in episodes
        if row["mean_inference_seconds"] is not None
    )
    return {
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes),
        "decision_steps": timed_steps,
        "mean_inference_seconds": (
            weighted_seconds / timed_steps if timed_steps else None
        ),
    }


def _write_variant_bddl(variant: NovelVariant, directory: Path) -> Path:
    source_text = variant.source_bddl.read_text(encoding="utf-8")
    path = directory / f"{variant.variant_id}.bddl"
    path.write_text(render_variant_bddl(source_text, variant), encoding="utf-8")
    return path


def _variant_environment(bddl_path: Path):
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(0)
    return env


def run_novel_object_evaluation(
    model_dir: Path,
    variant_ids: list[str],
    trials_per_variant: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    _configure_libero()
    from libero.libero import benchmark

    if trials_per_variant < 1:
        raise ValueError("trials_per_variant must be at least one")
    unknown = sorted(set(variant_ids) - set(NOVEL_VARIANTS))
    if unknown:
        raise ValueError(f"unknown variant IDs: {unknown}")

    _set_seed(seed)
    model, processor = _load_model(model_dir)
    unnorm_key = "libero_goal"
    if unnorm_key not in model.norm_stats:
        raise KeyError(f"missing action normalization key: {unnorm_key}")

    suite = benchmark.get_benchmark_dict()[unnorm_key]()
    variant_dir = output_dir / "variant_bddl"
    variant_dir.mkdir(parents=True, exist_ok=True)
    all_episodes: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []

    for variant_id in variant_ids:
        variant = NOVEL_VARIANTS[variant_id]
        initial_states = suite.get_task_init_states(variant.source_task_id)
        if trials_per_variant > len(initial_states):
            raise ValueError(
                f"{variant_id} exposes only {len(initial_states)} initial states"
            )
        bddl_path = _write_variant_bddl(variant, variant_dir)
        env = _variant_environment(bddl_path)
        episodes: list[dict[str, Any]] = []

        for episode_index in range(trials_per_variant):
            env.reset()
            observation = env.set_init_state(initial_states[episode_index])
            done = False
            error: str | None = None
            decision_steps = 0
            inference_seconds: list[float] = []

            for step in range(310):
                try:
                    if step < 10:
                        observation, _, done, _ = env.step(
                            [0, 0, 0, 0, 0, 0, -1]
                        )
                        continue
                    image = _libero_image(observation)
                    started = time.perf_counter()
                    action = _predict_openvla_action(
                        model,
                        processor,
                        image,
                        variant.novel_task,
                        unnorm_key,
                    )
                    inference_seconds.append(time.perf_counter() - started)
                    decision_steps += 1
                    action[-1] = np.sign(2.0 * action[-1] - 1.0)
                    action[-1] *= -1.0
                    observation, _, done, _ = env.step(action.tolist())
                    if done:
                        break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    break

            row = {
                "variant_id": variant_id,
                "source_task_id": variant.source_task_id,
                "source_task": variant.source_task,
                "source_object": variant.source_object,
                "novel_task": variant.novel_task,
                "novel_object": variant.novel_object,
                "episode_index": episode_index,
                "success": bool(done),
                "decision_steps": decision_steps,
                "mean_inference_seconds": (
                    float(np.mean(inference_seconds))
                    if inference_seconds
                    else None
                ),
                "error": error,
            }
            episodes.append(row)
            all_episodes.append(row)
        env.close()

        summary = summarize_episodes(episodes)
        variant_summaries.append(
            {
                "variant_id": variant_id,
                "source_task_id": variant.source_task_id,
                "source_task": variant.source_task,
                "source_object": variant.source_object,
                "novel_task": variant.novel_task,
                "novel_object": variant.novel_object,
                **summary,
            }
        )

    report = {
        "evaluation_kind": "libero_goal_task_level_novel_object_substitution",
        "checkpoint": str(model_dir.resolve()),
        "seed": seed,
        "trials_per_variant": trials_per_variant,
        "uses_expert_recovery": False,
        "uses_simulator_state_policy": False,
        "novelty_definition": (
            "The manipulated object or articulated sub-part is substituted "
            "relative to the source task while scene geometry, robot, camera, "
            "initial-state vector, and target region remain fixed."
        ),
        "training_trajectory_count": 428,
        "variants": variant_summaries,
        "episodes": all_episodes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--variant-ids",
        default=",".join(NOVEL_VARIANTS),
        help="Comma-separated variant IDs.",
    )
    parser.add_argument("--trials-per-variant", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = run_novel_object_evaluation(
        model_dir=args.model_dir,
        variant_ids=[part.strip() for part in args.variant_ids.split(",")],
        trials_per_variant=args.trials_per_variant,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(report["variants"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
