"""Record auditable visual evidence for the final experiment 4.2.5 tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.openvla_libero_goal_pure_eval import (
    _configure_libero,
    _environment_for_task,
    _libero_image,
    _load_model,
    _predict_openvla_action,
    _set_seed,
)


@dataclass(frozen=True)
class EvidenceCase:
    case_id: str
    action_group: str
    condition: str
    task_id: int
    episode_index: int
    display_name: str
    object_name: str


EVIDENCE_CASES = (
    EvidenceCase(
        "drawer_nominal",
        "drawer_open",
        "nominal",
        0,
        0,
        "Open middle drawer",
        "middle drawer",
    ),
    EvidenceCase(
        "drawer_generalization",
        "drawer_open",
        "generalization",
        3,
        0,
        "Open top drawer and place bowl",
        "top drawer",
    ),
    EvidenceCase(
        "plate_push_nominal",
        "plate_push",
        "nominal",
        5,
        0,
        "Push plate to target region",
        "plate",
    ),
    EvidenceCase(
        "plate_push_generalization",
        "plate_push",
        "generalization",
        5,
        4,
        "Push held-out plate instance",
        "plate instance",
    ),
    EvidenceCase(
        "bottle_place_nominal",
        "object_place",
        "nominal",
        2,
        0,
        "Place wine bottle on cabinet",
        "wine bottle",
    ),
    EvidenceCase(
        "bottle_place_generalization",
        "object_place",
        "generalization",
        4,
        0,
        "Place bowl on cabinet",
        "bowl",
    ),
)


def select_evidence_frame_indices(frame_count: int) -> tuple[int, int, int]:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    return 0, frame_count // 2, frame_count - 1


def annotate_frame(frame: np.ndarray, title: str, detail: str) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 44), fill=(0, 0, 0, 190))
    font = ImageFont.load_default()
    draw.text((7, 5), title, fill=(255, 255, 255, 255), font=font)
    draw.text((7, 24), detail, fill=(255, 230, 90, 255), font=font)
    return np.asarray(image)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_episode_media(
    case: EvidenceCase,
    frames: list[np.ndarray],
    success: bool,
    decision_steps: int,
    mean_inference_seconds: float | None,
    case_dir: Path,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    status = "SUCCESS" if success else "FAILURE"
    latency_ms = (
        f"{1000 * mean_inference_seconds:.1f} ms"
        if mean_inference_seconds is not None
        else "n/a"
    )
    annotated = [
        annotate_frame(
            frame,
            f"{case.display_name} | {case.condition}",
            f"frame {index:03d} | {status} | inference {latency_ms}",
        )
        for index, frame in enumerate(frames)
    ]
    initial_idx, process_idx, final_idx = select_evidence_frame_indices(
        len(annotated)
    )
    frame_paths = {
        "initial": case_dir / "01_initial.png",
        "process": case_dir / "02_process.png",
        "final": case_dir / "03_final.png",
    }
    Image.fromarray(annotated[initial_idx]).save(frame_paths["initial"])
    Image.fromarray(annotated[process_idx]).save(frame_paths["process"])
    Image.fromarray(annotated[final_idx]).save(frame_paths["final"])

    video_path = case_dir / "rollout.mp4"
    imageio.mimsave(
        video_path,
        annotated,
        fps=15,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    files = {name: str(path.resolve()) for name, path in frame_paths.items()}
    files["video"] = str(video_path.resolve())
    hashes = {
        name: _sha256(Path(path))
        for name, path in files.items()
    }
    return {
        "frame_count": len(frames),
        "selected_frame_indices": [initial_idx, process_idx, final_idx],
        "decision_steps": decision_steps,
        "mean_inference_seconds": mean_inference_seconds,
        "files": files,
        "sha256": hashes,
    }


def _compose_condition_figure(
    cases: list[dict[str, Any]],
    condition: str,
    output_path: Path,
) -> None:
    selected = [case for case in cases if case["condition"] == condition]
    order = {"drawer_open": 0, "plate_push": 1, "object_place": 2}
    selected.sort(key=lambda case: order[case["action_group"]])
    cell = 224
    left = 150
    header = 44
    canvas = Image.new("RGB", (left + cell * 3, header + cell * 3), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, label in enumerate(("Initial", "Process", "Final")):
        draw.text(
            (left + column * cell + 82, 15),
            label,
            fill="black",
            font=font,
        )
    for row, case in enumerate(selected):
        draw.text(
            (8, header + row * cell + 96),
            case["display_name"],
            fill="black",
            font=font,
        )
        for column, phase in enumerate(("initial", "process", "final")):
            image = Image.open(case["media"]["files"][phase]).convert("RGB")
            canvas.paste(image, (left + column * cell, header + row * cell))
    canvas.save(output_path)


def record_evidence(model_dir: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    _configure_libero()
    from libero.libero import benchmark

    _set_seed(seed)
    model, processor = _load_model(model_dir)
    unnorm_key = "libero_goal"
    suite = benchmark.get_benchmark_dict()[unnorm_key]()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for case in EVIDENCE_CASES:
        task = suite.get_task(case.task_id)
        initial_states = suite.get_task_init_states(case.task_id)
        env = _environment_for_task(task)
        env.reset()
        observation = env.set_init_state(initial_states[case.episode_index])
        frames: list[np.ndarray] = []
        inference_seconds: list[float] = []
        decision_steps = 0
        done = False
        error: str | None = None

        for step in range(310):
            try:
                if step < 10:
                    observation, _, done, _ = env.step(
                        [0, 0, 0, 0, 0, 0, -1]
                    )
                    continue
                image = _libero_image(observation)
                frames.append(image.copy())
                started = time.perf_counter()
                action = _predict_openvla_action(
                    model,
                    processor,
                    image,
                    task.language,
                    unnorm_key,
                )
                inference_seconds.append(time.perf_counter() - started)
                decision_steps += 1
                action[-1] = np.sign(2.0 * action[-1] - 1.0)
                action[-1] *= -1.0
                observation, _, done, _ = env.step(action.tolist())
                if done:
                    frames.append(_libero_image(observation))
                    break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                break
        env.close()

        mean_inference = (
            float(np.mean(inference_seconds)) if inference_seconds else None
        )
        media = _save_episode_media(
            case,
            frames,
            bool(done),
            decision_steps,
            mean_inference,
            output_dir / case.case_id,
        )
        result = {
            **asdict(case),
            "task_description": task.language,
            "success": bool(done),
            "error": error,
            "media": media,
        }
        (output_dir / case.case_id / "report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        results.append(result)

    nominal_figure = output_dir / "figure_4_2_5_nominal.png"
    generalization_figure = output_dir / "figure_4_2_5_generalization.png"
    _compose_condition_figure(results, "nominal", nominal_figure)
    _compose_condition_figure(results, "generalization", generalization_figure)
    report = {
        "experiment": "4.2.5 visual evidence remediation",
        "checkpoint": str(model_dir.resolve()),
        "seed": seed,
        "uses_expert_recovery": False,
        "cases": results,
        "figures": {
            "nominal": str(nominal_figure.resolve()),
            "generalization": str(generalization_figure.resolve()),
        },
    }
    (output_dir / "evidence_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    report = record_evidence(args.model_dir, args.output_dir, args.seed)
    print(
        json.dumps(
            [
                {
                    "case_id": case["case_id"],
                    "success": case["success"],
                    "video": case["media"]["files"]["video"],
                }
                for case in report["cases"]
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
