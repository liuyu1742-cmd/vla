"""Collect YOLO detection labels from RoboCasa's MuJoCo segmentation buffer."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT.parent / "third_party"
sys.path.insert(0, str(THIRD_PARTY / "robosuite"))
sys.path.insert(0, str(THIRD_PARTY / "robocasa"))
sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402
import robocasa  # noqa: E402,F401 - registers Gym environments

from tools.robocasa_auto_label import (  # noqa: E402
    bbox_from_mask,
    bbox_to_yolo,
    mask_from_geom_ids,
    split_for_index,
)


CLASS_NAMES = {
    0: "bed",
    1: "book",
    2: "bottle",
    3: "bowl",
    4: "cup",
    5: "fork",
    6: "knife",
    7: "laptop",
    8: "microwave",
    9: "oven",
    10: "refrigerator",
    11: "remote_control",
    12: "sink",
    13: "spoon",
    14: "toilet",
    15: "wine_glass",
}

CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


@dataclass(frozen=True)
class TaskSpec:
    slug: str
    gym_task: str
    object_group: str
    class_id: int


TASKS = (
    TaskSpec("water_cup", "PickPlaceCounterToCabinet", "glass_cup", 4),
    TaskSpec("household_organizing", "PickPlaceCounterToCabinet", "salt_and_pepper_shaker", 2),
    TaskSpec("dish_loading", "PickPlaceCounterToSink", "glass_cup", 4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds-per-task", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--min-visible-pixels", type=int, default=20)
    parser.add_argument(
        "--resume-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip complete image/label/preview triplets already present in the output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets" / "robocasa_detection_v1",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.seeds_per_task < 2:
        raise ValueError("--seeds-per-task must be at least 2")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32")
    if args.min_visible_pixels < 1:
        raise ValueError("--min-visible-pixels must be positive")


def visual_geom_ids(raw_env) -> list[int]:
    names = raw_env.objects["obj"].visual_geoms
    return [raw_env.sim.model.geom_name2id(name) for name in names]


def write_yaml(output: Path) -> Path:
    path = output / "dataset.yaml"
    names = "\n".join(f"  {index}: {name}" for index, name in CLASS_NAMES.items())
    path.write_text(
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )
    return path


def save_preview(rgb: np.ndarray, bbox, path: Path, label: str) -> None:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(255, 40, 40), width=3)
        draw.text((x1, max(0, y1 - 12)), label, fill=(255, 40, 40))
    image.save(path)


def collect(args: argparse.Namespace) -> dict:
    validate_args(args)
    output = args.output.resolve()
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "previews").mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for task in TASKS:
        for index in range(args.seeds_per_task):
            seed = args.seed_start + index
            split = split_for_index(index, args.seeds_per_task, args.val_fraction)
            expected = []
            for camera in CAMERAS:
                stem = f"{task.slug}_seed{seed:04d}_{camera}"
                expected.append(
                    (
                        camera,
                        output / "images" / split / f"{stem}.png",
                        output / "labels" / split / f"{stem}.txt",
                        output / "previews" / f"{stem}.png",
                    )
                )
            if args.resume_existing and all(
                image_path.exists() and label_path.exists() and preview_path.exists()
                for _, image_path, label_path, preview_path in expected
            ):
                print(f"[{task.slug}] seed={seed} split={split} already complete", flush=True)
                for camera, image_path, label_path, preview_path in expected:
                    positive = bool(label_path.read_text(encoding="utf-8").strip())
                    records.append(
                        {
                            "task": task.slug,
                            "gym_task": task.gym_task,
                            "object_group": task.object_group,
                            "class_id": task.class_id,
                            "class_name": CLASS_NAMES[task.class_id],
                            "seed": seed,
                            "split": split,
                            "camera": camera,
                            "visual_geom_ids": None,
                            "visible_pixels": None,
                            "bbox_xyxy": "stored_in_yolo_label" if positive else None,
                            "positive": positive,
                            "image": str(image_path),
                            "label": str(label_path),
                            "preview": str(preview_path),
                            "collection_status": "reused_existing",
                        }
                    )
                continue
            print(f"[{task.slug}] seed={seed} split={split}", flush=True)
            env = gym.make(
                f"robocasa/{task.gym_task}",
                split="pretrain",
                seed=seed,
                obj_registries=("lightwheel",),
                obj_groups=task.object_group,
                disable_env_checker=True,
            )
            try:
                env.reset(seed=seed)
                raw = env.unwrapped.env
                geom_ids = visual_geom_ids(raw)
                for camera in CAMERAS:
                    rgb = raw.sim.render(
                        width=args.image_size,
                        height=args.image_size,
                        camera_name=camera,
                    )[::-1]
                    segmentation = raw.sim.render(
                        width=args.image_size,
                        height=args.image_size,
                        camera_name=camera,
                        segmentation=True,
                    )[::-1]
                    mask = mask_from_geom_ids(segmentation, geom_ids)
                    visible_pixels = int(mask.sum())
                    bbox = bbox_from_mask(mask, args.min_visible_pixels)
                    stem = f"{task.slug}_seed{seed:04d}_{camera}"
                    image_path = output / "images" / split / f"{stem}.png"
                    label_path = output / "labels" / split / f"{stem}.txt"
                    Image.fromarray(rgb).save(image_path)
                    if bbox is None:
                        label_path.write_text("", encoding="utf-8")
                    else:
                        row = bbox_to_yolo(
                            task.class_id,
                            bbox,
                            image_width=args.image_size,
                            image_height=args.image_size,
                        )
                        label_path.write_text(
                            f"{row[0]} {row[1]:.8f} {row[2]:.8f} {row[3]:.8f} {row[4]:.8f}\n",
                            encoding="utf-8",
                        )
                    preview_path = output / "previews" / f"{stem}.png"
                    save_preview(rgb, bbox, preview_path, CLASS_NAMES[task.class_id])
                    records.append(
                        {
                            "task": task.slug,
                            "gym_task": task.gym_task,
                            "object_group": task.object_group,
                            "class_id": task.class_id,
                            "class_name": CLASS_NAMES[task.class_id],
                            "seed": seed,
                            "split": split,
                            "camera": camera,
                            "visual_geom_ids": geom_ids,
                            "visible_pixels": visible_pixels,
                            "bbox_xyxy": list(bbox) if bbox is not None else None,
                            "positive": bbox is not None,
                            "image": str(image_path),
                            "label": str(label_path),
                            "preview": str(preview_path),
                            "collection_status": "collected",
                        }
                    )
            finally:
                env.close()

    yaml_path = write_yaml(output)
    visible = sum(record["positive"] for record in records)
    report = {
        "format": "robocasa_mujoco_yolo_detection_v1",
        "output": str(output),
        "dataset_yaml": str(yaml_path),
        "seed_split_policy": "last round(N * val_fraction) seeds per task are validation",
        "settings": {
            "seeds_per_task": args.seeds_per_task,
            "seed_start": args.seed_start,
            "val_fraction": args.val_fraction,
            "image_size": args.image_size,
            "min_visible_pixels": args.min_visible_pixels,
            "cameras": list(CAMERAS),
        },
        "summary": {
            "scenes": len(TASKS) * args.seeds_per_task,
            "images": len(records),
            "positive_images": visible,
            "negative_images": len(records) - visible,
        },
        "records": records,
    }
    report_path = output / "collection_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"dataset: {yaml_path}", flush=True)
    print(f"report: {report_path}", flush=True)
    return report


def main() -> None:
    collect(parse_args())


if __name__ == "__main__":
    main()
