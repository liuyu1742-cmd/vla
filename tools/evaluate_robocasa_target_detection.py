"""Evaluate whether detections overlap the actual RoboCasa target instance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT.parent / "third_party"
sys.path.insert(0, str(THIRD_PARTY / "robosuite"))
sys.path.insert(0, str(THIRD_PARTY / "robocasa"))
sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402
import robocasa  # noqa: E402,F401
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from tools.collect_robocasa_detection_dataset import CAMERAS, CLASS_NAMES, TASKS  # noqa: E402
from tools.robocasa_auto_label import bbox_from_mask, bbox_iou, mask_from_geom_ids  # noqa: E402


DEFAULT_WEIGHTS = (
    ROOT
    / "models"
    / "coco_household"
    / "yolov8n_16cls_robocasa_synth_v1"
    / "weights"
    / "best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "robocasa_target_detection_eval",
    )
    return parser.parse_args()


def draw_preview(rgb: np.ndarray, gt_bbox, detections: list[dict], path: Path) -> None:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    if gt_bbox is not None:
        draw.rectangle(gt_bbox, outline=(30, 220, 60), width=3)
        draw.text((gt_bbox[0], max(0, gt_bbox[1] - 12)), "TARGET", fill=(30, 220, 60))
    for detection in detections:
        box = detection["bbox_xyxy"]
        draw.rectangle(box, outline=(255, 50, 50), width=2)
        draw.text(
            (box[0], min(rgb.shape[0] - 12, box[1] + 2)),
            f"PRED {detection['confidence']:.2f} IoU {detection['iou']:.2f}",
            fill=(255, 50, 50),
        )
    image.save(path)


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    args.output.mkdir(parents=True, exist_ok=True)
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    detector = YOLO(args.weights)
    records: list[dict] = []

    for task in TASKS:
        print(f"[{task.slug}] creating seed {args.seed}", flush=True)
        env = gym.make(
            f"robocasa/{task.gym_task}",
            split="pretrain",
            seed=args.seed,
            obj_registries=("lightwheel",),
            obj_groups=task.object_group,
            disable_env_checker=True,
        )
        try:
            env.reset(seed=args.seed)
            raw = env.unwrapped.env
            geom_ids = [
                raw.sim.model.geom_name2id(name)
                for name in raw.objects["obj"].visual_geoms
            ]
            for camera in CAMERAS:
                rgb = raw.sim.render(width=256, height=256, camera_name=camera)[::-1]
                segmentation = raw.sim.render(
                    width=256,
                    height=256,
                    camera_name=camera,
                    segmentation=True,
                )[::-1]
                gt_bbox = bbox_from_mask(
                    mask_from_geom_ids(segmentation, geom_ids),
                    min_visible_pixels=20,
                )
                result = detector.predict(
                    source=rgb,
                    conf=args.confidence,
                    imgsz=args.imgsz,
                    device=device,
                    verbose=False,
                )[0]
                detections = []
                if result.boxes is not None:
                    for class_id, confidence, bbox in zip(
                        result.boxes.cls.tolist(),
                        result.boxes.conf.tolist(),
                        result.boxes.xyxy.tolist(),
                    ):
                        if int(class_id) != task.class_id:
                            continue
                        overlap = bbox_iou(tuple(bbox), gt_bbox) if gt_bbox is not None else 0.0
                        detections.append(
                            {
                                "bbox_xyxy": bbox,
                                "confidence": float(confidence),
                                "iou": overlap,
                            }
                        )
                best_iou = max((item["iou"] for item in detections), default=0.0)
                matched = gt_bbox is not None and best_iou >= args.iou_threshold
                preview = args.output / f"{task.slug}_{camera}.png"
                draw_preview(rgb, gt_bbox, detections, preview)
                records.append(
                    {
                        "task": task.slug,
                        "object_group": task.object_group,
                        "target_class_id": task.class_id,
                        "target_class": CLASS_NAMES[task.class_id],
                        "camera": camera,
                        "target_visible": gt_bbox is not None,
                        "target_bbox_xyxy": list(gt_bbox) if gt_bbox is not None else None,
                        "detections": detections,
                        "best_iou": best_iou,
                        "matched": matched,
                        "preview": str(preview.resolve()),
                    }
                )
                print(
                    f"  {camera}: visible={gt_bbox is not None} "
                    f"detections={len(detections)} best_iou={best_iou:.3f} matched={matched}",
                    flush=True,
                )
        finally:
            env.close()

    visible_records = [record for record in records if record["target_visible"]]
    task_pass = {
        task.slug: any(record["matched"] for record in records if record["task"] == task.slug)
        for task in TASKS
    }
    false_positives = sum(
        item["iou"] < args.iou_threshold
        for record in records
        for item in record["detections"]
    )
    report = {
        "format": "robocasa_target_instance_detection_eval_v1",
        "weights": str(args.weights.resolve()),
        "seed": args.seed,
        "confidence_threshold": args.confidence,
        "iou_threshold": args.iou_threshold,
        "summary": {
            "tasks_passed": sum(task_pass.values()),
            "tasks_total": len(task_pass),
            "visible_camera_views": len(visible_records),
            "matched_camera_views": sum(record["matched"] for record in visible_records),
            "visible_view_recall": (
                sum(record["matched"] for record in visible_records) / len(visible_records)
                if visible_records
                else 0.0
            ),
            "false_positive_target_class_detections": false_positives,
            "task_pass": task_pass,
        },
        "records": records,
    }
    report_path = args.output / "target_detection_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"report: {report_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
