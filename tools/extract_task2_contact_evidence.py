"""Fuse MediaPipe hand tracks with open-vocabulary Task-2 object detections."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tools.extract_task2_hand_object_evidence import (
    normalized_point_box_distance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAND_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "task2_human_video_intake"
    / "hands"
    / "hand_manifest.json"
)
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "vision" / "yolov8s-worldv2.pt"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "task2_human_video_intake" / "contact"
)
DEFAULT_CLASSES = [
    "storage box",
    "plastic container",
    "food storage container",
    "container",
    "plastic tub",
    "storage bin",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_interaction_detection(
    candidates: Sequence[Mapping[str, Any]],
    hand_points: Sequence[Sequence[float]],
    *,
    maximum_box_area: float = 0.85,
    minimum_box_area: float = 0.001,
    proximity_weight: float = 0.5,
) -> dict[str, Any] | None:
    selected: dict[str, Any] | None = None
    selected_score = float("-inf")
    for raw in candidates:
        box = [float(item) for item in raw["box"]]
        width = max(0.0, box[2] - box[0])
        height = max(0.0, box[3] - box[1])
        area = width * height
        if area < minimum_box_area or area > maximum_box_area:
            continue
        distance = (
            min(
                normalized_point_box_distance(point[:2], box)
                for point in hand_points
            )
            if hand_points
            else None
        )
        confidence = float(raw["confidence"])
        score = confidence - (
            proximity_weight * distance if distance is not None else 0.0
        )
        if score > selected_score:
            selected_score = score
            selected = {
                **dict(raw),
                "box": box,
                "box_area": float(area),
                "nearest_hand_landmark_distance": distance,
                "selection_score": float(score),
            }
    return selected


def summarize_interaction_frames(
    frames: Sequence[Mapping[str, Any]],
    *,
    near_distance: float = 0.08,
    touching_distance: float = 0.02,
    minimum_detection_ratio: float = 0.5,
    minimum_contact_frames: int = 1,
) -> dict[str, Any]:
    if touching_distance < 0 or near_distance < touching_distance:
        raise ValueError("interaction distance thresholds are invalid")
    states: list[str] = []
    distances: list[float | None] = []
    confidences: list[float] = []
    for frame in frames:
        detection = frame.get("detection")
        hands = frame.get("hands", [])
        if detection is None:
            states.append("missing_object")
            distances.append(None)
            continue
        confidences.append(float(detection["confidence"]))
        points = [
            landmark
            for hand in hands
            for landmark in hand
        ]
        if not points:
            states.append("object_only")
            distances.append(None)
            continue
        distance = min(
            normalized_point_box_distance(point[:2], detection["box"])
            for point in points
        )
        distances.append(distance)
        if distance <= touching_distance:
            states.append("touching")
        elif distance <= near_distance:
            states.append("near")
        else:
            states.append("far")
    detected = len(confidences)
    ratio = detected / max(1, len(frames))
    near_frames = sum(state in {"near", "touching"} for state in states)
    finite_distances = [item for item in distances if item is not None]
    ready = (
        ratio >= minimum_detection_ratio
        and near_frames >= minimum_contact_frames
    )
    return {
        "interaction_evidence_ready": bool(ready),
        "reason": (
            "mediapipe_hand_and_open_vocabulary_object_contact"
            if ready
            else "need_object_track_and_hand_contact"
        ),
        "sampled_frames": len(frames),
        "detected_object_frames": detected,
        "object_detection_ratio": float(ratio),
        "near_or_touching_frames": near_frames,
        "touching_frames": states.count("touching"),
        "mean_object_confidence": (
            float(np.mean(confidences)) if confidences else 0.0
        ),
        "minimum_hand_box_distance": (
            min(finite_distances) if finite_distances else None
        ),
        "contact_states": states,
        "hand_box_distances": distances,
        "near_distance": float(near_distance),
        "touching_distance": float(touching_distance),
        "minimum_detection_ratio": float(minimum_detection_ratio),
        "minimum_contact_frames": int(minimum_contact_frames),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_frames(video_path: Path, times: Sequence[float]) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        for frame_time in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(frame_time) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"cannot read {video_path} at {float(frame_time):.3f}s"
                )
            frames.append(frame)
    finally:
        capture.release()
    return frames


def _candidates(result: Any) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    normalized = boxes.xyxyn.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    return [
        {
            "box": [float(item) for item in box],
            "confidence": float(confidence),
            "class_id": int(class_id),
            "class_name": str(result.names[int(class_id)]),
        }
        for box, confidence, class_id in zip(
            normalized, confidences, class_ids
        )
    ]


def _render_preview(
    frames: Sequence[np.ndarray],
    frame_records: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    import cv2

    tiles: list[np.ndarray] = []
    for frame, evidence in zip(frames, frame_records):
        canvas = frame.copy()
        height, width = canvas.shape[:2]
        detection = evidence["detection"]
        if detection is not None:
            x1, y1, x2, y2 = detection["box"]
            cv2.rectangle(
                canvas,
                (int(x1 * width), int(y1 * height)),
                (int(x2 * width), int(y2 * height)),
                (255, 180, 0),
                3,
            )
            cv2.putText(
                canvas,
                f"{detection['class_name']} {detection['confidence']:.2f}",
                (int(x1 * width), max(24, int(y1 * height) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )
        for hand_index, hand in enumerate(evidence["hands"]):
            color = (0, 255, 0) if hand_index == 0 else (0, 0, 255)
            for x, y, _ in hand:
                cv2.circle(
                    canvas,
                    (int(x * width), int(y * height)),
                    3,
                    color,
                    -1,
                )
        cv2.putText(
            canvas,
            f"{evidence['timestamp_seconds']:.2f}s {evidence['state']}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tile_width = 320
        tile_height = max(1, round(height * tile_width / width))
        tiles.append(cv2.resize(canvas, (tile_width, tile_height)))
    columns = min(4, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    tile_height, tile_width = tiles[0].shape[:2]
    montage = np.zeros(
        (rows * tile_height, columns * tile_width, 3), dtype=np.uint8
    )
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        montage[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = tile
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), montage):
        raise RuntimeError(f"failed to write contact preview: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-manifest", type=Path, default=DEFAULT_HAND_MANIFEST)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.03)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--maximum-box-area", type=float, default=0.85)
    parser.add_argument("--near-distance", type=float, default=0.08)
    parser.add_argument("--touching-distance", type=float, default=0.02)
    parser.add_argument("--minimum-detection-ratio", type=float, default=0.5)
    parser.add_argument("--minimum-contact-frames", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.hand_manifest.is_file():
        raise FileNotFoundError(f"hand manifest is missing: {args.hand_manifest}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"YOLO-World weights are missing: {args.weights}")
    hand_manifest = json.loads(args.hand_manifest.read_text(encoding="utf-8"))
    if hand_manifest.get("schema_version") != "task2_hand_keypoint_batch_v1":
        raise ValueError("unsupported hand manifest schema")
    source_records = hand_manifest.get("records")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("hand manifest contains no records")

    from ultralytics.models.yolo.model import YOLOWorld

    model = YOLOWorld(str(args.weights))
    model.set_classes(list(args.classes))
    output_records: list[dict[str, Any]] = []
    for index, source in enumerate(source_records, start=1):
        track_path = Path(str(source["track_path"]))
        if _sha256(track_path) != source["track_sha256"]:
            raise ValueError(f"hand track hash changed: {track_path}")
        track = json.loads(track_path.read_text(encoding="utf-8"))
        video_path = Path(str(track["video_path"]))
        if _sha256(video_path) != track["video_sha256"]:
            raise ValueError(f"video hash changed: {video_path}")
        hand_frames = track["frames"]
        times = [float(frame["timestamp_seconds"]) for frame in hand_frames]
        images = _read_frames(video_path, times)
        results = model.predict(
            source=images,
            conf=args.confidence,
            imgsz=args.image_size,
            device=args.device,
            verbose=False,
        )
        fused_frames: list[dict[str, Any]] = []
        for hand_frame, result in zip(hand_frames, results):
            hands = [
                hand["landmarks"] for hand in hand_frame.get("hands", [])
            ]
            points = [landmark for hand in hands for landmark in hand]
            detection = select_interaction_detection(
                _candidates(result),
                points,
                maximum_box_area=args.maximum_box_area,
            )
            fused_frames.append(
                {
                    "timestamp_seconds": float(
                        hand_frame["timestamp_seconds"]
                    ),
                    "hands": hands,
                    "detection": detection,
                }
            )
        summary = summarize_interaction_frames(
            fused_frames,
            near_distance=args.near_distance,
            touching_distance=args.touching_distance,
            minimum_detection_ratio=args.minimum_detection_ratio,
            minimum_contact_frames=args.minimum_contact_frames,
        )
        for frame, state in zip(fused_frames, summary["contact_states"]):
            frame["state"] = state
        object_track_ready = (
            summary["object_detection_ratio"]
            >= args.minimum_detection_ratio
        )
        phase_role = str(track["phase_role"])
        segment_ready = (
            summary["interaction_evidence_ready"]
            if phase_role == "pick"
            else object_track_ready
        )
        record_id = str(track["record_id"])
        preview_path = args.output_dir / "previews" / f"{record_id}.png"
        _render_preview(images, fused_frames, preview_path)
        payload = {
            "schema_version": "task2_contact_evidence_v1",
            "record_id": record_id,
            "episode_ids": list(track["episode_ids"]),
            "phase_role": phase_role,
            "video_path": str(video_path.resolve()),
            "video_sha256": track["video_sha256"],
            "hand_track_path": str(track_path.resolve()),
            "hand_track_sha256": source["track_sha256"],
            "hand_status": track["hand_status"],
            "object_classes": list(args.classes),
            "frames": fused_frames,
            **summary,
            "object_track_ready": object_track_ready,
            "segment_evidence_ready": segment_ready,
            "terminal_evidence_policy": (
                "object_track_plus_exact_task2_place_or_insert_annotation"
                if phase_role != "pick"
                else None
            ),
            "model": "yolov8s-worldv2",
            "weights_path": str(args.weights.resolve()),
            "weights_sha256": _sha256(args.weights),
            "confidence_threshold": float(args.confidence),
            "maximum_box_area": float(args.maximum_box_area),
            "preview_path": str(preview_path.resolve()),
        }
        output_path = args.output_dir / "tracks" / f"{record_id}.json"
        _write_json_atomic(output_path, payload)
        output_record = {
            "record_id": record_id,
            "episode_ids": list(track["episode_ids"]),
            "phase_role": phase_role,
            "hand_status": track["hand_status"],
            "interaction_evidence_ready": payload[
                "interaction_evidence_ready"
            ],
            "object_track_ready": object_track_ready,
            "segment_evidence_ready": segment_ready,
            "detected_object_frames": payload["detected_object_frames"],
            "near_or_touching_frames": payload["near_or_touching_frames"],
            "contact_path": str(output_path.resolve()),
            "contact_sha256": _sha256(output_path),
            "preview_path": str(preview_path.resolve()),
        }
        output_records.append(output_record)
        print(
            f"[{index}/{len(source_records)}] {record_id} "
            f"phase={phase_role} ready={segment_ready} "
            f"object={payload['detected_object_frames']}/{payload['sampled_frames']} "
            f"contact={payload['near_or_touching_frames']}"
        )
    ready = sum(item["segment_evidence_ready"] for item in output_records)
    manifest = {
        "schema_version": "task2_contact_batch_v1",
        "relation_key": hand_manifest["relation_key"],
        "hand_manifest_path": str(args.hand_manifest.resolve()),
        "hand_manifest_sha256": _sha256(args.hand_manifest),
        "weights_path": str(args.weights.resolve()),
        "weights_sha256": _sha256(args.weights),
        "object_classes": list(args.classes),
        "segment_count": len(output_records),
        "ready_segment_count": ready,
        "all_segments_ready": ready == len(output_records),
        "records": output_records,
    }
    manifest_path = args.output_dir / "contact_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "segment_count": len(output_records),
                "ready_segment_count": ready,
                "all_segments_ready": manifest["all_segments_ready"],
                "output": str(manifest_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if manifest["all_segments_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
