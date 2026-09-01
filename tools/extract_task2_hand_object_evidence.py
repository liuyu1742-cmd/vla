"""Extract open-vocabulary object tracks and wrist-to-object contact evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSE_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "task2_human_video_intake"
    / "keypoints"
    / "pose_manifest.json"
)
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "vision" / "yolov8s-worldv2.pt"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "task2_human_video_intake" / "interaction"
)
DEFAULT_CLASSES = ["storage box", "plastic container"]
WRIST_INDICES = (9, 10)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_point_box_distance(
    point: Sequence[float], box: Sequence[float]
) -> float:
    if len(point) != 2 or len(box) != 4:
        raise ValueError("point and box must have lengths two and four")
    x, y = (float(item) for item in point)
    x1, y1, x2, y2 = (float(item) for item in box)
    if x2 < x1 or y2 < y1:
        raise ValueError("box coordinates are inverted")
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(math.hypot(dx, dy))


def summarize_hand_object_sequence(
    detections: Sequence[Mapping[str, Any] | None],
    wrist_track: np.ndarray,
    *,
    wrist_confidence_threshold: float = 0.5,
    near_distance: float = 0.12,
    touching_distance: float = 0.03,
    minimum_detection_ratio: float = 0.5,
    minimum_near_frames: int = 2,
) -> dict[str, Any]:
    wrists = np.asarray(wrist_track, dtype=np.float32)
    if wrists.shape != (len(detections), 2, 3):
        raise ValueError("wrist track must have shape (N,2,3)")
    if touching_distance < 0 or near_distance < touching_distance:
        raise ValueError("distance thresholds are invalid")
    states: list[str] = []
    distances: list[float | None] = []
    confidences: list[float] = []
    for detection, frame_wrists in zip(detections, wrists):
        if detection is None:
            states.append("missing_object")
            distances.append(None)
            continue
        box = detection.get("box")
        confidence = detection.get("confidence")
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)):
            raise ValueError("detection box is missing")
        confidences.append(float(confidence))
        valid = frame_wrists[:, 2] >= wrist_confidence_threshold
        if not valid.any():
            states.append("object_only")
            distances.append(None)
            continue
        distance = min(
            normalized_point_box_distance(point[:2], box)
            for point in frame_wrists[valid]
        )
        distances.append(distance)
        if distance <= touching_distance:
            states.append("touching")
        elif distance <= near_distance:
            states.append("near")
        else:
            states.append("far")
    detected = len(confidences)
    detection_ratio = detected / max(1, len(detections))
    near_frames = sum(state in {"near", "touching"} for state in states)
    finite_distances = [item for item in distances if item is not None]
    ready = (
        detection_ratio >= minimum_detection_ratio
        and near_frames >= minimum_near_frames
    )
    return {
        "contact_evidence_ready": bool(ready),
        "reason": (
            "object_track_and_wrist_proximity_available"
            if ready
            else "need_object_coverage_and_wrist_proximity"
        ),
        "sampled_frames": len(detections),
        "detected_object_frames": detected,
        "object_detection_ratio": float(detection_ratio),
        "near_or_touching_frames": near_frames,
        "touching_frames": states.count("touching"),
        "mean_object_confidence": (
            float(np.mean(confidences)) if confidences else 0.0
        ),
        "minimum_wrist_box_distance": (
            min(finite_distances) if finite_distances else None
        ),
        "contact_states": states,
        "wrist_box_distances": distances,
        "wrist_confidence_threshold": float(wrist_confidence_threshold),
        "near_distance": float(near_distance),
        "touching_distance": float(touching_distance),
        "minimum_detection_ratio": float(minimum_detection_ratio),
        "minimum_near_frames": int(minimum_near_frames),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_video_frames(
    video_path: Path, frame_times: Sequence[float]
) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        for frame_time in frame_times:
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


def _best_detection(result: Any) -> dict[str, Any] | None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    confidences = boxes.conf.detach().cpu().numpy()
    best = int(np.argmax(confidences))
    box = boxes.xyxyn[best].detach().cpu().numpy().tolist()
    class_id = int(boxes.cls[best].detach().cpu().item())
    names = result.names
    return {
        "box": [float(item) for item in box],
        "confidence": float(confidences[best]),
        "class_id": class_id,
        "class_name": str(names[class_id]),
    }


def _render_preview(
    frames: Sequence[np.ndarray],
    frame_times: Sequence[float],
    wrists: np.ndarray,
    detections: Sequence[Mapping[str, Any] | None],
    states: Sequence[str],
    output: Path,
    *,
    wrist_threshold: float,
) -> None:
    import cv2

    tiles: list[np.ndarray] = []
    for frame, frame_time, frame_wrists, detection, state in zip(
        frames, frame_times, wrists, detections, states
    ):
        canvas = frame.copy()
        height, width = canvas.shape[:2]
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
                0.6,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )
        for wrist, color in zip(frame_wrists, ((0, 255, 0), (0, 0, 255))):
            if wrist[2] >= wrist_threshold:
                cv2.circle(
                    canvas,
                    (int(wrist[0] * width), int(wrist[1] * height)),
                    8,
                    color,
                    -1,
                )
        cv2.putText(
            canvas,
            f"{float(frame_time):.2f}s {state}",
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
        raise RuntimeError(f"failed to write interaction preview: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-manifest", type=Path, default=DEFAULT_POSE_MANIFEST)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--wrist-threshold", type=float, default=0.5)
    parser.add_argument("--near-distance", type=float, default=0.12)
    parser.add_argument("--touching-distance", type=float, default=0.03)
    parser.add_argument("--minimum-detection-ratio", type=float, default=0.5)
    parser.add_argument("--minimum-near-frames", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.pose_manifest.is_file():
        raise FileNotFoundError(
            f"pose manifest is missing: {args.pose_manifest}"
        )
    if not args.weights.is_file():
        raise FileNotFoundError(f"YOLO-World weights are missing: {args.weights}")
    pose_manifest = json.loads(args.pose_manifest.read_text(encoding="utf-8"))
    if pose_manifest.get("schema_version") != "task2_human_keypoint_batch_v1":
        raise ValueError("unsupported pose manifest schema")
    records = pose_manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("pose manifest contains no records")

    from ultralytics.models.yolo.model import YOLOWorld

    model = YOLOWorld(str(args.weights))
    model.set_classes(list(args.classes))
    output_records: list[dict[str, Any]] = []
    for index, pose_record in enumerate(records, start=1):
        track_path = Path(str(pose_record["track_path"]))
        if _sha256(track_path) != pose_record["track_sha256"]:
            raise ValueError(f"pose track hash changed: {track_path}")
        track = json.loads(track_path.read_text(encoding="utf-8"))
        if track.get("pose_status") != "available":
            raise ValueError(f"pose track is not available: {track_path}")
        video_path = Path(str(track["video_path"]))
        if _sha256(video_path) != track["video_sha256"]:
            raise ValueError(f"video hash changed: {video_path}")
        frame_times = [float(item) for item in track["frame_times"]]
        keypoints = np.asarray(track["keypoints"], dtype=np.float32)
        wrists = keypoints[:, WRIST_INDICES, :]
        frames = _read_video_frames(video_path, frame_times)
        results = model.predict(
            source=frames,
            conf=args.confidence,
            imgsz=args.image_size,
            device=args.device,
            verbose=False,
        )
        detections = [_best_detection(result) for result in results]
        summary = summarize_hand_object_sequence(
            detections,
            wrists,
            wrist_confidence_threshold=args.wrist_threshold,
            near_distance=args.near_distance,
            touching_distance=args.touching_distance,
            minimum_detection_ratio=args.minimum_detection_ratio,
            minimum_near_frames=args.minimum_near_frames,
        )
        record_id = str(track["record_id"])
        preview_path = (
            args.output_dir / "previews" / f"{record_id}.png"
        )
        _render_preview(
            frames,
            frame_times,
            wrists,
            detections,
            summary["contact_states"],
            preview_path,
            wrist_threshold=args.wrist_threshold,
        )
        payload = {
            "schema_version": "task2_hand_object_evidence_v1",
            "record_id": record_id,
            "episode_ids": list(track["episode_ids"]),
            "phase_role": track["phase_role"],
            "video_path": str(video_path.resolve()),
            "video_sha256": track["video_sha256"],
            "pose_track_path": str(track_path.resolve()),
            "pose_track_sha256": pose_record["track_sha256"],
            "frame_times": frame_times,
            "object_classes": list(args.classes),
            "detections": detections,
            **summary,
            "model": "yolov8s-worldv2",
            "weights_path": str(args.weights.resolve()),
            "weights_sha256": _sha256(args.weights),
            "device": str(args.device),
            "confidence_threshold": float(args.confidence),
            "preview_path": str(preview_path.resolve()),
        }
        output_path = args.output_dir / "tracks" / f"{record_id}.json"
        _write_json_atomic(output_path, payload)
        output_record = {
            "record_id": record_id,
            "episode_ids": list(track["episode_ids"]),
            "phase_role": track["phase_role"],
            "contact_evidence_ready": payload["contact_evidence_ready"],
            "detected_object_frames": payload["detected_object_frames"],
            "near_or_touching_frames": payload["near_or_touching_frames"],
            "interaction_path": str(output_path.resolve()),
            "interaction_sha256": _sha256(output_path),
            "preview_path": str(preview_path.resolve()),
        }
        output_records.append(output_record)
        print(
            f"[{index}/{len(records)}] {record_id} "
            f"ready={output_record['contact_evidence_ready']} "
            f"detected={output_record['detected_object_frames']}/"
            f"{payload['sampled_frames']} "
            f"near={output_record['near_or_touching_frames']}"
        )
    ready = sum(item["contact_evidence_ready"] for item in output_records)
    manifest = {
        "schema_version": "task2_hand_object_batch_v1",
        "relation_key": pose_manifest["relation_key"],
        "pose_manifest_path": str(args.pose_manifest.resolve()),
        "pose_manifest_sha256": _sha256(args.pose_manifest),
        "weights_path": str(args.weights.resolve()),
        "weights_sha256": _sha256(args.weights),
        "object_classes": list(args.classes),
        "segment_count": len(output_records),
        "ready_segment_count": ready,
        "all_segments_ready": ready == len(output_records),
        "records": output_records,
    }
    manifest_path = args.output_dir / "interaction_manifest.json"
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
