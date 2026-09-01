"""Extract true 21-point hand tracks from Task-2 egocentric video intake."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tools.extract_task2_human_video_keypoints import build_segment_jobs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTAKE = (
    PROJECT_ROOT
    / "outputs"
    / "task2_human_video_intake"
    / "organizing_storage_box_manifest.json"
)
DEFAULT_MODEL = PROJECT_ROOT / "models" / "vision" / "hand_landmarker.task"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "task2_human_video_intake" / "hands"
)
HAND_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]
HAND_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_hand_frames(
    frames: Sequence[Mapping[str, Any]],
    *,
    minimum_hand_frame_ratio: float = 0.4,
    minimum_wrist_trajectory: float = 0.02,
) -> dict[str, Any]:
    if len(frames) < 2:
        return {
            "hand_status": "insufficient",
            "reason": "need_at_least_two_sampled_frames",
            "sampled_frames": len(frames),
            "detected_hand_frames": 0,
            "hand_frame_ratio": 0.0,
            "maximum_hands_in_frame": 0,
            "mean_handedness_score": 0.0,
            "wrist_trajectory_magnitude": 0.0,
            "minimum_hand_frame_ratio": float(minimum_hand_frame_ratio),
            "minimum_wrist_trajectory": float(minimum_wrist_trajectory),
        }
    times = [float(frame["timestamp_seconds"]) for frame in frames]
    if any(second <= first for first, second in zip(times, times[1:])):
        raise ValueError("hand frame timestamps must be strictly increasing")
    detected = sum(bool(frame.get("hands")) for frame in frames)
    maximum_hands = max(len(frame.get("hands", [])) for frame in frames)
    scores: list[float] = []
    wrists_by_side: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for frame in frames:
        for hand in frame.get("hands", []):
            landmarks = hand.get("landmarks")
            if not isinstance(landmarks, list) or len(landmarks) != 21:
                raise ValueError("each detected hand must contain 21 landmarks")
            wrist = np.asarray(landmarks[0][:2], dtype=np.float64)
            if wrist.shape != (2,) or not np.isfinite(wrist).all():
                raise ValueError("hand wrist landmark is invalid")
            side = str(hand.get("handedness", "Unknown"))
            wrists_by_side[side].append((float(frame["timestamp_seconds"]), wrist))
            scores.append(float(hand.get("handedness_score", 0.0)))
    trajectory = 0.0
    for side_track in wrists_by_side.values():
        for (_, first), (_, second) in zip(side_track, side_track[1:]):
            trajectory += float(np.linalg.norm(second - first))
    ratio = detected / len(frames)
    available = (
        ratio >= minimum_hand_frame_ratio
        and trajectory >= minimum_wrist_trajectory
    )
    return {
        "hand_status": "available" if available else "insufficient",
        "reason": (
            "mediapipe_21_point_moving_hand_track"
            if available
            else "need_sufficient_moving_hand_track"
        ),
        "sampled_frames": len(frames),
        "detected_hand_frames": detected,
        "hand_frame_ratio": float(ratio),
        "maximum_hands_in_frame": maximum_hands,
        "mean_handedness_score": float(np.mean(scores)) if scores else 0.0,
        "wrist_trajectory_magnitude": float(trajectory),
        "minimum_hand_frame_ratio": float(minimum_hand_frame_ratio),
        "minimum_wrist_trajectory": float(minimum_wrist_trajectory),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _render_preview(
    samples: Sequence[tuple[float, np.ndarray, list[dict[str, Any]]]],
    output: Path,
) -> None:
    import cv2

    tiles: list[np.ndarray] = []
    colors = ((0, 255, 0), (0, 0, 255))
    for frame_time, frame, hands in samples:
        canvas = frame.copy()
        height, width = canvas.shape[:2]
        for hand_index, hand in enumerate(hands):
            landmarks = hand["landmarks"]
            color = colors[hand_index % len(colors)]
            for start, stop in HAND_EDGES:
                first = (
                    int(landmarks[start][0] * width),
                    int(landmarks[start][1] * height),
                )
                second = (
                    int(landmarks[stop][0] * width),
                    int(landmarks[stop][1] * height),
                )
                cv2.line(canvas, first, second, color, 2)
            for x, y, _ in landmarks:
                cv2.circle(canvas, (int(x * width), int(y * height)), 3, color, -1)
        cv2.putText(
            canvas,
            f"{frame_time:.2f}s hands={len(hands)}",
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
        raise RuntimeError(f"failed to write hand preview: {output}")


def _extract_hands(result: Any) -> list[dict[str, Any]]:
    hands: list[dict[str, Any]] = []
    for index, landmarks in enumerate(result.hand_landmarks):
        category = result.handedness[index][0]
        hands.append(
            {
                "handedness": str(category.category_name),
                "handedness_score": float(category.score),
                "landmarks": [
                    [float(point.x), float(point.y), float(point.z)]
                    for point in landmarks
                ],
            }
        )
    return hands


def _extract_one(
    detector: Any,
    mp: Any,
    cv2: Any,
    job: Mapping[str, Any],
    *,
    samples: int,
    output_dir: Path,
    minimum_hand_frame_ratio: float,
    minimum_wrist_trajectory: float,
    model_path: Path,
    model_sha256: str,
) -> dict[str, Any]:
    video_path = Path(str(job["video_path"]))
    if _sha256(video_path) != job["video_sha256"]:
        raise ValueError(f"intake video hash changed: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"video has invalid FPS or frame count: {video_path}")
    duration = frame_count / fps
    requested_times = np.linspace(0.0, duration, samples, endpoint=False)
    frame_records: list[dict[str, Any]] = []
    preview_samples: list[tuple[float, np.ndarray, list[dict[str, Any]]]] = []
    try:
        for frame_time in requested_times:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(frame_time) * 1000.0)
            ok, bgr = capture.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(round(float(frame_time) * 1000.0))
            hands = _extract_hands(
                detector.detect_for_video(image, timestamp_ms)
            )
            frame_record = {
                "timestamp_seconds": float(frame_time),
                "hands": hands,
            }
            frame_records.append(frame_record)
            preview_samples.append((float(frame_time), bgr, hands))
    finally:
        capture.release()
    summary = summarize_hand_frames(
        frame_records,
        minimum_hand_frame_ratio=minimum_hand_frame_ratio,
        minimum_wrist_trajectory=minimum_wrist_trajectory,
    )
    record_id = str(job["record_id"])
    preview_path = output_dir / "previews" / f"{record_id}.png"
    _render_preview(preview_samples, preview_path)
    payload = {
        "schema_version": "task2_hand_keypoint_track_v1",
        "record_id": record_id,
        "episode_ids": list(job["episode_ids"]),
        "phase_role": job["phase_role"],
        "video_path": str(video_path.resolve()),
        "video_sha256": job["video_sha256"],
        **summary,
        "landmark_names": HAND_LANDMARK_NAMES,
        "coordinates": "normalized_xyz",
        "frames": frame_records,
        "model": "mediapipe_hand_landmarker_float16",
        "model_path": str(model_path.resolve()),
        "model_sha256": model_sha256,
        "preview_path": str(preview_path.resolve()),
    }
    track_path = output_dir / "tracks" / f"{record_id}.json"
    _write_json_atomic(track_path, payload)
    return {
        "record_id": record_id,
        "episode_ids": list(job["episode_ids"]),
        "phase_role": job["phase_role"],
        "hand_status": payload["hand_status"],
        "detected_hand_frames": payload["detected_hand_frames"],
        "hand_frame_ratio": payload["hand_frame_ratio"],
        "wrist_trajectory_magnitude": payload["wrist_trajectory_magnitude"],
        "track_path": str(track_path.resolve()),
        "track_sha256": _sha256(track_path),
        "preview_path": str(preview_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, default=DEFAULT_INTAKE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--minimum-detection-confidence", type=float, default=0.3)
    parser.add_argument("--minimum-hand-frame-ratio", type=float, default=0.4)
    parser.add_argument("--minimum-wrist-trajectory", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 2:
        raise ValueError("samples must be at least two")
    if not args.intake.is_file():
        raise FileNotFoundError(f"intake manifest is missing: {args.intake}")
    if not args.model.is_file():
        raise FileNotFoundError(f"hand model is missing: {args.model}")
    intake = json.loads(args.intake.read_text(encoding="utf-8"))
    jobs = build_segment_jobs(intake)

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_sha256 = _sha256(args.model)
    records: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(args.model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=args.num_hands,
            min_hand_detection_confidence=args.minimum_detection_confidence,
            min_hand_presence_confidence=args.minimum_detection_confidence,
            min_tracking_confidence=args.minimum_detection_confidence,
        )
        with vision.HandLandmarker.create_from_options(options) as detector:
            record = _extract_one(
                detector,
                mp,
                cv2,
                job,
                samples=args.samples,
                output_dir=args.output_dir,
                minimum_hand_frame_ratio=args.minimum_hand_frame_ratio,
                minimum_wrist_trajectory=args.minimum_wrist_trajectory,
                model_path=args.model,
                model_sha256=model_sha256,
            )
        records.append(record)
        print(
            f"[{index}/{len(jobs)}] {record['record_id']} "
            f"status={record['hand_status']} "
            f"hand_frames={record['detected_hand_frames']}/{args.samples}"
        )
    available = sum(item["hand_status"] == "available" for item in records)
    manifest = {
        "schema_version": "task2_hand_keypoint_batch_v1",
        "relation_key": intake["relation_key"],
        "intake_path": str(args.intake.resolve()),
        "intake_sha256": _sha256(args.intake),
        "model_path": str(args.model.resolve()),
        "model_sha256": model_sha256,
        "sampled_frames_per_segment": args.samples,
        "segment_count": len(records),
        "available_segment_count": available,
        "all_segments_available": available == len(records),
        "records": records,
    }
    manifest_path = args.output_dir / "hand_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "segment_count": len(records),
                "available_segment_count": available,
                "all_segments_available": manifest["all_segments_available"],
                "output": str(manifest_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if manifest["all_segments_available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

