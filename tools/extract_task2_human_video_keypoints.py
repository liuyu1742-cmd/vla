"""Batch-extract egocentric upper-body keypoints for Task-2 video intake."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tools.extract_human_keypoints import (
    COCO_KEYPOINT_NAMES,
    DEFAULT_WEIGHTS,
    select_person_keypoints,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTAKE = (
    PROJECT_ROOT
    / "outputs"
    / "task2_human_video_intake"
    / "organizing_storage_box_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "task2_human_video_intake" / "keypoints"
)
WRIST_INDICES = (9, 10)
ARM_EDGES = ((5, 7), (7, 9), (6, 8), (8, 10))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_egocentric_track(
    track: np.ndarray,
    *,
    sampled_frames: int,
    wrist_confidence_threshold: float = 0.5,
    minimum_wrist_frame_ratio: float = 0.5,
    minimum_wrist_trajectory: float = 0.01,
) -> dict[str, Any]:
    values = np.asarray(track, dtype=np.float32)
    if values.size == 0:
        values = np.empty((0, len(COCO_KEYPOINT_NAMES), 3), dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (
        len(COCO_KEYPOINT_NAMES),
        3,
    ):
        raise ValueError("keypoint track must have shape (N,17,3)")
    detected = int(values.shape[0])
    mean_confidence = float(values[..., 2].mean()) if detected else 0.0
    wrist_confidences = (
        values[:, WRIST_INDICES, 2]
        if detected
        else np.empty((0, len(WRIST_INDICES)), dtype=np.float32)
    )
    frames_with_wrist = (
        int(
            (
                wrist_confidences.max(axis=1)
                >= float(wrist_confidence_threshold)
            ).sum()
        )
        if detected
        else 0
    )
    wrist_frame_ratio = frames_with_wrist / max(1, detected)
    wrist_trajectory = 0.0
    if detected >= 2:
        for wrist_offset, wrist_index in enumerate(WRIST_INDICES):
            points = values[:, wrist_index, :2]
            confidence = wrist_confidences[:, wrist_offset]
            pair_valid = (
                confidence[:-1] >= wrist_confidence_threshold
            ) & (confidence[1:] >= wrist_confidence_threshold)
            if pair_valid.any():
                wrist_trajectory += float(
                    np.linalg.norm(np.diff(points, axis=0), axis=1)[
                        pair_valid
                    ].sum()
                )
    coordinates_nonzero = bool(
        detected and not np.allclose(values[:, WRIST_INDICES, :2], 0.0)
    )
    available = (
        detected >= 2
        and wrist_frame_ratio >= minimum_wrist_frame_ratio
        and wrist_trajectory >= minimum_wrist_trajectory
        and coordinates_nonzero
    )
    return {
        "pose_status": "available" if available else "insufficient",
        "reason": (
            "egocentric_confident_moving_wrist_track"
            if available
            else "need_confident_moving_wrist_track"
        ),
        "readiness_profile": "egocentric_upper_body_v1",
        "sampled_frames": int(sampled_frames),
        "detected_person_frames": detected,
        "mean_keypoint_confidence": mean_confidence,
        "frames_with_confident_wrist": frames_with_wrist,
        "confident_wrist_frame_ratio": float(wrist_frame_ratio),
        "wrist_trajectory_magnitude": float(wrist_trajectory),
        "wrist_confidence_threshold": float(wrist_confidence_threshold),
        "minimum_wrist_frame_ratio": float(minimum_wrist_frame_ratio),
        "minimum_wrist_trajectory": float(minimum_wrist_trajectory),
    }


def build_segment_jobs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != "task2_human_video_intake_v1":
        raise ValueError("unsupported Task-2 human-video intake schema")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("intake manifest must contain episodes")
    jobs: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("episode entries must be objects")
        episode_id = str(episode.get("episode_id", "")).strip()
        segments = episode.get("segments")
        if not episode_id or not isinstance(segments, list):
            raise ValueError("episode is missing episode_id or segments")
        for segment in segments:
            if not isinstance(segment, Mapping):
                raise ValueError("segment entries must be objects")
            record_id = str(segment.get("record_id", "")).strip()
            job = {
                "record_id": record_id,
                "phase_role": str(segment.get("phase_role", "")).strip(),
                "video_path": str(segment.get("video_path", "")).strip(),
                "video_sha256": str(segment.get("video_sha256", "")).strip(),
                "episode_ids": [episode_id],
            }
            if (
                not record_id
                or not job["phase_role"]
                or not job["video_path"]
                or len(job["video_sha256"]) != 64
            ):
                raise ValueError("segment is missing required fingerprint fields")
            previous = jobs.get(record_id)
            if previous is None:
                jobs[record_id] = job
                continue
            comparable = ("phase_role", "video_path", "video_sha256")
            if any(previous[key] != job[key] for key in comparable):
                raise ValueError(
                    f"conflicting duplicate segment fingerprint: {record_id}"
                )
            previous["episode_ids"].append(episode_id)
    return list(jobs.values())


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _render_preview(
    samples: list[tuple[float, np.ndarray, np.ndarray | None]],
    output: Path,
    *,
    confidence_threshold: float,
) -> None:
    import cv2

    tiles: list[np.ndarray] = []
    for frame_time, frame, keypoints in samples:
        canvas = frame.copy()
        height, width = canvas.shape[:2]
        if keypoints is not None:
            for start, stop in ARM_EDGES:
                if (
                    keypoints[start, 2] >= confidence_threshold
                    and keypoints[stop, 2] >= confidence_threshold
                ):
                    first = (
                        int(keypoints[start, 0] * width),
                        int(keypoints[start, 1] * height),
                    )
                    second = (
                        int(keypoints[stop, 0] * width),
                        int(keypoints[stop, 1] * height),
                    )
                    cv2.line(canvas, first, second, (0, 220, 255), 2)
            for wrist_index, color in zip(
                WRIST_INDICES, ((0, 255, 0), (0, 0, 255))
            ):
                if keypoints[wrist_index, 2] >= confidence_threshold:
                    point = (
                        int(keypoints[wrist_index, 0] * width),
                        int(keypoints[wrist_index, 1] * height),
                    )
                    cv2.circle(canvas, point, 8, color, -1)
        cv2.putText(
            canvas,
            f"{frame_time:.2f}s",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        target_width = 320
        target_height = max(1, round(height * target_width / width))
        tiles.append(cv2.resize(canvas, (target_width, target_height)))
    if not tiles:
        return
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
        raise RuntimeError(f"failed to write preview: {output}")


def _extract_one(
    *,
    model: Any,
    torch: Any,
    cv2: Any,
    job: Mapping[str, Any],
    samples: int,
    device: Any,
    person_threshold: float,
    wrist_confidence_threshold: float,
    minimum_wrist_frame_ratio: float,
    minimum_wrist_trajectory: float,
    output_dir: Path,
    weights_path: Path,
    weights_sha256: str,
) -> dict[str, Any]:
    video_path = Path(str(job["video_path"]))
    if not video_path.is_file():
        raise FileNotFoundError(f"intake video is missing: {video_path}")
    actual_video_hash = _sha256(video_path)
    if actual_video_hash != job["video_sha256"]:
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
    frame_times = np.linspace(0.0, duration, samples, endpoint=False)
    detected_times: list[float] = []
    detected_keypoints: list[np.ndarray] = []
    preview_samples: list[tuple[float, np.ndarray, np.ndarray | None]] = []
    frame_size: tuple[int, int] | None = None
    try:
        with torch.inference_mode():
            for frame_time in frame_times:
                capture.set(cv2.CAP_PROP_POS_MSEC, float(frame_time) * 1000.0)
                ok, bgr = capture.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                height, width = rgb.shape[:2]
                frame_size = (width, height)
                tensor = (
                    torch.from_numpy(np.ascontiguousarray(rgb))
                    .permute(2, 0, 1)
                    .float()
                    .div(255.0)
                    .to(device)
                )
                output = model([tensor])[0]
                prediction = {
                    key: value.detach().cpu().numpy()
                    for key, value in output.items()
                    if key in {"scores", "keypoints", "keypoints_scores"}
                }
                selected = select_person_keypoints(
                    prediction,
                    width=width,
                    height=height,
                    person_threshold=person_threshold,
                )
                preview_samples.append((float(frame_time), bgr, selected))
                if selected is not None:
                    detected_times.append(float(frame_time))
                    detected_keypoints.append(selected)
    finally:
        capture.release()
    track = (
        np.stack(detected_keypoints)
        if detected_keypoints
        else np.empty((0, len(COCO_KEYPOINT_NAMES), 3), dtype=np.float32)
    )
    summary = summarize_egocentric_track(
        track,
        sampled_frames=samples,
        wrist_confidence_threshold=wrist_confidence_threshold,
        minimum_wrist_frame_ratio=minimum_wrist_frame_ratio,
        minimum_wrist_trajectory=minimum_wrist_trajectory,
    )
    record_id = str(job["record_id"])
    preview_path = output_dir / "previews" / f"{record_id}.png"
    _render_preview(
        preview_samples,
        preview_path,
        confidence_threshold=wrist_confidence_threshold,
    )
    payload = {
        "format": "human_keypoint_track_v2_egocentric",
        "record_id": record_id,
        "episode_ids": list(job["episode_ids"]),
        "phase_role": job["phase_role"],
        "video_path": str(video_path.resolve()),
        "video_sha256": actual_video_hash,
        **summary,
        "keypoint_names": COCO_KEYPOINT_NAMES,
        "frame_times": detected_times,
        "keypoints": track.tolist(),
        "frame_size": list(frame_size) if frame_size else None,
        "coordinates": "normalized_xy_plus_confidence",
        "model": "keypointrcnn_resnet50_fpn_coco17",
        "weights_path": str(weights_path.resolve()),
        "weights_sha256": weights_sha256,
        "device": str(device),
        "person_threshold": float(person_threshold),
        "preview_path": str(preview_path.resolve()),
    }
    output_path = output_dir / "tracks" / f"{record_id}.json"
    _write_json_atomic(output_path, payload)
    return {
        "record_id": record_id,
        "episode_ids": list(job["episode_ids"]),
        "phase_role": job["phase_role"],
        "pose_status": payload["pose_status"],
        "detected_person_frames": payload["detected_person_frames"],
        "frames_with_confident_wrist": payload["frames_with_confident_wrist"],
        "confident_wrist_frame_ratio": payload["confident_wrist_frame_ratio"],
        "wrist_trajectory_magnitude": payload["wrist_trajectory_magnitude"],
        "track_path": str(output_path.resolve()),
        "track_sha256": _sha256(output_path),
        "preview_path": str(preview_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, default=DEFAULT_INTAKE)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--person-threshold", type=float, default=0.7)
    parser.add_argument("--wrist-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-wrist-frame-ratio", type=float, default=0.5)
    parser.add_argument("--minimum-wrist-trajectory", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 2:
        raise ValueError("samples must be at least two")
    if not args.intake.is_file():
        raise FileNotFoundError(f"intake manifest is missing: {args.intake}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"local keypoint checkpoint is missing: {args.weights}")
    intake = json.loads(args.intake.read_text(encoding="utf-8"))
    jobs = build_segment_jobs(intake)
    if not jobs:
        raise ValueError("intake manifest contains no segment jobs")

    import cv2
    import torch
    from torchvision.models.detection import keypointrcnn_resnet50_fpn

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    weights_sha256 = _sha256(args.weights)
    model = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    records: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        record = _extract_one(
            model=model,
            torch=torch,
            cv2=cv2,
            job=job,
            samples=args.samples,
            device=device,
            person_threshold=args.person_threshold,
            wrist_confidence_threshold=args.wrist_threshold,
            minimum_wrist_frame_ratio=args.minimum_wrist_frame_ratio,
            minimum_wrist_trajectory=args.minimum_wrist_trajectory,
            output_dir=args.output_dir,
            weights_path=args.weights,
            weights_sha256=weights_sha256,
        )
        records.append(record)
        print(
            f"[{index}/{len(jobs)}] {record['record_id']} "
            f"status={record['pose_status']} "
            f"wrist_frames={record['frames_with_confident_wrist']}/{args.samples}"
        )
    available = sum(item["pose_status"] == "available" for item in records)
    summary = {
        "schema_version": "task2_human_keypoint_batch_v1",
        "relation_key": intake["relation_key"],
        "intake_path": str(args.intake.resolve()),
        "intake_sha256": _sha256(args.intake),
        "weights_path": str(args.weights.resolve()),
        "weights_sha256": weights_sha256,
        "device": str(device),
        "sampled_frames_per_segment": args.samples,
        "segment_count": len(records),
        "available_segment_count": available,
        "all_segments_available": available == len(records),
        "records": records,
    }
    summary_path = args.output_dir / "pose_manifest.json"
    _write_json_atomic(summary_path, summary)
    print(
        json.dumps(
            {
                "segment_count": len(records),
                "available_segment_count": available,
                "all_segments_available": summary["all_segments_available"],
                "output": str(summary_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if summary["all_segments_available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
