"""Extract an auditable COCO-17 human keypoint track from a local video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = (
    Path.home()
    / ".cache"
    / "torch"
    / "hub"
    / "checkpoints"
    / "keypointrcnn_resnet50_fpn_coco-fc266e95.pth"
)
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def select_person_keypoints(
    prediction: Mapping[str, Any],
    *,
    width: int,
    height: int,
    person_threshold: float,
) -> np.ndarray | None:
    if width < 1 or height < 1:
        raise ValueError("frame width and height must be positive")
    scores = np.asarray(prediction.get("scores"), dtype=np.float32)
    keypoints = np.asarray(prediction.get("keypoints"), dtype=np.float32)
    confidences = np.asarray(prediction.get("keypoints_scores"), dtype=np.float32)
    if scores.ndim != 1 or scores.size == 0:
        return None
    index = int(np.argmax(scores))
    if float(scores[index]) < person_threshold:
        return None
    if keypoints.shape != (scores.size, len(COCO_KEYPOINT_NAMES), 3):
        raise ValueError(f"unexpected Keypoint R-CNN keypoint shape: {keypoints.shape}")
    if confidences.shape != (scores.size, len(COCO_KEYPOINT_NAMES)):
        raise ValueError(
            f"unexpected Keypoint R-CNN confidence shape: {confidences.shape}"
        )
    selected = np.empty((len(COCO_KEYPOINT_NAMES), 3), dtype=np.float32)
    selected[:, 0] = np.clip(keypoints[index, :, 0] / width, 0.0, 1.0)
    selected[:, 1] = np.clip(keypoints[index, :, 1] / height, 0.0, 1.0)
    selected[:, 2] = np.clip(confidences[index], 0.0, 1.0)
    return selected


def summarize_keypoint_track(
    track: np.ndarray, *, sampled_frames: int, confidence_threshold: float = 0.5
) -> dict[str, Any]:
    values = np.asarray(track, dtype=np.float32)
    if values.size == 0:
        values = np.empty((0, len(COCO_KEYPOINT_NAMES), 3), dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (len(COCO_KEYPOINT_NAMES), 3):
        raise ValueError("keypoint track must have shape (N,17,3)")
    detected = int(values.shape[0])
    mean_confidence = float(values[..., 2].mean()) if detected else 0.0
    wrist_trajectory = 0.0
    if detected >= 2:
        wrists = values[:, [9, 10], :2]
        wrist_trajectory = float(
            np.linalg.norm(np.diff(wrists, axis=0), axis=2).sum()
        )
    available = (
        detected >= 2
        and mean_confidence >= confidence_threshold
        and not np.allclose(values[..., :2], 0.0)
    )
    return {
        "pose_status": "available" if available else "insufficient",
        "reason": (
            "local_keypointrcnn_coco17_track"
            if available
            else "need_at_least_two_confident_nonzero_person_frames"
        ),
        "sampled_frames": int(sampled_frames),
        "detected_person_frames": detected,
        "mean_keypoint_confidence": mean_confidence,
        "wrist_trajectory_magnitude": wrist_trajectory,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--clip-id")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--stop", type=float)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--person-threshold", type=float, default=0.7)
    parser.add_argument("--keypoint-threshold", type=float, default=0.5)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"input video is missing: {args.video}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"local keypoint checkpoint is missing: {args.weights}")
    if args.samples < 2:
        raise ValueError("at least two frames must be sampled")
    if args.start < 0 or args.stop is not None and args.stop <= args.start:
        raise ValueError("video interval must satisfy 0 <= start < stop")

    import cv2
    import torch
    from torchvision.models.detection import keypointrcnn_resnet50_fpn

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("video has invalid FPS or frame count")
    duration = frame_count / fps
    stop = min(float(args.stop) if args.stop is not None else duration, duration)
    if stop <= args.start:
        capture.release()
        raise ValueError("requested interval is outside the video")

    model = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = torch.device(args.device)
    model.to(device).eval()
    frame_times = np.linspace(args.start, stop, args.samples, endpoint=False)
    detected_times: list[float] = []
    detected_keypoints: list[np.ndarray] = []
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
                    person_threshold=args.person_threshold,
                )
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
    summary = summarize_keypoint_track(
        track,
        sampled_frames=args.samples,
        confidence_threshold=args.keypoint_threshold,
    )
    payload = {
        "format": "human_keypoint_track_v1",
        "clip_id": args.clip_id or args.video.stem,
        "video_path": str(args.video.resolve()),
        "video_sha256": _sha256(args.video),
        "pose_status": summary["pose_status"],
        "reason": summary["reason"],
        "sampled_frames": summary["sampled_frames"],
        "detected_person_frames": summary["detected_person_frames"],
        "mean_keypoint_confidence": summary["mean_keypoint_confidence"],
        "wrist_trajectory_magnitude": summary["wrist_trajectory_magnitude"],
        "keypoint_names": COCO_KEYPOINT_NAMES,
        "frame_times": detected_times,
        "keypoints": track.tolist(),
        "frame_size": list(frame_size) if frame_size else None,
        "coordinates": "normalized_xy_plus_confidence",
        "model": "keypointrcnn_resnet50_fpn_coco17",
        "weights_path": str(args.weights.resolve()),
        "weights_sha256": _sha256(args.weights),
        "device": str(device),
    }
    output = args.output or ROOT / "outputs" / f"{payload['clip_id']}_keypoints.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps({
        "clip_id": payload["clip_id"],
        "pose_status": payload["pose_status"],
        "sampled_frames": payload["sampled_frames"],
        "detected_person_frames": payload["detected_person_frames"],
        "mean_keypoint_confidence": payload["mean_keypoint_confidence"],
        "output": str(output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0 if payload["pose_status"] == "available" else 2


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
