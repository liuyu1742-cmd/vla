"""Create an auditable VLA trajectory manifest from local RoboCasa capability."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "datasets" / "robocasa_vla"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
ROBOCASA_SOURCE = ROOT / "third_party" / "robocasa"


def prepare_local_robocasa_source() -> bool:
    """Expose checked-out RoboCasa source without requiring setup.py develop."""
    if not (ROBOCASA_SOURCE / "robocasa").is_dir():
        return False
    source = str(ROBOCASA_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)
    return True


def validate_episode(episode: dict[str, Any]) -> list[str]:
    """Validate the minimum observation-action contract for VLA supervision."""
    errors = []
    required = ("instruction", "task_id", "object", "observation_paths", "actions", "action_dim", "control_hz", "success", "seed")
    errors.extend(f"missing_{key}" for key in required if key not in episode)
    if len(episode.get("observation_paths", [])) != len(episode.get("actions", [])):
        errors.append("observation_action_count_mismatch")
    action_dim = episode.get("action_dim")
    if action_dim is not None and any(len(action) != action_dim for action in episode.get("actions", [])):
        errors.append("inconsistent_action_dimension")
    return errors


def inspect_simulator() -> dict[str, Any]:
    source_available = prepare_local_robocasa_source()
    robocasa = importlib.util.find_spec("robocasa") is not None
    mujoco = importlib.util.find_spec("mujoco") is not None
    status = "ready_for_export" if robocasa and mujoco else "simulator_unavailable"
    return {
        "robocasa_source_available": source_available,
        "robocasa_importable": robocasa,
        "mujoco_importable": mujoco,
        "status": status,
    }


def build_manifest() -> dict[str, Any]:
    simulator = inspect_simulator()
    return {
        "format": "robocasa_vla_manifest_v1",
        "simulator": simulator,
        "episodes": [],
        "episode_count": 0,
        "note": "No synthetic trajectory is created when the local simulator is not verified.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = args.output.parent / "README.md"
    readme.write_text("# RoboCasa VLA 轨迹\n\n仅存放本机模拟器实际导出的机器人观测—动作轨迹；当前清单状态见 `manifest.json`。\n", encoding="utf-8")
    print(json.dumps(manifest["simulator"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
