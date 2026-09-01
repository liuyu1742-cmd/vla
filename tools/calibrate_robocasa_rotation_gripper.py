"""Probe RoboCasa rotation and gripper dimensions of the 12D native action."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def native_probe(index: int, value: float) -> list[float]:
    if not 0 <= index < 7:
        raise ValueError("OpenVLA probe index must be in [0, 6]")
    action = [0.0] * 12
    action[index] = float(value)
    return action


def main() -> int:
    import numpy as np
    from robocasa.utils.env_utils import create_env

    probes = []
    for index, value, label in [(3, 0.1, "rotation_x"), (4, 0.1, "rotation_y"), (5, 0.1, "rotation_z"), (6, -1.0, "gripper_close"), (6, 1.0, "gripper_open")]:
        env = create_env("PickPlaceCounterToCabinet", camera_names=["robot0_agentview_left"], seed=0)
        try:
            env.reset()
            arm = env.robots[0].arms[0]
            site_id = env.robots[0].eef_site_id[arm]
            before = np.array(env.sim.data.site_xmat[site_id]).reshape(3, 3).copy()
            env.step(np.asarray(native_probe(index, value), dtype=np.float32))
            after = np.array(env.sim.data.site_xmat[site_id]).reshape(3, 3).copy()
            probes.append({"label": label, "source_index": index, "source_value": value, "eef_rotation_delta_matrix": (after @ before.T).tolist()})
        finally:
            env.close()
    output = ROOT / "outputs" / "robocasa_rotation_gripper_calibration.json"
    output.write_text(json.dumps({"seed": 0, "probes": probes}, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
