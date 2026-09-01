"""Measure single-axis OpenVLA-to-RoboCasa action responses in simulation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def to_native_action(source: list[float]) -> list[float]:
    if len(source) != 7:
        raise ValueError("source action must have 7 elements")
    return [float(value) for value in source] + [0.0] * 5


def main() -> int:
    import numpy as np
    from robocasa.utils.env_utils import create_env

    probes = []
    for axis in range(3):
        env = create_env("PickPlaceCounterToCabinet", camera_names=["robot0_agentview_left"], seed=0)
        try:
            env.reset()
            arm = env.robots[0].arms[0]
            site_id = env.robots[0].eef_site_id[arm]
            before = np.array(env.sim.data.site_xpos[site_id]).copy()
            action = [0.0] * 7
            action[axis] = 0.1
            env.step(np.asarray(to_native_action(action), dtype=np.float32))
            after = np.array(env.sim.data.site_xpos[site_id]).copy()
            probes.append({"source_axis": axis, "source_value": 0.1, "observed_eef_delta": (after - before).tolist()})
        finally:
            env.close()
    output = ROOT / "outputs" / "robocasa_action_calibration.json"
    output.write_text(json.dumps({"seed": 0, "probes": probes}, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
