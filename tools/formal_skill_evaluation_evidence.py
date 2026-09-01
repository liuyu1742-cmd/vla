"""Independent acceptance checks for formal OpenVLA model rollout evidence."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


EXPECTED_ACTIONS = [
    "locate(toy)",
    "grasp(toy)",
    "move(storage)",
    "place(toy)",
]
EXPECTED_PHASES = ["locate", "grasp", "move", "place"]
TRANSLATION_LIMITS = {"locate": 1.0, "grasp": 0.25, "move": 0.30, "place": 0.50}
GRIPPER = {"locate": -1.0, "grasp": 1.0, "move": 1.0, "place": -1.0}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_model_evaluation(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != "formal_skill_model_evaluation_v1":
        raise ValueError("unsupported formal model evaluation schema")
    if report.get("evidence_scope") != "formal_skill_model_evaluation":
        raise ValueError("formal model evaluation evidence scope mismatch")
    if report.get("relation_key") != "organizing::toy":
        raise ValueError("formal model evaluation relation mismatch")
    if report.get("split") != "held_out":
        raise ValueError("formal model evaluation must use the held_out split")
    if report.get("canonical_actions") != EXPECTED_ACTIONS:
        raise ValueError("formal model evaluation canonical action sequence mismatch")
    if report.get("success") is not True or report.get("counts_toward_task2_coverage") is not True:
        raise ValueError("formal model evaluation must be successful and count toward coverage")
    predicates = report.get("success_predicates")
    if not isinstance(predicates, Mapping) or not all(
        predicates.get(name) is True
        for name in ("simulator_success", "placed_in_storage", "gripper_released")
    ):
        raise ValueError("all formal task success predicates must be true")
    if report.get("ever_grasped") is not True or report.get("ever_inside") is not True:
        raise ValueError("formal evidence must show both a grasp and cabinet insertion")
    if int(report.get("max_decision_steps", 0)) != 300:
        raise ValueError("formal evaluation must use the 300-decision protocol")
    executed = int(report.get("executed_decision_steps", 0))
    if executed < 1 or executed > 300:
        raise ValueError("formal evaluation executed decision count is invalid")
    for field in ("manifest_sha256", "skill_ir_sha256", "adapter_sha256"):
        if not SHA256.fullmatch(str(report.get(field, ""))):
            raise ValueError(f"formal model evaluation has invalid {field}")

    trajectory = report.get("trajectory")
    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)):
        raise ValueError("formal model evaluation trajectory must be a sequence")
    if len(trajectory) != executed:
        raise ValueError("formal model evaluation trajectory length mismatch")
    covered: set[str] = set()
    for index, step in enumerate(trajectory):
        if not isinstance(step, Mapping):
            raise ValueError(f"trajectory step {index} must be an object")
        phase = str(step.get("phase", ""))
        if phase not in TRANSLATION_LIMITS:
            raise ValueError(f"trajectory step {index} has an unsupported phase")
        action = step.get("guarded_action")
        if not isinstance(action, Sequence) or len(action) != 7:
            raise ValueError(f"trajectory step {index} must contain a seven-dimensional action")
        values = [float(value) for value in action]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"trajectory step {index} contains a non-finite action")
        limit = TRANSLATION_LIMITS[phase] + 1e-6
        if any(abs(value) > limit for value in values[:3]):
            raise ValueError(f"trajectory step {index} exceeds the phase translation limit")
        if any(abs(value) > 1e-7 for value in values[3:6]):
            raise ValueError(f"trajectory step {index} uses an untrained rotation channel")
        if abs(values[6] - GRIPPER[phase]) > 1e-7:
            raise ValueError(f"trajectory step {index} violates the phase gripper guard")
        covered.add(phase)
    missing = [phase for phase in EXPECTED_PHASES if phase not in covered]
    if missing:
        raise ValueError(f"formal model evaluation is missing phases: {missing}")
    return {
        "relation_key": "organizing::toy",
        "seed": int(report["seed"]),
        "split": "held_out",
        "covered_phases": EXPECTED_PHASES,
        "executed_decision_steps": executed,
        "adapter_sha256": report["adapter_sha256"],
    }


__all__ = ["validate_model_evaluation"]
