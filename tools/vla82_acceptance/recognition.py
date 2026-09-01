"""Record honest OpenVLA target-conditioned visual response evidence."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from tools.openvla_tcp_client import predict


def validate_policy_response(
    *,
    expected_object: str,
    image_path: Path,
    instruction: str,
    action: list[float],
    latency_seconds: float,
    endpoint: str,
) -> dict[str, Any]:
    errors: list[str] = []
    if not image_path.is_file():
        errors.append("image_missing")
    if not instruction.strip():
        errors.append("instruction_empty")
    if expected_object not in instruction:
        errors.append("expected_object_absent_from_instruction")
    if len(action) != 7:
        errors.append("invalid_action_shape")
    elif not np.isfinite(np.asarray(action, dtype=np.float64)).all():
        errors.append("nonfinite_action")
    return {
        "schema_version": "vla82_openvla_target_conditioned_response_v1",
        "recognition_kind": "openvla_target_conditioned_visual_policy_response",
        "expected_object": expected_object,
        "predicted_object": None,
        "classification_accuracy_claimed": False,
        "target_conditioning_verified": expected_object in instruction,
        "image_path": str(image_path.resolve()),
        "instruction": instruction,
        "raw_openvla_action": [float(value) for value in action],
        "latency_seconds": float(latency_seconds),
        "endpoint": endpoint,
        "errors": errors,
        "success": not errors,
        "interpretation": (
            "PASS means the exact selected-object source frame and object-conditioned "
            "instruction produced a finite OpenVLA action; it is not a top-1 class claim."
        ),
    }


def run_openvla_policy_response(
    *,
    expected_object: str,
    image_path: Path,
    instruction: str,
    host: str,
    port: int,
    timeout: float = 120.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    action = predict(
        host,
        port,
        {"image_path": str(image_path.resolve()), "instruction": instruction},
        timeout=timeout,
    )
    latency = time.perf_counter() - started
    return validate_policy_response(
        expected_object=expected_object,
        image_path=image_path,
        instruction=instruction,
        action=action,
        latency_seconds=latency,
        endpoint=f"{host}:{port}",
    )
