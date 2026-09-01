"""Auditable DAgger action mixing and persistence for ``organizing::toy``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np


RELATION_KEY = "organizing::toy"


class RandomSource(Protocol):
    def random(self) -> float: ...


class OracleDecision(Protocol):
    action: np.ndarray
    phase: str
    canonical_action: str
    force_expert: bool


@dataclass(frozen=True)
class MixingResult:
    executed_action: np.ndarray
    oracle_label: np.ndarray
    source: str
    gripper_gated: bool


def validate_training_seed(seed: int, heldout_seeds: Iterable[int]) -> None:
    value = int(seed)
    if value < 0:
        raise ValueError("seed must be non-negative")
    if value in {int(item) for item in heldout_seeds}:
        raise ValueError(f"held-out seed {value} cannot be used for DAgger training")


def _action(name: str, value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    return np.clip(result, -1.0, 1.0)


def choose_executed_action(
    policy_action: np.ndarray,
    oracle_decision: OracleDecision,
    *,
    beta: float,
    rng: RandomSource,
) -> MixingResult:
    """Mix policy and expert while preserving the formal ``-1=open`` contract."""

    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be between zero and one")
    policy = _action("policy_action", policy_action)
    oracle = _action("oracle_action", oracle_decision.action)
    if oracle_decision.force_expert:
        return MixingResult(oracle.copy(), oracle.copy(), "oracle_forced", False)
    if rng.random() < beta:
        return MixingResult(oracle.copy(), oracle.copy(), "oracle_mixed", False)

    executed = policy.copy()
    premature_close = (
        oracle_decision.phase in {"approach_object", "descend_to_object"}
        and executed[6] > 0.5
    )
    if premature_close:
        executed[6] = -1.0
    return MixingResult(
        executed,
        oracle.copy(),
        "policy_gated" if premature_close else "policy",
        premature_close,
    )


def _sample_array(
    name: str,
    value: Any,
    *,
    samples: int,
    tail: tuple[int, ...],
    dtype: Any,
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    expected = (samples, *tail)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {result.shape}")
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    return result


def save_dagger_episode(
    output_root: Path,
    *,
    seed: int,
    heldout_seeds: Iterable[int],
    frames: Any,
    oracle_actions: Any,
    policy_actions: Any,
    executed_actions: Any,
    phases: Sequence[str],
    canonical_phases: Sequence[str],
    eef_positions: Any,
    object_positions: Any,
    grasped: Sequence[bool],
    report: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Save oracle labels as canonical actions and retain policy execution audit data."""

    validate_training_seed(seed, heldout_seeds)
    if report.get("relation_key") != RELATION_KEY:
        raise ValueError("DAgger report relation_key mismatch")
    instruction = str(report.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("DAgger report instruction is required")
    frame_array = np.asarray(frames, dtype=np.uint8)
    if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
        raise ValueError(f"frames must have shape (N,H,W,3), got {frame_array.shape}")
    samples = int(frame_array.shape[0])
    oracle_array = _sample_array(
        "oracle_actions",
        oracle_actions,
        samples=samples,
        tail=(7,),
        dtype=np.float32,
    )
    policy_array = _sample_array(
        "policy_actions",
        policy_actions,
        samples=samples,
        tail=(7,),
        dtype=np.float32,
    )
    executed_array = _sample_array(
        "executed_actions",
        executed_actions,
        samples=samples,
        tail=(7,),
        dtype=np.float32,
    )
    eef_array = _sample_array(
        "eef_positions",
        eef_positions,
        samples=samples,
        tail=(3,),
        dtype=np.float32,
    )
    object_array = _sample_array(
        "object_positions",
        object_positions,
        samples=samples,
        tail=(3,),
        dtype=np.float32,
    )
    if not all(
        len(values) == samples for values in (phases, canonical_phases, grasped)
    ):
        raise ValueError("phase and grasp state lengths must match frame count")

    round_index = int(report.get("round", 0))
    episode_dir = (
        Path(output_root) / f"seed_{int(seed):03d}" / f"round_{round_index:02d}"
    )
    episode_dir.mkdir(parents=True, exist_ok=True)
    episode_path = episode_dir / "episode.npz"
    report_path = episode_dir / "report.json"
    sidecar_path = episode_dir / "manifest_entry.json"
    np.savez_compressed(
        episode_path,
        frames=frame_array,
        actions=oracle_array,
        oracle_actions=oracle_array,
        policy_actions=policy_array,
        executed_actions=executed_array,
        phases=np.asarray(phases, dtype="U48"),
        canonical_phases=np.asarray(canonical_phases, dtype="U16"),
        eef_positions=eef_array,
        object_positions=object_array,
        grasped=np.asarray(grasped, dtype=bool),
        relation_key=np.asarray(RELATION_KEY),
        instruction=np.asarray(instruction),
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sidecar = {
        "seed": int(seed),
        "round": round_index,
        "relation_key": RELATION_KEY,
        "instruction": instruction,
        "source": "dagger_recovery",
        "success": bool(report.get("success", False)),
        "samples": samples,
        "beta": float(report.get("beta", 0.0)),
        "episode": str(episode_path.resolve()),
        "report": str(report_path.resolve()),
    }
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return episode_path, report_path, sidecar_path


__all__ = [
    "MixingResult",
    "choose_executed_action",
    "save_dagger_episode",
    "validate_training_seed",
]
