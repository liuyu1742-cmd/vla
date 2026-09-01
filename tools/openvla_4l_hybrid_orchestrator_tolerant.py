"""Hybrid evaluator using the calibrated 16 mm glass-contact transition."""

from __future__ import annotations

from pathlib import Path

from tools import openvla_4l_hybrid_orchestrator as base


_original_guarded_command = base.guarded_command


def guarded_command(
    robocasa_python: Path,
    *,
    seed: int,
    port: int,
    steps: int,
    output_dir: Path,
    max_translation_error: float,
) -> list[str]:
    command = _original_guarded_command(
        robocasa_python,
        seed=seed,
        port=port,
        steps=steps,
        output_dir=output_dir,
        max_translation_error=max_translation_error,
    )
    command[2] = "tools.openvla_gym_water_cup_guarded_rollout_tolerant"
    return command


def main() -> int:
    base.guarded_command = guarded_command
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
