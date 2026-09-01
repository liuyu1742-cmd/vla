from pathlib import Path

from tools.openvla_4l_hybrid_orchestrator_tolerant import guarded_command


def test_tolerant_orchestrator_calls_calibrated_rollout():
    command = guarded_command(
        Path("robocasa-python"),
        seed=3,
        port=8941,
        steps=700,
        output_dir=Path("outputs/tolerant"),
        max_translation_error=0.15,
    )

    assert command[2] == "tools.openvla_gym_water_cup_guarded_rollout_tolerant"
    assert command[command.index("--seed") + 1] == "3"
