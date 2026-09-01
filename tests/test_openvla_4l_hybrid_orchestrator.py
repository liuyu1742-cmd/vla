from pathlib import Path

from tools.openvla_4l_hybrid_orchestrator import guarded_command


def test_guarded_command_uses_same_closed_loop_budget_and_recovery_threshold():
    command = guarded_command(
        Path("robocasa-python"),
        seed=3,
        port=8940,
        steps=700,
        output_dir=Path("outputs/hybrid_seed3"),
        max_translation_error=0.15,
    )

    assert command[:3] == [
        "robocasa-python",
        "-m",
        "tools.openvla_gym_water_cup_guarded_rollout",
    ]
    assert command[command.index("--seed") + 1] == "3"
    assert command[command.index("--port") + 1] == "8940"
    assert command[command.index("--steps") + 1] == "700"
    assert command[command.index("--max-translation-error") + 1] == "0.15"
