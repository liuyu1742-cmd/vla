from pathlib import Path

from tools.openvla_4l_dagger_orchestrator import collector_command


def test_collector_command_uses_requested_round_beta_and_seed():
    command = collector_command(
        Path("robocasa-python"),
        seed=5,
        round_index=1,
        beta=0.3,
        port=8933,
        steps=900,
        output_dir=Path("datasets/recovery"),
    )

    assert command[:3] == ["robocasa-python", "-m", "tools.collect_water_cup_dagger_v2"]
    assert command[command.index("--seed") + 1] == "5"
    assert command[command.index("--round") + 1] == "1"
    assert command[command.index("--beta") + 1] == "0.3"
    assert command[command.index("--port") + 1] == "8933"
    assert command[command.index("--steps") + 1] == "900"
    assert command[command.index("--output-dir") + 1] == "datasets/recovery"
