from pathlib import Path

from tools.run_openvla_oft_libero_4_6_v2 import build_command, parse_run_metrics


def test_build_command_uses_absolute_evaluator_and_local_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    cmd = build_command(
        Path("python.exe"),
        Path("third_party/openvla-oft"),
        "libero_goal",
        1,
        7,
        checkpoint=checkpoint,
        local_log_dir=tmp_path / "logs",
    )
    assert Path(cmd[1]).is_absolute()
    index = cmd.index("--pretrained_checkpoint")
    assert Path(cmd[index + 1]) == checkpoint.resolve()
    assert "--center_crop True" in " ".join(cmd)
    assert "expert" not in " ".join(cmd).lower()


def test_parse_run_metrics_reads_final_totals_and_tasks():
    text = """
Task: put the red mug on the plate
Current task success rate: 0.8
Task: open the middle drawer
Current task success rate: 1.0
Total episodes: 20
Total successes: 18
Overall success rate: 0.9000 (90.0%)
"""
    assert parse_run_metrics(text) == {
        "success_rate": 90.0,
        "total_episodes": 20,
        "total_successes": 18,
        "task_success_rates": [
            {"task": "put the red mug on the plate", "success_rate": 80.0},
            {"task": "open the middle drawer", "success_rate": 100.0},
        ],
    }

