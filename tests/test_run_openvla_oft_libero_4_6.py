from pathlib import Path

from tools.run_openvla_oft_libero_4_6 import build_command, parse_success_counts


def test_build_command_uses_pure_official_policy():
    cmd = build_command(
        Path("python.exe"),
        Path("third_party/openvla-oft"),
        "libero_goal",
        1,
        7,
    )
    joined = " ".join(cmd)
    assert "run_libero_eval.py" in joined
    assert "--task_suite_name libero_goal" in joined
    assert "--num_trials_per_task 1" in joined
    assert "--center_crop True" in joined
    assert "expert" not in joined.lower()


def test_parse_success_counts_uses_final_suite_counter():
    text = """
Task 1 success rate: 0.8
Current total success rate: 0.8
Current total success rate: 0.9
"""
    assert parse_success_counts(text) == {"success_rate": 90.0}

