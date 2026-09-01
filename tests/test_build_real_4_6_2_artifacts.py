import importlib
from pathlib import Path

import pytest

from tools.build_fast_4_6_2_report import NormalizedEpisodeEvidence


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def artifacts_module():
    return importlib.import_module("tools.build_real_4_6_2_artifacts")


def normalized_rows(method: str, suite: str, successes: int):
    return tuple(
        NormalizedEpisodeEvidence(
            method=method,
            suite=suite,
            task_id=task_id,
            trial=0,
            success=task_id < successes,
            source=f"C:/evidence/{method}/{suite}/task_{task_id:03d}.json",
        )
        for task_id in range(10)
    )


def result_matrix(module):
    counts = {
        "openvla_oft": (10, 10, 9, 10),
        "openvla_baseline": (4, 7, 4, 1),
        "diffusion_policy": (0, 0, 0, 0),
    }
    return {
        method: {
            suite: module.summarize_normalized(
                normalized_rows(method, suite, successes)
            )
            for suite, successes in zip(SUITES, method_counts)
        }
        for method, method_counts in counts.items()
    }


def test_default_paths_are_exact_and_do_not_point_at_old_report(tmp_path: Path):
    module = artifacts_module()

    inputs = module.default_artifact_inputs(tmp_path)

    assert inputs.baseline["libero_goal"] == (
        tmp_path
        / "outputs/experiment_4_6/formal_40/baseline/libero_goal/seed_7/summary.json"
    )
    assert inputs.diffusion_policy["libero_10"] == (
        tmp_path
        / "outputs/experiment_4_6/fast_reproduction/diffusion_policy/evaluations/formal_seed42"
        / "libero_10/libero_10/seed_42/suite_summary.json"
    )
    assert inputs.oft["libero_spatial"].name == "manifest.json"
    assert "run_20260730T132751Z_e990aef8" in str(inputs.oft["libero_spatial"])
    assert "final_4_6_2" not in str(inputs.training_process)


def test_normalized_boolean_rows_produce_exact_real_counts():
    module = artifacts_module()

    aggregate = module.summarize_normalized(
        normalized_rows("openvla_baseline", "libero_object", 7)
    )

    assert aggregate.successes == 7
    assert aggregate.episodes == 10
    assert aggregate.rate == 0.7


def test_zero_success_is_preserved_as_real_measurement():
    module = artifacts_module()

    aggregate = module.summarize_normalized(
        normalized_rows("diffusion_policy", "libero_10", 0)
    )

    assert aggregate.successes == 0
    assert aggregate.episodes == 10
    assert aggregate.rate == 0.0


def test_missing_real_inputs_raise_instead_of_using_reference_fallback(tmp_path: Path):
    module = artifacts_module()
    inputs = module.default_artifact_inputs(tmp_path)

    with pytest.raises(FileNotFoundError):
        module.collect_real_results(inputs)


def test_chinese_markdown_has_real_table_labels_figures_and_no_reference_values():
    module = artifacts_module()

    markdown = module.build_chinese_markdown(result_matrix(module))

    assert "| 套件 | OpenVLA-OFT | OpenVLA baseline | Diffusion Policy |" in markdown
    assert "| 平均（40 episodes） | 39/40 (97.5%) | 16/40 (40.0%) | 0/40 (0.0%) |" in markdown
    assert "DP 在本次严格缩减协议下为 0/40" in markdown
    assert "特征缓存" in markdown
    assert "单种子 20k" in markdown
    assert "Navigation 只是本报告沿用的称呼" in markdown
    assert "官方 LIBERO-10 长时程套件" in markdown
    assert "LIBERO-Navigation" not in markdown
    assert "本实验不包含真实机械臂实验" in markdown
    assert "training_process.png" in markdown
    assert "process_montage_real.png" in markdown
    assert "comparison_real.png" in markdown
    assert markdown.count("插入位置：") == 3
    assert markdown.count("**图题：**") == 3
    assert markdown.count("**图意：**") == 3
    assert markdown.count("**本项目解释：**") == 3
    for forbidden_reference in ("76.5%", "72.1%", "69.4%", "68.2%", "73.2%", "70.8%", "67.5%", "64.8%"):
        assert forbidden_reference not in markdown
