"""Build final 4.6.2 artifacts exclusively from strict local simulator evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from tools.build_fast_4_6_2_report import (
    NormalizedEpisodeEvidence,
    normalize_baseline_summary,
    normalize_diffusion_summary,
    normalize_oft_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
METHODS = ("openvla_oft", "openvla_baseline", "diffusion_policy")
METHOD_LABELS = {
    "openvla_oft": "OpenVLA-OFT",
    "openvla_baseline": "OpenVLA baseline",
    "diffusion_policy": "Diffusion Policy",
}
SUITE_LABELS = {
    "libero_spatial": "LIBERO-Spatial",
    "libero_object": "LIBERO-Object",
    "libero_goal": "LIBERO-Goal",
    "libero_10": "LIBERO-10（长时程任务）",
}
OFFICIAL_HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
OFT_RUN_IDS = {
    "libero_spatial": "run_20260730T132751Z_e990aef8",
    "libero_object": "run_20260730T133209Z_681a1486",
    "libero_goal": "run_20260730T132058Z_56d71214",
    "libero_10": "run_20260730T133613Z_45db29a1",
}
DEFAULT_OUTPUT_RELATIVE = Path("outputs/experiment_4_6/final_4_6_2_real")
REPORT_FILENAME = "4.6.2_OpenVLA_OFT_实验结果.md"


@dataclass(frozen=True)
class ArtifactInputs:
    oft: Mapping[str, Path]
    baseline: Mapping[str, Path]
    diffusion_policy: Mapping[str, Path]
    training_process: Path


@dataclass(frozen=True)
class SuiteAggregate:
    method: str
    suite: str
    successes: int
    episodes: int
    rate: float
    rows: tuple[NormalizedEpisodeEvidence, ...]


def default_artifact_inputs(repo_root: Path = ROOT) -> ArtifactInputs:
    root = Path(repo_root).resolve()
    oft_root = root / "outputs/experiment_4_6/oft_official_deps_smoke"
    baseline_root = root / "outputs/experiment_4_6/formal_40/baseline"
    diffusion_root = (
        root
        / "outputs/experiment_4_6/fast_reproduction/diffusion_policy/evaluations/formal_seed42"
    )
    return ArtifactInputs(
        oft={
            suite: oft_root
            / suite
            / "seed_7"
            / OFT_RUN_IDS[suite]
            / "manifest.json"
            for suite in SUITES
        },
        baseline={
            suite: baseline_root / suite / "seed_7" / "summary.json"
            for suite in SUITES
        },
        diffusion_policy={
            suite: diffusion_root
            / suite
            / suite
            / "seed_42"
            / "suite_summary.json"
            for suite in SUITES
        },
        training_process=(
            root
            / "outputs/experiment_4_6/fast_reproduction/diffusion_policy/runs"
            / "joint_20k_seed42/training_process.png"
        ),
    )


def summarize_normalized(
    rows: Iterable[NormalizedEpisodeEvidence],
) -> SuiteAggregate:
    episodes = tuple(rows)
    if len(episodes) != 10:
        raise ValueError("strict suite aggregation requires exactly 10 episodes")
    identities = {(row.task_id, row.trial) for row in episodes}
    expected = {(task_id, 0) for task_id in range(10)}
    if identities != expected or len(identities) != len(episodes):
        raise ValueError("strict suite aggregation requires task IDs 0..9 and trial 0")
    if any(type(row.success) is not bool for row in episodes):
        raise ValueError("strict suite aggregation accepts only boolean success")
    methods = {row.method for row in episodes}
    suites = {row.suite for row in episodes}
    if len(methods) != 1 or len(suites) != 1:
        raise ValueError("normalized episodes disagree on method or suite")
    ordered = tuple(sorted(episodes, key=lambda row: (row.task_id, row.trial)))
    successes = sum(row.success for row in ordered)
    return SuiteAggregate(
        method=ordered[0].method,
        suite=ordered[0].suite,
        successes=successes,
        episodes=len(ordered),
        rate=successes / len(ordered),
        rows=ordered,
    )


def collect_real_results(
    inputs: ArtifactInputs,
) -> dict[str, dict[str, SuiteAggregate]]:
    adapters = {
        "openvla_oft": (inputs.oft, normalize_oft_manifest),
        "openvla_baseline": (inputs.baseline, normalize_baseline_summary),
        "diffusion_policy": (inputs.diffusion_policy, normalize_diffusion_summary),
    }
    results: dict[str, dict[str, SuiteAggregate]] = {}
    for method in METHODS:
        paths, adapter = adapters[method]
        results[method] = {}
        if set(paths) != set(SUITES):
            raise ValueError(f"{method} input mapping must contain exactly four suites")
        for suite in SUITES:
            aggregate = summarize_normalized(adapter(paths[suite]))
            if aggregate.method != method or aggregate.suite != suite:
                raise ValueError(f"adapter identity mismatch for {method}/{suite}")
            results[method][suite] = aggregate
    return results


def _overall(
    suites: Mapping[str, SuiteAggregate],
) -> tuple[int, int, float]:
    if set(suites) != set(SUITES):
        raise ValueError("overall aggregation requires all four suites")
    successes = sum(suites[suite].successes for suite in SUITES)
    episodes = sum(suites[suite].episodes for suite in SUITES)
    if episodes != 40:
        raise ValueError("overall aggregation requires exactly 40 episodes")
    return successes, episodes, successes / episodes


def _format_count(successes: int, episodes: int) -> str:
    return f"{successes}/{episodes} ({successes / episodes * 100:.1f}%)"


def build_chinese_markdown(
    results: Mapping[str, Mapping[str, SuiteAggregate]],
) -> str:
    table = [
        "| 套件 | OpenVLA-OFT | OpenVLA baseline | Diffusion Policy |",
        "|---|---:|---:|---:|",
    ]
    for suite in SUITES:
        table.append(
            "| "
            + SUITE_LABELS[suite]
            + " | "
            + " | ".join(
                _format_count(
                    results[method][suite].successes,
                    results[method][suite].episodes,
                )
                for method in METHODS
            )
            + " |"
        )
    overall = {method: _overall(results[method]) for method in METHODS}
    table.append(
        "| 平均（40 episodes） | "
        + " | ".join(
            _format_count(overall[method][0], overall[method][1])
            for method in METHODS
        )
        + " |"
    )
    diffusion_successes, diffusion_episodes, _ = overall["diffusion_policy"]

    return "\n".join(
        [
            "# 4.6.2 OpenVLA-OFT 实验结果",
            "",
            "本小节只使用本项目本地 LIBERO 仿真产生的严格 episode 证据；不存在论文表格、旧报告或人工参考成功率 fallback。本实验不包含真实机械臂实验，真实机械臂部分不在本节范围。",
            "",
            "## 实验协议",
            "",
            "- 四个套件均覆盖 10 个官方任务，每任务只执行 1 次，并固定使用官方初态列表中的第 0 个初态（trial 0）。",
            "- 各套件沿用官方 horizon：Spatial 220、Object 280、Goal 300、LIBERO-10 520。",
            "- OpenVLA-OFT 与 OpenVLA baseline 使用 seed 7；Diffusion Policy 使用 seed 42。",
            "- Navigation 只是本报告沿用的称呼，实际证据 key `libero_10` 对应官方 LIBERO-10 长时程套件，不表示移动导航任务。",
            "- 成功率均由严格 boolean `success` 的分子/分母直接重算；带执行错误、缺任务、重复 identity 或非布尔结果的套件不会进入本表。",
            "",
            "## 本地实测汇总",
            "",
            *table,
            "",
            f"DP 在本次严格缩减协议下为 {diffusion_successes}/{diffusion_episodes}。这是本地真实结果，不能用参考值、历史值或插值替换。该缩减实现使用预计算的特征缓存，并只进行单种子 20k 步训练；现有结果说明模型未学到可迁移到四个套件的闭环控制，而不是证据缺失。",
            "",
            "## 训练过程图",
            "",
            "插入位置：Diffusion Policy 训练设置和结果解释之后。",
            "",
            "**图题：** 图 4.6.2-1 Diffusion Policy 单种子 20k 步训练过程。",
            "",
            "![图 4.6.2-1 Diffusion Policy 训练过程](training_process.png)",
            "",
            "**图意：** 展示已验证训练日志对应的损失变化和训练进程，不把评估成功率混入训练曲线。",
            "",
            "**本项目解释：** 训练目标下降只证明缓存特征上的优化发生，不能单独证明策略已经获得跨任务闭环控制能力；最终判断必须服从 40 个真实仿真 episode。",
            "",
            "## 四套件过程证据图",
            "",
            "插入位置：训练过程图之后、本地成功率对比图之前。",
            "",
            "**图题：** 图 4.6.2-2 三种方法在四个 LIBERO 套件中的真实仿真过程帧。",
            "",
            "![图 4.6.2-2 三方法四套件真实过程帧](process_montage_real.png)",
            "",
            "**图意：** 3 行依次为 OpenVLA-OFT、OpenVLA baseline、Diffusion Policy，4 列依次为 Spatial、Object、Goal、LIBERO-10；OFT 帧从真实 MP4 中段抽取，另外两行使用真实 episode 的 middle PNG。",
            "",
            "**本项目解释：** 该图用于证明每个方法与套件均有可追溯的仿真视觉过程；它不替代严格 boolean episode 计数，也不根据画面主观判断成功。",
            "",
            "## 三方法成功率对比图",
            "",
            "插入位置：本小节结果表和过程证据之后。",
            "",
            "**图题：** 图 4.6.2-3 三种方法在四个 LIBERO 套件上的本地真实成功率。",
            "",
            "![图 4.6.2-3 三方法四套件真实成功率](comparison_real.png)",
            "",
            "**图意：** 每个柱对应一个方法与套件的 successes/10，零高度柱仍保留并标注 0/10；图中没有论文参考值或旧报告值。",
            "",
            "**本项目解释：** OpenVLA-OFT 在该缩减协议下保持最高成功数，baseline 的套件间差异明显，DP 的 0/40 与训练曲线共同表明当前缓存特征、单种子和 20k 步预算不足以学习可迁移闭环策略。",
            "",
            "所有机器可读输入路径、SHA256、文件大小、逐 suite 计数、过程图源文件和输出哈希见同目录 `evidence_index.json`。",
            "",
        ]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_info(path: Path) -> dict[str, object]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    size = source.stat().st_size
    if size <= 0:
        raise ValueError(f"evidence file is empty: {source}")
    return {"path": str(source), "sha256": _sha256(source), "size_bytes": size}


def _font_properties():
    from matplotlib.font_manager import FontProperties

    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    return FontProperties(fname=str(font_path)) if font_path.is_file() else FontProperties()


def _render_comparison(
    results: Mapping[str, Mapping[str, SuiteAggregate]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    font = _font_properties()
    x = np.arange(len(SUITES))
    width = 0.24
    colors = ("#2563eb", "#f59e0b", "#64748b")
    figure, axis = plt.subplots(figsize=(13.5, 7.2), dpi=160)
    for method_index, method in enumerate(METHODS):
        values = [results[method][suite].rate * 100 for suite in SUITES]
        bars = axis.bar(
            x + (method_index - 1) * width,
            values,
            width,
            label=METHOD_LABELS[method],
            color=colors[method_index],
        )
        for bar, suite in zip(bars, SUITES):
            aggregate = results[method][suite]
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                max(bar.get_height() + 1.5, 1.5),
                f"{aggregate.successes}/{aggregate.episodes}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axis.set_xticks(x, [SUITE_LABELS[suite] for suite in SUITES], fontproperties=font)
    axis.set_ylabel("成功率（%）", fontproperties=font)
    axis.set_ylim(0, 108)
    axis.set_title("三种方法 × 四个 LIBERO 套件：本地真实成功率", fontproperties=font, fontsize=16)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def _load_png(path: Path):
    from PIL import Image
    import numpy as np

    source = Path(path).resolve()
    with Image.open(source) as image:
        image.verify()
    with Image.open(source) as image:
        return np.asarray(image.convert("RGB")), source, {}


def _extract_video_middle(path: Path):
    import cv2

    source = Path(path).resolve()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"cannot decode OFT rollout video: {source}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"OFT rollout has no decodable frames: {source}")
        frame_index = frame_count // 2
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"cannot decode OFT middle frame: {source}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb, source, {"frame_index": frame_index, "frame_count": frame_count}
    finally:
        capture.release()


def _representative_row(aggregate: SuiteAggregate) -> NormalizedEpisodeEvidence:
    return next((row for row in aggregate.rows if row.success), aggregate.rows[0])


def _montage_cell(
    method: str,
    aggregate: SuiteAggregate,
):
    row = _representative_row(aggregate)
    episode_source = Path(row.source)
    if method == "openvla_oft":
        image, media_source, extraction = _extract_video_middle(episode_source)
    elif method == "openvla_baseline":
        episode = json.loads(episode_source.read_text(encoding="utf-8"))
        frame_paths = episode.get("frame_paths")
        if not isinstance(frame_paths, list):
            raise ValueError(f"baseline episode has no frame_paths: {episode_source}")
        matches = [Path(value) for value in frame_paths if Path(value).name == "middle.png"]
        if len(matches) != 1:
            raise ValueError(f"baseline episode must have one middle.png: {episode_source}")
        image, media_source, extraction = _load_png(matches[0])
    else:
        media_source = episode_source.with_name(episode_source.stem + "_middle.png")
        image, media_source, extraction = _load_png(media_source)
    record = {
        "method": method,
        "suite": aggregate.suite,
        "task_id": row.task_id,
        "trial": row.trial,
        "success": row.success,
        "episode_source": _file_info(episode_source),
        "media_source": _file_info(media_source),
        **extraction,
    }
    return image, record


def _render_process_montage(
    results: Mapping[str, Mapping[str, SuiteAggregate]],
    output: Path,
) -> list[dict[str, object]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font = _font_properties()
    figure, axes = plt.subplots(3, 4, figsize=(16, 10), dpi=150)
    records = []
    for row_index, method in enumerate(METHODS):
        for column_index, suite in enumerate(SUITES):
            image, record = _montage_cell(method, results[method][suite])
            records.append(record)
            axis = axes[row_index][column_index]
            axis.imshow(image)
            axis.axis("off")
            status = "success=True" if record["success"] else "success=False"
            axis.set_title(
                f"{METHOD_LABELS[method]}\n{SUITE_LABELS[suite]}\nTask {record['task_id']} · {status}",
                fontproperties=font,
                fontsize=10,
            )
    figure.suptitle(
        "三种方法 × 四个 LIBERO 套件：真实仿真过程帧",
        fontproperties=font,
        fontsize=18,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output)
    plt.close(figure)
    return records


def _source_mapping(inputs: ArtifactInputs) -> Mapping[str, Mapping[str, Path]]:
    return {
        "openvla_oft": inputs.oft,
        "openvla_baseline": inputs.baseline,
        "diffusion_policy": inputs.diffusion_policy,
    }


def _source_metadata(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "schema_version",
        "suite",
        "seed",
        "checkpoint",
        "checkpoint_sha256",
        "source_commit",
        "started_utc",
        "ended_utc",
        "finished_at",
    )
    return {key: payload[key] for key in keys if key in payload}


def _build_evidence_index(
    inputs: ArtifactInputs,
    results: Mapping[str, Mapping[str, SuiteAggregate]],
    montage_sources: list[dict[str, object]],
    output_paths: Mapping[str, Path],
) -> dict[str, object]:
    sources = _source_mapping(inputs)
    methods: dict[str, object] = {}
    for method in METHODS:
        suites: dict[str, object] = {}
        for suite in SUITES:
            aggregate = results[method][suite]
            input_path = sources[method][suite]
            suites[suite] = {
                "successes": aggregate.successes,
                "episodes": aggregate.episodes,
                "rate": aggregate.rate,
                "input": _file_info(input_path),
                "input_metadata": _source_metadata(input_path),
                "episodes_evidence": [
                    {
                        "task_id": row.task_id,
                        "trial": row.trial,
                        "success": row.success,
                        "source": _file_info(Path(row.source)),
                    }
                    for row in aggregate.rows
                ],
            }
        successes, episodes, rate = _overall(results[method])
        methods[method] = {
            "label": METHOD_LABELS[method],
            "suites": suites,
            "overall": {"successes": successes, "episodes": episodes, "rate": rate},
        }
    return {
        "schema_version": "openvla_4_6_2_real_evidence_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": "local_measured",
        "protocol": {
            "suite_order": list(SUITES),
            "tasks_per_suite": 10,
            "trials_per_task": 1,
            "fixed_initial_state_index": 0,
            "official_horizons": OFFICIAL_HORIZONS,
            "seeds": {"openvla_oft": 7, "openvla_baseline": 7, "diffusion_policy": 42},
            "libero_10_note": "Navigation is only a report name; evidence is official LIBERO-10 long-horizon.",
        },
        "methods": methods,
        "process_montage_sources": montage_sources,
        "training_process": {
            "source": _file_info(inputs.training_process),
            "copied_output": _file_info(output_paths["training_process"]),
        },
        "outputs": {
            name: _file_info(path)
            for name, path in output_paths.items()
            if name != "evidence_index"
        },
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_real_artifacts(
    *,
    repo_root: Path = ROOT,
    output_dir: Path | None = None,
) -> dict[str, object]:
    root = Path(repo_root).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (root / DEFAULT_OUTPUT_RELATIVE).resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    inputs = default_artifact_inputs(root)
    results = collect_real_results(inputs)
    output_paths = {
        "comparison": destination / "comparison_real.png",
        "process_montage": destination / "process_montage_real.png",
        "training_process": destination / "training_process.png",
        "markdown": destination / REPORT_FILENAME,
        "evidence_index": destination / "evidence_index.json",
    }

    _render_comparison(results, output_paths["comparison"])
    montage_sources = _render_process_montage(results, output_paths["process_montage"])
    _file_info(inputs.training_process)
    shutil.copy2(inputs.training_process, output_paths["training_process"])
    if _sha256(inputs.training_process) != _sha256(output_paths["training_process"]):
        raise RuntimeError("training_process.png copy hash mismatch")
    output_paths["markdown"].write_text(
        build_chinese_markdown(results),
        encoding="utf-8",
    )
    index = _build_evidence_index(inputs, results, montage_sources, output_paths)
    _write_json(output_paths["evidence_index"], index)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build final 4.6.2 artifacts from strict local evidence only."
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    index = build_real_artifacts(repo_root=args.repo_root, output_dir=args.output_dir)
    summary = {
        method: entry["overall"]
        for method, entry in index["methods"].items()
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
