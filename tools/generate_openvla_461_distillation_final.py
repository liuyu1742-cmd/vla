"""Generate the evidence-backed final Section 4.6.1 and real process figures."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "experiment_4_6"
RUNS = OUT / "openvla_4l"


def format_rate(successes: int, trials: int) -> str:
    percent = 100.0 * successes / trials if trials else 0.0
    return f"{successes}/{trials} ({percent:.1f}%)"


def method_rows() -> list[dict]:
    return [
        {
            "mode": "full",
            "method": "Full Fine-tuning",
            "trainable_m": 1874.5,
            "trainable_percent": 100.0,
            "vram_gb": 8.92,
            "pure_successes": 0,
            "pure_trials": 5,
            "hybrid_dir": "hybrid_tolerant_full",
        },
        {
            "mode": "lora_r32",
            "method": "LoRA (rank=32)",
            "trainable_m": 40.9,
            "trainable_percent": 2.18,
            "vram_gb": 4.34,
            "pure_successes": 0,
            "pure_trials": 1,
            "hybrid_dir": "hybrid_tolerant_lora_distilled_r1",
        },
        {
            "mode": "last_layer_only",
            "method": "Last-Layer-Only",
            "trainable_m": 333.7,
            "trainable_percent": 17.8,
            "vram_gb": 6.11,
            "pure_successes": 0,
            "pure_trials": 5,
            "hybrid_dir": "hybrid_tolerant_last_layer_only",
        },
        {
            "mode": "frozen_vision",
            "method": "Frozen-Vision",
            "trainable_m": 1143.6,
            "trainable_percent": 61.0,
            "vram_gb": 7.62,
            "pure_successes": 0,
            "pure_trials": 5,
            "hybrid_dir": "hybrid_tolerant_frozen_vision",
        },
    ]


def configure_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_results() -> list[dict]:
    rows = method_rows()
    for row in rows:
        path = RUNS / row["hybrid_dir"] / "hybrid_summary.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        row["hybrid_successes"] = int(data["successes"])
        row["hybrid_trials"] = int(data["episodes"])
        row["hybrid_success_rate"] = float(data["success_rate"])
        row["openvla_direct_fraction"] = float(
            data["mean_openvla_direct_fraction"]
        )
        row["hybrid_summary"] = str(path.resolve())
    return rows


def training_curve() -> tuple[list[int], list[float]]:
    path = RUNS / "lora_r32_distilled_r1_train.stdout.log"
    pattern = re.compile(r"step=(\d+)/300 loss=([0-9.]+)")
    steps, losses = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))
    if not steps:
        raise ValueError("distillation training log contains no loss samples")
    return steps, losses


def render_training_process(rows: list[dict], target: Path) -> None:
    configure_font()
    steps, losses = training_curve()
    reports = [
        json.loads(
            (
                ROOT
                / "datasets"
                / "water_cup_dagger"
                / f"episode_seed_{seed:03d}_round_01_report.json"
            ).read_text(encoding="utf-8")
        )
        for seed in (0, 1, 3, 5)
    ]
    phase_names = [
        "approach_object",
        "descend_to_object",
        "close_gripper",
        "lift_object",
    ]
    phase_counts = [
        sum(int(report["phase_counts"].get(name, 0)) for report in reports)
        for name in phase_names
    ]

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=180)
    axes[0].plot(steps, losses, color="#1565C0", linewidth=1.5)
    axes[0].set_title("第二轮 LoRA 视觉—动作蒸馏")
    axes[0].set_xlabel("优化步")
    axes[0].set_ylabel("训练损失")
    axes[0].grid(alpha=0.25)

    axes[1].bar(
        ["接近", "下降", "闭合", "抬升"],
        phase_counts,
        color=["#42A5F5", "#26A69A", "#FF7043", "#7E57C2"],
    )
    axes[1].set_title("DAgger 学生访问状态（4种子）")
    axes[1].set_ylabel("真实采集帧数")
    axes[1].grid(axis="y", alpha=0.25)

    labels = [row["method"].replace(" Fine-tuning", "") for row in rows]
    direct = [100.0 * row["openvla_direct_fraction"] for row in rows]
    axes[2].barh(labels, direct, color="#43A047")
    axes[2].set_xlim(0, 100)
    axes[2].set_xlabel("OpenVLA 直接控制占比（%）")
    axes[2].set_title("成功混合闭环中的模型贡献")
    axes[2].grid(axis="x", alpha=0.25)
    for index, value in enumerate(direct):
        axes[2].text(value + 1, index, f"{value:.1f}%", va="center")

    figure.suptitle(
        "OpenVLA-4L 本机实验过程证据：训练、DAgger 与闭环贡献",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)


def render_real_rollout(target: Path) -> None:
    configure_font()
    dagger_path = (
        ROOT
        / "datasets"
        / "water_cup_dagger"
        / "episode_seed_000_round_01.npz"
    )
    with np.load(dagger_path, allow_pickle=False) as episode:
        frames = np.asarray(episode["frames"], dtype=np.uint8)
    indexes = [0, 50, 75, 100, 300]
    titles = ["初始接近", "接触下降", "夹爪闭合", "抓取抬升", "学生访问状态"]

    success_root = RUNS / "hybrid_tolerant_lora_distilled_r1"
    final_frames = [
        success_root / f"seed_{seed:03d}" / "current_frame.png"
        for seed in (0, 1, 2, 3, 5)
    ]
    figure, axes = plt.subplots(2, 5, figsize=(16, 7.2), dpi=180)
    for axis, index, title in zip(axes[0], indexes, titles):
        axis.imshow(frames[index])
        axis.set_title(f"{title}\nDAgger step {index}")
        axis.axis("off")
    for axis, seed, path in zip(axes[1], (0, 1, 2, 3, 5), final_frames):
        axis.imshow(Image.open(path).convert("RGB"))
        axis.set_title(f"seed {seed}：成功结束帧")
        axis.axis("off")
    figure.suptitle(
        "真实 RoboCasa 实验帧：DAgger 纠偏过程与 LoRA 5/5 成功闭环",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)


def render_result_table(rows: list[dict], target: Path) -> None:
    configure_font()
    columns = [
        "微调方法",
        "可训练参数",
        "峰值显存",
        "纯学生成功率",
        "专家恢复成功率",
        "OpenVLA直接占比",
    ]
    cells = [
        [
            row["method"],
            f"{row['trainable_m']:,.1f}M\n({row['trainable_percent']:.2f}%)",
            f"{row['vram_gb']:.2f}GB",
            format_rate(row["pure_successes"], row["pure_trials"]),
            format_rate(row["hybrid_successes"], row["hybrid_trials"]),
            f"{100.0 * row['openvla_direct_fraction']:.1f}%",
        ]
        for row in rows
    ]
    figure, axis = plt.subplots(figsize=(15.5, 4.2), dpi=180)
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.22, 0.16, 0.12, 0.16, 0.18, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 2.1)
    for (row_index, _), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", weight="bold")
        elif row_index == 2:
            cell.set_facecolor("#E8F5E9")
        else:
            cell.set_facecolor("#F7F9FC")
    axis.set_title(
        "表 4-7  OpenVLA-4L 微调方法真实对比（RTX 3090，RoboCasa）",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )
    figure.tight_layout()
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)


def markdown(rows: list[dict], images: dict[str, Path]) -> str:
    table_lines = [
        "| 微调方法 | 可训练参数量 | 显存 | 纯学生成功率 | 专家恢复成功率 | OpenVLA直接控制占比 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            "| {method} | {params:,.1f}M ({percent:.2f}%) | {vram:.2f}GB | "
            "{pure} | {hybrid} | {direct:.1f}% |".format(
                method=row["method"],
                params=row["trainable_m"],
                percent=row["trainable_percent"],
                vram=row["vram_gb"],
                pure=format_rate(row["pure_successes"], row["pure_trials"]),
                hybrid=format_rate(row["hybrid_successes"], row["hybrid_trials"]),
                direct=100.0 * row["openvla_direct_fraction"],
            )
        )
    table = "\n".join(table_lines)
    return f"""### 4.6.1 OpenVLA 微调方法对比

本节在本项目可实际运行的 **OpenVLA-4L 裁剪模型**上复刻四种指定微调方法：Full Fine-tuning、LoRA（rank=32）、Last-Layer-Only 和 Frozen-Vision。实验硬件为单张 NVIDIA GeForce RTX 3090（24GB），任务采用 RoboCasa `PickPlaceCounterToCabinet`，目标物体为 `glass_cup`，物体尺度为 0.7。训练数据使用种子 0、1、2、3、5 的五次专家演示；其中 LoRA 进一步进行了两轮视觉—动作蒸馏。第二轮蒸馏集由原始专家轨迹与种子 0、1、3、5 的 3600 帧 DAgger 学生访问状态组成，每个种子重采样 1200 帧，静止动作比例限制为 15%。LoRA 第二轮训练共执行 300 个优化步，有效 batch size 为 16，损失由 7.715 降至最终记录值 0.744，峰值显存为 4.34GB。

闭环评测首先使用不含任何专家恢复的纯学生协议。四种本地裁剪模型在原五种子协议中均未完成任务；第二轮蒸馏 LoRA 在新增的 700 步 seed 0 评测中也未完成任务。因此，本节严格保留纯学生的零成功结果，未将系统级恢复伪装成模型成功。随后按照已批准的回退方案加入显式专家恢复控制器：当模型夹爪状态与任务阶段冲突、平移动作与恢复动作最大分量误差超过 0.15，或处于强制闭合/释放阶段时才接管。实验发现玻璃杯的稳定碰撞壳距离为 13.8–15.0mm，原 10mm 接触阈值会使 seed 3 永久停在下降阶段，因此将接触判定校准为 16mm，并对四种方法以完全相同的种子、700 步上限和接管规则重新评测。

表 4-7 给出了真实本机资源占用、纯学生结果和混合闭环结果。四种方法在专家恢复协议下均达到 5/5 成功。由于该成功率已饱和，区分方法自主性的关键指标是 OpenVLA 直接控制占比：LoRA 蒸馏模型达到 66.9%，高于 Frozen-Vision 的 61.3%、Last-Layer-Only 的 55.3% 和 Full Fine-tuning 的 45.0%。同时，LoRA 仅训练 40.9M 参数（占裁剪模型 2.18%），峰值显存 4.34GB，是四种方法中资源需求最低的方法。因此，在本项目单 GPU 条件下，LoRA 仍是效率与闭环自主性综合最优的微调方法。

**表 4-7 OpenVLA-4L 微调方法真实对比**

{table}

需要强调的是，表中的“专家恢复成功率”属于 OpenVLA 与规则恢复器组成的系统级指标；“纯学生成功率”才表示裁剪模型独立完成任务的能力。该区分避免将恢复控制器的贡献错误归因于 OpenVLA。训练、DAgger 采集以及成功闭环均保留了日志、JSON 报告和相机帧，可复核每个种子的动作来源与成功状态。

**图 4-18 视觉—动作蒸馏训练与闭环贡献过程**

![视觉—动作蒸馏训练过程]({images['training'].as_posix()})

**图 4-19 真实 DAgger 与成功闭环相机帧**

![真实实验过程帧]({images['rollout'].as_posix()})

**图 4-20 OpenVLA 微调方法结果表**

![OpenVLA微调结果表]({images['table'].as_posix()})
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_results()
    images = {
        "training": OUT / "4.6.1_distillation_training_process.png",
        "rollout": OUT / "4.6.1_real_rollout_process.png",
        "table": OUT / "4.6.1_distillation_result_table.png",
    }
    render_training_process(rows, images["training"])
    render_real_rollout(images["rollout"])
    render_result_table(rows, images["table"])
    summary = {
        "schema_version": "openvla_461_distillation_final_v1",
        "task": "RoboCasa PickPlaceCounterToCabinet glass_cup",
        "base_model": "project-adapted OpenVLA-4L",
        "hardware": "NVIDIA GeForce RTX 3090 24GB",
        "contact_tolerance_m": 0.016,
        "max_translation_error": 0.15,
        "rows": rows,
        "images": {key: str(path.resolve()) for key, path in images.items()},
    }
    summary_path = OUT / "4.6.1_distillation_final_summary.json"
    markdown_path = OUT / "4.6.1_distillation_final.md"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(markdown(rows, images), encoding="utf-8")
    print(summary_path.resolve())
    print(markdown_path.resolve())
    for path in images.values():
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
