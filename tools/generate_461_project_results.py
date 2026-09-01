"""Generate the evidence-backed Section 4.6.1 report and result figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "outputs" / "experiment_4_6" / "finetune_modes_v2"
LIBERO_REPORT = (
    ROOT
    / "outputs"
    / "experiment_4_2_5"
    / "libero_goal_pure_four_tasks"
    / "report.json"
)
OUT_DIR = ROOT / "outputs" / "experiment_4_6"


def load_results() -> dict:
    ordered = ("full", "lora_r32", "last_layer_only", "frozen_vision")
    probes = {
        mode: json.loads((PROBE_DIR / f"{mode}.json").read_text(encoding="utf-8"))
        for mode in ordered
    }
    libero = json.loads(LIBERO_REPORT.read_text(encoding="utf-8"))
    episodes = libero["episodes"]
    lora_successes = sum(bool(row["success"]) for row in episodes)
    return {
        "schema_version": "openvla_461_project_results_v1",
        "experiment_scope": "project-adapted simulation only",
        "resource_probe": {
            "dataset": "water_cup_expert",
            "task": "pick up the glass cup and place it in the cabinet",
            "demonstration_seeds": [0, 1, 2, 3, 5],
            "physical_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "effective_batch_size": 16,
            "gpu": "NVIDIA GeForce RTX 3090 24GB",
        },
        "rows": [probes[mode] for mode in ordered],
        "closed_loop_validation": {
            "method": "LoRA (rank=32)",
            "checkpoint": "models/openvla-7b-finetuned-libero-goal",
            "checkpoint_training": "LoRA r=32, documented in checkpoint README",
            "suite": "LIBERO-Goal",
            "tasks": [1, 2, 5, 7],
            "trials": len(episodes),
            "successes": lora_successes,
            "success_rate": 100.0 * lora_successes / len(episodes),
            "uses_expert_recovery": bool(libero["uses_expert_recovery"]),
            "source": str(LIBERO_REPORT),
        },
        "limitations": [
            "Full Fine-tuning and Frozen-Vision exceeded the 24GB device limit during a real backward pass.",
            "Last-Layer-Only fit the GPU, but the project has no completed matching closed-loop checkpoint.",
            "Only LoRA rank 32 has a completed pure-policy closed-loop result in the current project.",
            "The LoRA closed-loop checkpoint was trained on LIBERO-Goal; it is not the five-shot water-cup probe.",
        ],
    }


def render_figure(summary: dict, output: Path) -> None:
    labels = [row["method_label"] for row in summary["rows"]]
    trainable = [row["trainable_parameters"] / 1e6 for row in summary["rows"]]
    measured_vram = []
    colors = []
    for row in summary["rows"]:
        if row["status"] == "oom":
            measured_vram.append(24.0)
            colors.append("#b8b8b8")
        else:
            measured_vram.append(row["peak_vram_gb"])
            colors.append("#4472c4")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3), dpi=180)
    y = np.arange(len(labels))

    axes[0].barh(y, trainable, color="#5b9bd5")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xscale("log")
    axes[0].set_xlabel("可训练参数量（M，对数坐标）")
    axes[0].set_title("项目 OpenVLA-7B 参数规模")
    for index, value in enumerate(trainable):
        axes[0].text(value * 1.05, index, f"{value:,.1f}M", va="center", fontsize=9)

    axes[1].barh(y, measured_vram, color=colors)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].axvline(24, color="#c00000", linestyle="--", linewidth=1.4)
    axes[1].set_xlim(0, 27)
    axes[1].set_xlabel("峰值显存（GB）")
    axes[1].set_title("有效 batch=16 的真实训练探测")
    for index, row in enumerate(summary["rows"]):
        text = (
            "OOM（>24GB）"
            if row["status"] == "oom"
            else f"{row['peak_vram_gb']:.2f}GB"
        )
        axes[1].text(measured_vram[index] + 0.35, index, text, va="center", fontsize=9)

    success = summary["closed_loop_validation"]["success_rate"]
    axes[2].bar(["LoRA\n(rank=32)"], [success], color="#70ad47", width=0.55)
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel("任务成功率（%）")
    axes[2].set_title("项目已有纯策略闭环复核")
    axes[2].text(0, success + 2.5, f"{success:.1f}%\n(38/40)", ha="center", fontsize=11)
    axes[2].text(
        0,
        8,
        "LIBERO-Goal 子集\n无专家恢复",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )

    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("OpenVLA 微调方法对比：本项目实测结果", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def render_chapter(summary: dict) -> str:
    rows = {row["mode"]: row for row in summary["rows"]}

    def params(mode: str) -> str:
        return f"{rows[mode]['trainable_parameters'] / 1e6:,.1f}M"

    def vram(mode: str) -> str:
        row = rows[mode]
        return (
            ">24GB（OOM）"
            if row["status"] == "oom"
            else f"{row['peak_vram_gb']:.2f}GB"
        )

    return f"""### 4.6.1 OpenVLA 微调方法对比

为避免直接套用与本项目条件不一致的论文数据，本实验将图中四种方法完整保留，即 Full Fine-tuning、LoRA（rank=32）、Last-Layer-Only 和 Frozen-Vision，并将实验对象调整为本项目本地 OpenVLA-7B 与五条水杯抓取放置示教。五条示教分别来自 seed 0、1、2、3 和 5，任务指令为“抓取玻璃杯并放入柜体”。实验平台为单张 NVIDIA GeForce RTX 3090（24GB），物理 batch size 为 1，通过 16 次梯度累积得到有效 batch size 16。每种方法均加载同一基础模型、同一真实图像—动作样本并执行反向传播；显存不足按 OOM 记录，不用估算值替代。

表 4-7 给出本项目实测结果。Full Fine-tuning 需要训练全部 {params('full')} 参数，反向传播时显存需求超过 24GB，实测发生 OOM；Frozen-Vision 虽冻结视觉编码器，仍需训练 {params('frozen_vision')} 参数，同样超过单卡容量。LoRA（rank=32）仅训练 {params('lora_r32')} 参数，峰值显存为 {vram('lora_r32')}；Last-Layer-Only 训练 {params('last_layer_only')} 参数，峰值显存为 {vram('last_layer_only')}。两种方法均在 24GB 显存内完成一次有效 batch=16 的真实优化更新。

| 微调方法 | 可训练参数量 | 显存（有效 batch=16） | 本项目任务成功率 |
|---|---:|---:|---:|
| Full Fine-tuning | {params('full')} | {vram('full')} | 未评测（训练 OOM） |
| LoRA（rank=32） | {params('lora_r32')} | {vram('lora_r32')} | 95.0%（38/40） |
| Last-Layer-Only | {params('last_layer_only')} | {vram('last_layer_only')} | 未评测（无匹配闭环检查点） |
| Frozen-Vision | {params('frozen_vision')} | {vram('frozen_vision')} | 未评测（训练 OOM） |

LoRA 的成功率来自项目已有且已完成的纯策略 LIBERO-Goal 闭环复核：检查点 `models/openvla-7b-finetuned-libero-goal` 的 README 明确记录其采用 LoRA（rank=32）微调；评测覆盖任务 1、2、5、7，每个任务 10 回合，共成功 38/40 回合，未使用专家恢复。该闭环检查点使用 LIBERO-Goal 数据训练，因此成功率不应解释为五条水杯示教训练后的结果；五条示教仅用于本次四种参数模式的统一资源探测。当前项目没有 Full Fine-tuning、Last-Layer-Only 和 Frozen-Vision 的同协议闭环检查点，故不填入推测成功率。

综合本项目的真实硬件和已有闭环证据，LoRA（rank=32）是唯一同时满足单卡可训练、参数效率高且已有高成功率纯策略验证的方法。Last-Layer-Only 虽然显存占用与 LoRA 接近，但可训练参数约为 LoRA 的 3.0 倍，且当前没有同协议闭环结果；Full Fine-tuning 与 Frozen-Vision 则无法在 24GB 单卡上完成有效 batch=16 的训练。因此，本项目后续 OpenVLA 任务适配采用 LoRA（rank=32）。
"""


def main() -> int:
    summary = load_results()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "4.6.1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_figure(summary, OUT_DIR / "4.6.1_OpenVLA_finetune_comparison.png")
    (OUT_DIR / "4.6.1_实验验证.md").write_text(
        render_chapter(summary), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

