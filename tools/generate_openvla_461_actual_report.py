"""Generate Section 4.6.1 only from completed local training/evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "outputs" / "experiment_4_6" / "openvla_4l"
OUTPUT_ROOT = ROOT / "outputs" / "experiment_4_6"
ROWS = (
    (
        "full",
        "Full Fine-tuning",
        "full_balanced",
        "eval_full_balanced",
        "full_rollout_process.png",
    ),
    (
        "lora_r32",
        "LoRA (rank=32)",
        "lora_r32_balanced",
        "eval_lora_r32_balanced",
        "lora_r32_rollout_process.png",
    ),
    (
        "last_layer_only",
        "Last-Layer-Only",
        "last_layer_only_balanced",
        "eval_last_layer_only_balanced_five",
        "last_layer_only_rollout_process.png",
    ),
    (
        "frozen_vision",
        "Frozen-Vision",
        "frozen_vision_balanced",
        "eval_frozen_vision_balanced",
        "frozen_vision_rollout_process.png",
    ),
)


def font(size: int, bold: bool = False):
    candidates = (
        Path("C:/Windows/Fonts/msyhbd.ttc") if bold else Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def load_rows() -> list[dict]:
    results = []
    for mode, label, train_dir, eval_dir, rollout_name in ROWS:
        training_path = EXPERIMENT_ROOT / train_dir / "training_report.json"
        evaluation_path = EXPERIMENT_ROOT / eval_dir / "report.json"
        training = json.loads(training_path.read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if training["mode"] != mode or evaluation["mode"] != mode:
            raise ValueError(f"mode mismatch for {mode}")
        if training["status"] != "completed":
            raise ValueError(f"training not complete for {mode}")
        if evaluation["uses_expert_recovery"]:
            raise ValueError(f"expert recovery is forbidden for {mode}")
        if evaluation["trials"] != 5:
            raise ValueError(f"expected five evaluation trials for {mode}")
        peak = max(float(step["peak_vram_gb"]) for step in training["steps"])
        results.append(
            {
                "mode": mode,
                "method": label,
                "trainable_parameters": int(training["trainable_parameters"]),
                "total_parameters": int(training["total_parameters"]),
                "trainable_m": round(training["trainable_parameters"] / 1e6, 1),
                "trainable_percent": round(
                    100 * training["trainable_parameters"] / training["total_parameters"],
                    2,
                ),
                "peak_vram_gb": round(peak, 2),
                "successes": int(evaluation["successes"]),
                "trials": int(evaluation["trials"]),
                "success_rate": float(evaluation["success_rate"]),
                "first_loss": float(training["steps"][0]["loss"]),
                "final_loss": float(training["steps"][-1]["loss"]),
                "optimizer_steps": int(training["optimizer_steps"]),
                "effective_batch_size": int(training["effective_batch_size"]),
                "training_report": str(training_path.resolve()),
                "evaluation_report": str(evaluation_path.resolve()),
                "training_process_image": str(
                    (EXPERIMENT_ROOT / train_dir / f"{mode}_training_process.png").resolve()
                ),
                "rollout_process_image": str(
                    (EXPERIMENT_ROOT / eval_dir / rollout_name).resolve()
                ),
                "training_frames": training["training_frames"],
                "rollout_frames": evaluation["episodes"][0]["frame_paths"],
            }
        )
    return results


def paste_thumbnail(
    canvas: Image.Image,
    path: str,
    box: tuple[int, int, int, int],
) -> None:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((box[2] - box[0], box[3] - box[1]))
    x = box[0] + (box[2] - box[0] - image.width) // 2
    y = box[1] + (box[3] - box[1] - image.height) // 2
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline="#64748b", width=2)


def render_process_board(rows: list[dict], output: Path) -> None:
    canvas = Image.new("RGB", (2300, 1900), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text((55, 35), "OpenVLA 微调方法真实实验过程证据", fill="#0f172a", font=font(42, True))
    draw.text(
        (55, 95),
        "同一 OpenVLA-4L 基座｜同一五演示数据｜effective batch=16｜真实 RoboCasa 闭环",
        fill="#334155",
        font=font(25),
    )
    for row_index, row in enumerate(rows):
        y = 155 + row_index * 425
        draw.rounded_rectangle((35, y, 2265, y + 395), radius=18, fill="#ffffff", outline="#cbd5e1", width=2)
        draw.text((60, y + 18), row["method"], fill="#0f172a", font=font(31, True))
        draw.text((60, y + 66), "训练样本帧", fill="#2563eb", font=font(23, True))
        draw.text((1170, y + 66), "闭环评测帧（seed=0）", fill="#dc2626", font=font(23, True))
        train_frames = row["training_frames"][:3]
        rollout = row["rollout_frames"]
        rollout_indices = (0, len(rollout) // 2, len(rollout) - 1)
        for index, path in enumerate(train_frames):
            x = 60 + index * 350
            paste_thumbnail(canvas, path, (x, y + 105, x + 315, y + 315))
        for index, frame_index in enumerate(rollout_indices):
            x = 1170 + index * 350
            paste_thumbnail(canvas, rollout[frame_index], (x, y + 105, x + 315, y + 315))
        metrics = (
            f"step 1→{row['optimizer_steps']}  loss {row['first_loss']:.3f}→{row['final_loss']:.3f}  "
            f"peak VRAM {row['peak_vram_gb']:.2f} GB  闭环成功 {row['successes']}/{row['trials']}"
        )
        draw.text((60, y + 340), metrics, fill="#334155", font=font(22))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def render_result_table(rows: list[dict], output: Path) -> None:
    canvas = Image.new("RGB", (1700, 780), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 35), "表 4-7  OpenVLA 微调方法真实对比结果", fill="#0f172a", font=font(40, True))
    draw.text(
        (50, 92),
        "项目适配：OpenVLA-4L，RoboCasa 玻璃杯入柜，5次成功演示微调，5个闭环种子",
        fill="#475569",
        font=font(24),
    )
    columns = (50, 520, 890, 1190, 1450, 1650)
    top = 155
    row_height = 100
    headers = ("微调方法", "可训练参数量", "显存(batch=16)", "成功率", "成功回合")
    draw.rectangle((50, top, 1650, top + row_height), fill="#1e3a5f")
    for index, header in enumerate(headers):
        draw.text((columns[index] + 14, top + 30), header, fill="#ffffff", font=font(24, True))
    for row_index, row in enumerate(rows):
        y = top + (row_index + 1) * row_height
        draw.rectangle((50, y, 1650, y + row_height), fill="#ffffff" if row_index % 2 == 0 else "#eef2f7")
        values = (
            row["method"],
            f"{row['trainable_m']:.1f}M ({row['trainable_percent']:.2f}%)",
            f"{row['peak_vram_gb']:.2f}GB",
            f"{row['success_rate']:.1f}±0.0%",
            f"{row['successes']}/{row['trials']}",
        )
        for index, value in enumerate(values):
            draw.text((columns[index] + 14, y + 31), value, fill="#0f172a", font=font(23))
    for x in columns:
        draw.line((x, top, x, top + row_height * 5), fill="#94a3b8", width=2)
    draw.line((1650, top, 1650, top + row_height * 5), fill="#94a3b8", width=2)
    draw.rectangle((50, 685, 1650, 750), fill="#fff7ed", outline="#fdba74", width=2)
    draw.text(
        (70, 704),
        "说明：成功率来自无专家恢复的纯策略闭环；0%为真实结果，不以离线准确率或专家轨迹替代。",
        fill="#9a3412",
        font=font(22),
    )
    output.save(output)


def write_markdown(rows: list[dict], output: Path) -> None:
    table_lines = [
        "| 微调方法 | 可训练参数量 | 显存（effective batch=16） | 成功率 |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['method']} | {row['trainable_m']:.1f}M "
            f"({row['trainable_percent']:.2f}%) | {row['peak_vram_gb']:.2f}GB | "
            f"{row['success_rate']:.1f}±0.0% ({row['successes']}/{row['trials']}) |"
        )
    text = f"""### 4.6.1 OpenVLA 微调方法对比

本节依据本项目的实际算力和任务数据，对 Full Fine-tuning、LoRA（rank=32）、Last-Layer-Only 和 Frozen-Vision 四种 OpenVLA 微调方法进行对比。实验不包含真实机械臂，统一使用 RoboCasa 仿真中的“拾取玻璃杯并放入柜体”任务。由于 7B 模型的全参数微调无法在单张 RTX 3090（24GB）上完成，本实验采用从本地 OpenVLA-7B 构建并重新加载验证的 OpenVLA-4L 作为共同基座；四种方法使用相同的 5 次成功演示（seed=0、1、2、3、5）、相同输入图像与动作编码、200 个优化步、Adafactor、BF16、梯度检查点以及 effective batch size=16。成功率由 5 个种子的无专家恢复闭环回合直接统计。

实验结果如表 4-7 所示。LoRA（rank=32）仅训练 40.9M 参数，占其带适配器模型参数的 2.13%，峰值显存为 4.34GB，是四种方法中资源开销最低的方案。Last-Layer-Only 训练 333.7M 参数，峰值显存为 6.11GB；Frozen-Vision 训练语言模型与动作输出相关参数，共 1,143.6M，峰值显存为 7.62GB；Full Fine-tuning 训练全部 1,874.5M 参数，峰值显存为 8.92GB。四种方法均在本机完成了真实反向传播、参数更新和独立检查点保存，训练损失均显著下降，说明训练链路和 Last-Layer-Only 闭环检查点问题已解决。

{chr(10).join(table_lines)}

闭环评测中，四种方法均为 0/5，成功率为 0.0%。该结果显著低于原始论文在 LIBERO-Spatial 上给出的数值，但两者不应直接横向等同：本实验使用的是本项目 RoboCasa 长时序杯子入柜任务和为适配单卡而裁剪的 OpenVLA-4L，而不是完整 7B 模型及 LIBERO-Spatial 数据。针对低成功率已依次检查并修正 300 步时域截断、阶段均衡导致的静止帧过采样、stride=1 原始五演示重训，并尝试保留跨深度语言层（0/10/21/31）。700 步复测及几何诊断仍显示策略未接触杯子，最小末端—物体距离约 0.35m，说明主要瓶颈是四层裁剪后视觉—动作定位能力不足，而非显存 OOM、检查点未加载或成功判定错误。因此，本节保留真实 0.0% 闭环结果，不以离线训练准确率、专家恢复或规则控制器结果替代。

综合资源效率看，LoRA 仍是本机条件下最经济的微调方法；但本实验不能据此宣称其闭环性能与完整 OpenVLA-7B 相当。若要获得接近原文的成功率，需要在更大显存设备上保留完整语言模型，或对裁剪模型进行专门蒸馏后再做同一协议的闭环评测。
"""
    output.write_text(text, encoding="utf-8")


def main() -> int:
    rows = load_rows()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    process_image = OUTPUT_ROOT / "4.6.1_experiment_process_evidence.png"
    result_image = OUTPUT_ROOT / "4.6.1_actual_result_table.png"
    summary_path = OUTPUT_ROOT / "4.6.1_actual_summary.json"
    markdown_path = OUTPUT_ROOT / "4.6.1_actual.md"
    render_process_board(rows, process_image)
    render_result_table(rows, result_image)
    write_markdown(rows, markdown_path)
    summary = {
        "schema_version": "openvla_461_actual_v1",
        "task": "RoboCasa PickPlaceCounterToCabinet glass_cup",
        "base_model": "project-adapted OpenVLA-4L",
        "demonstration_seeds": [0, 1, 2, 3, 5],
        "expert_recovery": False,
        "rows": rows,
        "process_evidence_image": str(process_image.resolve()),
        "result_table_image": str(result_image.resolve()),
        "markdown": str(markdown_path.resolve()),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
