"""Run the 4.6.1 report generator with the corrected result-table save call."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from tools import generate_openvla_461_actual_report as report


def render_result_table(rows: list[dict], output: Path) -> None:
    canvas = Image.new("RGB", (1700, 780), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 35), "表 4-7  OpenVLA 微调方法真实对比结果", fill="#0f172a", font=report.font(40, True))
    draw.text(
        (50, 92),
        "项目适配：OpenVLA-4L，RoboCasa 玻璃杯入柜，5次成功演示微调，5个闭环种子",
        fill="#475569",
        font=report.font(24),
    )
    columns = (50, 520, 890, 1190, 1450, 1650)
    top = 155
    row_height = 100
    headers = ("微调方法", "可训练参数量", "显存(batch=16)", "成功率", "成功回合")
    draw.rectangle((50, top, 1650, top + row_height), fill="#1e3a5f")
    for index, header in enumerate(headers):
        draw.text((columns[index] + 14, top + 30), header, fill="#ffffff", font=report.font(24, True))
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
            draw.text((columns[index] + 14, y + 31), value, fill="#0f172a", font=report.font(23))
    for x in columns:
        draw.line((x, top, x, top + row_height * 5), fill="#94a3b8", width=2)
    draw.line((1650, top, 1650, top + row_height * 5), fill="#94a3b8", width=2)
    draw.rectangle((50, 685, 1650, 750), fill="#fff7ed", outline="#fdba74", width=2)
    draw.text(
        (70, 704),
        "说明：成功率来自无专家恢复的纯策略闭环；0%为真实结果，不以离线准确率或专家轨迹替代。",
        fill="#9a3412",
        font=report.font(22),
    )
    canvas.save(output)


report.render_result_table = render_result_table


if __name__ == "__main__":
    raise SystemExit(report.main())
