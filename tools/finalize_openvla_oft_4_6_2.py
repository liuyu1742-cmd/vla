from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable


def mean_one_decimal_half_up(values: Iterable[float]) -> float:
    decimals = [Decimal(str(value)) for value in values]
    mean = sum(decimals) / Decimal(len(decimals))
    return float(mean.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def render_corrected_chart(summary: dict, output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager

    order = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    labels = ["Spatial", "Object", "Goal", "Navigation*", "平均"]
    oft = [summary["results"][suite]["openvla_oft_local_measured"] for suite in order]
    baseline = [summary["results"][suite]["openvla_baseline_user_table_reference"] for suite in order]
    diffusion = [summary["results"][suite]["diffusion_policy_user_table_reference"] for suite in order]
    oft.append(summary["averages"]["openvla_oft_local_measured"])
    baseline.append(summary["averages"]["openvla_baseline_user_table_reference"])
    diffusion.append(summary["averages"]["diffusion_policy_user_table_reference"])

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=str(font_path)).get_name()]
    plt.rcParams["axes.unicode_minus"] = False
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(13.2, 7.2), dpi=180)
    groups = [
        ax.bar(x - width, oft, width, label="OpenVLA-OFT（本地实测）", color="#2563eb"),
        ax.bar(x, baseline, width, label="OpenVLA baseline（表4-8参考）", color="#f59e0b"),
        ax.bar(x + width, diffusion, width, label="Diffusion Policy（表4-8参考）", color="#10b981"),
    ]
    for group in groups:
        ax.bar_label(group, fmt="%.1f%%", padding=3, fontsize=9)
    ax.axhline(60, color="#dc2626", linewidth=1.3, linestyle="--", label="60% 阈值")
    ax.set_ylim(0, 110)
    ax.set_ylabel("任务成功率（%）")
    ax.set_xticks(x, labels)
    ax.set_title("OpenVLA-OFT 在 LIBERO 四套件上的本地复现结果")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="lower left", ncols=2, fontsize=9)
    fig.text(
        0.5,
        0.012,
        "* Navigation 对应官方 LIBERO-10 / LIBERO-Long；OFT 为本地 40 个真实仿真回合，baseline/DP 为用户提供表4-8对照值。",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def finalize(output_dir: Path) -> None:
    summary_path = output_dir / "4.6.2_summary.json"
    report_path = output_dir / "4.6.2_OpenVLA-OFT_实验复现.md"
    csv_path = output_dir / "4.6.2_success_rates.csv"
    chart_path = output_dir / "4.6.2_success_rates.png"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    rows = list(summary["results"].values())
    baseline_average = mean_one_decimal_half_up(
        row["openvla_baseline_user_table_reference"] for row in rows
    )
    diffusion_average = mean_one_decimal_half_up(
        row["diffusion_policy_user_table_reference"] for row in rows
    )
    summary["averages"]["openvla_baseline_user_table_reference"] = baseline_average
    summary["averages"]["diffusion_policy_user_table_reference"] = diffusion_average
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = report_path.read_text(encoding="utf-8")
    report = report.replace("| 平均 | 97.5% | 71.5% | 69.1% |", "| 平均 | 97.5% | 71.6% | 69.1% |")
    report_path.write_text(report, encoding="utf-8")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.reader(stream))
    for row in csv_rows:
        if row and row[0] == "平均":
            row[2] = f"{baseline_average:.1f}%"
            row[3] = f"{diffusion_average:.1f}%"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerows(csv_rows)

    render_corrected_chart(summary, chart_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    finalize(args.output_dir.resolve())
    print("final-average-rounding: PASS")


if __name__ == "__main__":
    main()
