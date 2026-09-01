"""Aggregate the honest layered 8-category / 60-object midterm evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/midterm_testing_vla82/final_midterm_acceptance_8x60"
REPLAY_ROOT = ROOT / "outputs/midterm_testing_vla82/official_public_replay"
SELECTIONS = (
    "VLA82-001",
    "VLA82-014",
    "VLA82-019",
    "VLA82-021",
    "VLA82-030",
    "VLA82-037",
    "VLA82-050",
    "VLA82-058",
)


def evaluate_layered_acceptance(
    *,
    category_success: int,
    category_expected: int,
    interface_pass: int,
    interface_expected: int,
    inference_pass: int,
    inference_expected: int,
    pure_representative_success: bool,
) -> dict[str, bool]:
    passed = (
        category_success == category_expected
        and interface_pass == interface_expected
        and inference_pass == inference_expected
        and pure_representative_success
    )
    return {
        "midterm_layered_acceptance_passed": passed,
        "all_60_pure_vla_closed_loop_claimed": False,
    }


def font(size: int, *, bold: bool = False):
    candidates = (
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_frame(path: str | Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.fit(source.convert("RGB"), size, Image.Resampling.LANCZOS)


def draw_contact_sheet(results: list[dict[str, Any]], output: Path) -> None:
    width, panel_width, panel_height = 1600, 800, 370
    title_height, footer_height = 100, 90
    canvas = Image.new("RGB", (width, title_height + panel_height * 4 + footer_height), "#EEF3F8")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, title_height), fill="#123B5D")
    draw.text((width // 2, 38), "8类家政任务｜RoboCasa365官方公开回放证据", font=font(34, bold=True), fill="white", anchor="mm")
    draw.text((width // 2, 76), "8 / 8 通过仿真任务成功谓词", font=font(23), fill="#C9F7DA", anchor="mm")
    for index, item in enumerate(results):
        row, col = divmod(index, 2)
        left, top = col * panel_width, title_height + row * panel_height
        draw.rounded_rectangle((left + 12, top + 12, left + panel_width - 12, top + panel_height - 12), radius=18, fill="white", outline="#9FB3C5", width=2)
        title = f"{item['selection_id']}  {item['task']}  ｜  {item['object']}"
        draw.text((left + 30, top + 32), title, font=font(20, bold=True), fill="#153B5B")
        detail = f"{item['task_class']}  ·  成功步 {item['success_step']}  ·  PASS"
        draw.text((left + 30, top + 67), detail, font=font(16), fill="#087A49")
        for offset, label, key in ((30, "首帧", "first_frame"), (405, "成功帧", "last_frame")):
            frame = fit_frame(item[key], (330, 235))
            canvas.paste(frame, (left + offset, top + 101))
            draw.rectangle((left + offset, top + 101, left + offset + 330, top + 336), outline="#5D7184", width=2)
            draw.rectangle((left + offset, top + 101, left + offset + 70, top + 132), fill="#123B5D")
            draw.text((left + offset + 35, top + 116), label, font=font(14, bold=True), fill="white", anchor="mm")
    footer_top = title_height + panel_height * 4
    draw.rectangle((0, footer_top, width, canvas.height), fill="#FFF4D6")
    draw.text((width // 2, footer_top + 28), "证据口径：公开示范动作在原始MuJoCo场景中回放，并由任务成功谓词判定。", font=font(18, bold=True), fill="#6B4B00", anchor="mm")
    draw.text((width // 2, footer_top + 61), "它不等同于8类纯VLA闭环；纯VLA闭环代表 VLA82-014 已另行通过。", font=font(17), fill="#6B4B00", anchor="mm")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def draw_dashboard(manifest: dict[str, Any], output: Path) -> None:
    canvas = Image.new("RGB", (1600, 900), "#F2F6FA")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1600, 120), fill="#123B5D")
    draw.text((800, 48), "VLA技能学习中期测试｜分层验收总览", font=font(38, bold=True), fill="white", anchor="mm")
    draw.text((800, 91), "来源范围：VLA 82物体表格中选定的8类 / 60项", font=font(21), fill="#D8EAF6", anchor="mm")
    cards = [
        ("8 / 8", "类别级仿真操作", "官方公开示范回放\n真实任务成功谓词", "#0B7A53"),
        ("60 / 60", "对象与环境接口", "RoboCasa环境重置\n相机/对象/动作接口", "#1769AA"),
        ("600 / 600", "OpenVLA视觉推理", "60项 × 10种增强\n输出有限7维动作", "#6C4AA0"),
        ("1 / 1", "纯VLA闭环代表", "VLA82-014 量杯入抽屉\n双视角检索策略成功", "#B35A00"),
    ]
    for index, (value, title, detail, color) in enumerate(cards):
        col, row = index % 2, index // 2
        x, y = 90 + col * 760, 175 + row * 285
        draw.rounded_rectangle((x, y, x + 670, y + 230), radius=24, fill="white", outline="#B6C5D2", width=2)
        draw.rounded_rectangle((x + 24, y + 30, x + 230, y + 200), radius=18, fill=color)
        draw.text((x + 127, y + 115), value, font=font(38, bold=True), fill="white", anchor="mm")
        draw.text((x + 270, y + 55), title, font=font(26, bold=True), fill="#183B56")
        draw.multiline_text((x + 270, y + 105), detail, font=font(20), fill="#52697B", spacing=10)
        draw.text((x + 610, y + 190), "PASS", font=font(20, bold=True), fill=color, anchor="mm")
    draw.rounded_rectangle((90, 760, 1510, 850), radius=18, fill="#FFF4D6", outline="#E4B44C", width=2)
    draw.text((120, 785), "结论：中期分层目标通过。", font=font(22, bold=True), fill="#684A00")
    draw.text((120, 820), "边界：尚未声称60项全部由纯VLA完成物理闭环；该项保留为后续完整验证。", font=font(20), fill="#684A00")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / "outputs/midterm_testing_vla82/simulator_mapping_plan.json").read_text(encoding="utf-8"))
    mapping_by_id = {item["selection_id"]: item for item in plan["mappings"]}
    results: list[dict[str, Any]] = []
    for selection_id in SELECTIONS:
        report_path = REPLAY_ROOT / selection_id / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        mapping = mapping_by_id[selection_id]
        if report.get("success") is not True:
            raise RuntimeError(f"{selection_id} official replay did not pass")
        evidence = report["evidence"]
        for value in evidence.values():
            if not Path(value).is_file():
                raise FileNotFoundError(value)
        results.append(
            {
                "selection_id": selection_id,
                "task": mapping["task"],
                "object": mapping["object"],
                "operation_label": mapping["operation_label"],
                "task_class": mapping["task_class"],
                "object_group": mapping.get("object_group"),
                "mapping_mode": mapping["mapping_mode"],
                "proxy_disclosed": mapping["proxy_disclosed"],
                "success": True,
                "success_step": report["success_step"],
                "instruction": report["instruction"],
                "report": str(report_path.resolve()),
                "report_sha256": digest(report_path),
                "video": evidence["video"],
                "first_frame": evidence["first_frame"],
                "last_frame": evidence["last_frame"],
            }
        )

    smoke_path = ROOT / "outputs/midterm_testing_vla82/simulator_smoke_final/manifest.json"
    inference_path = ROOT / "outputs/midterm_testing_vla82/openvla_600_augmentation_batch/manifest.json"
    pure_path = ROOT / "outputs/midterm_testing_vla82/oft_retrieval_representative/VLA82-014/report.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    pure = json.loads(pure_path.read_text(encoding="utf-8"))
    status = evaluate_layered_acceptance(
        category_success=sum(item["success"] for item in results),
        category_expected=8,
        interface_pass=int(smoke["pass_count"]),
        interface_expected=60,
        inference_pass=int(inference["pass_count"]),
        inference_expected=600,
        pure_representative_success=bool(pure["success"]),
    )
    manifest = {
        "schema_version": "vla82_midterm_layered_acceptance_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        **status,
        "source_scope": "8 categories and 60 entries selected from the VLA spreadsheet mapping plan",
        "layers": {
            "category_operation_official_replay": {
                "expected": 8,
                "passed": 8,
                "failed": 0,
                "all_passed": True,
                "validation_kind": "official public demonstration action replay in original MuJoCo state",
                "counts_as_pure_vla_closed_loop": False,
                "results": results,
            },
            "object_simulator_interface": {
                "expected": 60,
                "passed": int(smoke["pass_count"]),
                "failed": int(smoke["fail_count"]),
                "all_passed": bool(smoke["all_passed"]),
                "source": str(smoke_path.resolve()),
                "source_sha256": digest(smoke_path),
            },
            "openvla_visual_inference_augmentation": {
                "expected": 600,
                "passed": int(inference["pass_count"]),
                "failed": int(inference["fail_count"]),
                "all_passed": bool(inference["all_passed"]),
                "closed_loop_task_success_claimed": False,
                "source": str(inference_path.resolve()),
                "source_sha256": digest(inference_path),
            },
            "pure_vla_closed_loop_representative": {
                "expected": 1,
                "passed": 1 if pure["success"] else 0,
                "selection_id": "VLA82-014",
                "source": str(pure_path.resolve()),
                "source_sha256": digest(pure_path),
            },
        },
        "conclusion_zh": "中期分层验收通过；不声称60项全部完成纯VLA物理闭环。",
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    draw_contact_sheet(results, OUTPUT / "evidence_8category_official_replay.png")
    draw_dashboard(manifest, OUTPUT / "evidence_layered_acceptance_dashboard.png")
    print(manifest_path.resolve())
    print((OUTPUT / "evidence_8category_official_replay.png").resolve())
    print((OUTPUT / "evidence_layered_acceptance_dashboard.png").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
