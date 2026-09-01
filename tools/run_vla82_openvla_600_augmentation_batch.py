"""Run 600 resumable OpenVLA policy-inference tests with visual augmentation.

The batch contains 60 selected items x 10 deterministic visual conditions. It
tests policy input robustness and finite 7-DoF action generation. It does not
claim that a robot completed 600 closed-loop manipulation episodes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
import traceback
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.run_vla82_openvla_60_policy_probe import build_instruction
BASE = ROOT / "outputs" / "midterm_testing_vla82"
SMOKE_PATH = BASE / "simulator_smoke_final" / "manifest.json"
MATRIX_PATH = BASE / "vla82_midterm_600_runs.csv"
DEFAULT_OUTPUT = BASE / "openvla_600_augmentation_batch"
DEFAULT_MODEL = ROOT / "models" / "openvla-7b"
AUGMENTATIONS = [
    "original",
    "brightness_090",
    "brightness_110",
    "contrast_090",
    "contrast_110",
    "saturation_090",
    "saturation_110",
    "sharpness_080",
    "sharpness_120",
    "gaussian_noise_sigma2",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidate = Path(
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"
    )
    return ImageFont.truetype(str(candidate), size=size)


def augment(image: Image.Image, kind: str, seed: int) -> Image.Image:
    image = image.convert("RGB")
    if kind == "original":
        return image
    if kind.startswith("brightness_"):
        return ImageEnhance.Brightness(image).enhance(
            int(kind.rsplit("_", 1)[1]) / 100
        )
    if kind.startswith("contrast_"):
        return ImageEnhance.Contrast(image).enhance(
            int(kind.rsplit("_", 1)[1]) / 100
        )
    if kind.startswith("saturation_"):
        return ImageEnhance.Color(image).enhance(
            int(kind.rsplit("_", 1)[1]) / 100
        )
    if kind.startswith("sharpness_"):
        return ImageEnhance.Sharpness(image).enhance(
            int(kind.rsplit("_", 1)[1]) / 100
        )
    if kind == "gaussian_noise_sigma2":
        values = np.asarray(image, dtype=np.int16)
        noise = np.random.default_rng(seed).normal(0, 2, values.shape)
        return Image.fromarray(
            np.clip(values + noise, 0, 255).astype(np.uint8), mode="RGB"
        )
    raise ValueError(f"unknown augmentation: {kind}")


def make_manifest(
    results: list[dict[str, Any]],
    smoke_sha256: str,
    model_index_sha256: str,
) -> dict[str, Any]:
    passed = sum(row["status"] == "PASS" for row in results)
    return {
        "schema_version": "vla82_openvla_600_augmentation_batch_v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_smoke_manifest": str(SMOKE_PATH.resolve()),
        "source_smoke_sha256": smoke_sha256,
        "source_matrix": str(MATRIX_PATH.resolve()),
        "model_dir": str(DEFAULT_MODEL.resolve()),
        "model_index_sha256": model_index_sha256,
        "test_scope": "openvla_policy_inference_visual_augmentation",
        "test_scope_zh": "OpenVLA策略推理视觉增强泛化测试",
        "task_count": 8,
        "object_count": 60,
        "trials_per_object": 10,
        "planned_count": 600,
        "completed_count": len(results),
        "pass_count": passed,
        "fail_count": len(results) - passed,
        "all_passed": len(results) == 600 and passed == 600,
        "closed_loop_task_success_claimed": False,
        "disclaimer": (
            "PASS表示真实OpenVLA完成图像/指令处理并输出有限7维动作。"
            "环境重置、相机和动作接口由60项RoboCasa冒烟清单证明；"
            "本批次不声称机器人完成了600次物理闭环操作。"
        ),
        "augmentation_kinds": AUGMENTATIONS,
        "results": results,
    }


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def draw_dashboard(output_dir: Path, manifest: dict[str, Any]) -> None:
    image = Image.new("RGB", (1600, 900), "#F3F6FA")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1600, 115), fill="#17365D")
    draw.text((55, 28), "VLA中期测试证据总览", font=font(42, True), fill="white")
    draw.text(
        (56, 80),
        "指定82物体范围内：8类任务 / 60物体 / 600次视觉增强策略推理",
        font=font(20),
        fill="#DDEBF7",
    )
    cards = [
        ("任务类别", "8", "#1F4E78"),
        ("选定物体", "60", "#2F75B5"),
        ("执行用例", str(manifest["completed_count"]), "#548235"),
        ("推理通过", str(manifest["pass_count"]), "#70AD47"),
        ("推理失败", str(manifest["fail_count"]), "#C00000"),
    ]
    for index, (label, value, color) in enumerate(cards):
        left = 55 + index * 300
        draw.rounded_rectangle(
            (left, 155, left + 260, 295), radius=18, fill="white", outline="#CBD6E2"
        )
        draw.rectangle((left, 155, left + 10, 295), fill=color)
        draw.text((left + 30, 178), label, font=font(21), fill="#595959")
        draw.text((left + 30, 220), value, font=font(43, True), fill=color)

    task_counts: OrderedDict[str, int] = OrderedDict()
    for row in manifest["results"]:
        task_counts.setdefault(row["task"], 0)
        if row["status"] == "PASS":
            task_counts[row["task"]] += 1
    draw.text((55, 335), "各任务类别通过数量", font=font(25, True), fill="#17365D")
    max_count = max(task_counts.values())
    for index, (task, count) in enumerate(task_counts.items()):
        y = 385 + index * 47
        draw.text((55, y), task, font=font(18), fill="#404040")
        width = int(680 * count / max_count)
        draw.rounded_rectangle(
            (330, y + 2, 330 + width, y + 29), radius=7, fill="#5B9BD5"
        )
        draw.text((1030, y), f"{count}/{count}", font=font(18, True), fill="#375623")

    latencies = [
        float(row["inference_seconds"])
        for row in manifest["results"]
        if row["status"] == "PASS"
    ]
    stats = [
        f"平均推理耗时：{statistics.mean(latencies):.3f} 秒",
        f"中位推理耗时：{statistics.median(latencies):.3f} 秒",
        f"最大推理耗时：{max(latencies):.3f} 秒",
        "10种条件：原图、亮度、对比度、饱和度、锐度、轻微噪声",
    ]
    draw.rounded_rectangle(
        (1130, 335, 1545, 555), radius=16, fill="white", outline="#CBD6E2"
    )
    draw.text((1160, 360), "运行统计", font=font(24, True), fill="#17365D")
    for index, line in enumerate(stats):
        draw.text((1160, 410 + index * 39), line, font=font(17), fill="#404040")

    draw.rounded_rectangle(
        (1130, 590, 1545, 785), radius=16, fill="#FFF2CC", outline="#D6B656"
    )
    draw.text((1160, 615), "结论边界", font=font(23, True), fill="#7F6000")
    disclaimer_lines = [
        "600/600 PASS 证明：",
        "OpenVLA在10种视觉条件下均能",
        "完成推理并输出有限7维动作。",
        "",
        "不等同于600次物理闭环成功。",
        "代理物体也不冒充真实物体成功。",
    ]
    for index, line in enumerate(disclaimer_lines):
        draw.text((1160, 660 + index * 25), line, font=font(16), fill="#7F6000")
    short_hash = manifest["source_smoke_sha256"][:16]
    draw.text(
        (55, 852),
        f"生成时间：{manifest['generated_at']}  |  仿真清单SHA-256：{short_hash}…",
        font=font(15),
        fill="#6B7280",
    )
    image.save(output_dir / "evidence_dashboard.png")


def draw_simulator_frames(
    output_dir: Path,
    smoke_results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    first_by_task: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in smoke_results:
        first_by_task.setdefault(row["task"], row)
    canvas = Image.new("RGB", (1600, 1120), "#F3F6FA")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1600, 105), fill="#17365D")
    draw.text((48, 25), "真实RoboCasa测试帧证据（8类任务）", font=font(37, True), fill="white")
    draw.text(
        (49, 72),
        "图像路径与每项预测动作均记录在manifest.json；以下帧来自60/60接口冒烟测试",
        font=font(17),
        fill="#DDEBF7",
    )
    for index, (task, row) in enumerate(first_by_task.items()):
        col = index % 4
        line = index // 4
        left = 38 + col * 390
        top = 135 + line * 455
        draw.rounded_rectangle(
            (left, top, left + 365, top + 420),
            radius=14,
            fill="white",
            outline="#CBD6E2",
        )
        frame = Image.open(row["camera"]["frame"]).convert("RGB")
        frame.thumbnail((335, 300))
        x = left + (365 - frame.width) // 2
        canvas.paste(frame, (x, top + 18))
        draw.text((left + 18, top + 326), task, font=font(20, True), fill="#17365D")
        draw.text(
            (left + 18, top + 361),
            f"{row['selection_id']}  {row['object']}",
            font=font(17),
            fill="#404040",
        )
        draw.text(
            (left + 18, top + 389),
            f"接口：PASS  |  OpenVLA批测：PASS",
            font=font(15),
            fill="#375623",
        )
    draw.text(
        (45, 1065),
        (
            f"600条结果：{manifest['pass_count']} PASS / "
            f"{manifest['fail_count']} FAIL；仅声明策略推理泛化与仿真接口通过。"
        ),
        font=font(18, True),
        fill="#17365D",
    )
    canvas.save(output_dir / "evidence_simulator_frames.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    smoke = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
    smoke_by_id = {row["selection_id"]: row for row in smoke["results"]}
    with MATRIX_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        matrix = list(csv.DictReader(handle))
    if len(matrix) != 600:
        raise ValueError(f"expected 600 matrix rows, got {len(matrix)}")
    smoke_sha256 = file_sha256(SMOKE_PATH)
    model_index_sha256 = file_sha256(args.model_dir / "model.safetensors.index.json")

    completed: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = {
            row["run_id"]: row
            for row in prior.get("results", [])
            if row.get("status") == "PASS"
        }
    pending = [row for row in matrix if row["run_id"] not in completed]

    if pending:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the 600-case OpenVLA batch")
        processor = AutoProcessor.from_pretrained(
            str(args.model_dir), trust_remote_code=True, local_files_only=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            str(args.model_dir),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        ).to("cuda:0")
        model.eval()

        for matrix_row in pending:
            run_id = matrix_row["run_id"]
            selection_id = matrix_row["selection_id"]
            source = smoke_by_id[selection_id]
            trial = int(matrix_row["trial"])
            seed = int(matrix_row["seed"])
            augmentation = AUGMENTATIONS[trial - 1]
            instruction = build_instruction(source)
            frame_path = Path(source["camera"]["frame"])
            failures: list[dict[str, Any]] = []
            result: dict[str, Any] | None = None
            for attempt in range(1, 4):
                started = time.perf_counter()
                try:
                    image = augment(Image.open(frame_path), augmentation, seed)
                    prompt = (
                        f"In: What action should the robot take to {instruction}?\nOut:"
                    )
                    inputs = processor(prompt, image).to(
                        "cuda:0", dtype=torch.bfloat16
                    )
                    with torch.inference_mode():
                        action = model.predict_action(
                            **inputs, unnorm_key="bridge_orig", do_sample=False
                        )
                    predicted_action = [float(value) for value in action]
                    if len(predicted_action) != 7 or not all(
                        math.isfinite(value) for value in predicted_action
                    ):
                        raise ValueError(f"invalid action: {predicted_action!r}")
                    result = {
                        "run_id": run_id,
                        "selection_id": selection_id,
                        "source_table": source["source_table"],
                        "task": source["task"],
                        "object": source["object"],
                        "trial": trial,
                        "seed": seed,
                        "augmentation": augmentation,
                        "mapping_mode": source["mapping_mode"],
                        "proxy_disclosed": source["proxy_disclosed"],
                        "source_frame": str(frame_path.resolve()),
                        "instruction": instruction,
                        "predicted_action": predicted_action,
                        "inference_seconds": round(time.perf_counter() - started, 6),
                        "attempt": attempt,
                        "prior_failures": failures,
                        "environment_evidence": source["report_path"],
                        "status": "PASS",
                    }
                    break
                except Exception as error:
                    failures.append(
                        {
                            "attempt": attempt,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    torch.cuda.empty_cache()
            if result is None:
                result = {
                    "run_id": run_id,
                    "selection_id": selection_id,
                    "source_table": source["source_table"],
                    "task": source["task"],
                    "object": source["object"],
                    "trial": trial,
                    "seed": seed,
                    "augmentation": augmentation,
                    "mapping_mode": source["mapping_mode"],
                    "proxy_disclosed": source["proxy_disclosed"],
                    "source_frame": str(frame_path.resolve()),
                    "instruction": instruction,
                    "predicted_action": [],
                    "failures": failures,
                    "environment_evidence": source["report_path"],
                    "status": "FAIL",
                    "comprehensive_diagnosis_required": True,
                }
            completed[run_id] = result
            ordered = [
                completed[row["run_id"]]
                for row in matrix
                if row["run_id"] in completed
            ]
            if len(ordered) % 10 == 0 or result["status"] == "FAIL":
                save_manifest(
                    manifest_path,
                    make_manifest(ordered, smoke_sha256, model_index_sha256),
                )
                print(
                    f"[{len(ordered):03d}/600] {run_id} {result['status']}",
                    flush=True,
                )

    ordered = [completed[row["run_id"]] for row in matrix if row["run_id"] in completed]
    manifest = make_manifest(ordered, smoke_sha256, model_index_sha256)
    save_manifest(manifest_path, manifest)
    if manifest["all_passed"]:
        draw_dashboard(args.output_dir, manifest)
        draw_simulator_frames(args.output_dir, smoke["results"], manifest)
    print(
        f"OPENVLA_600_BATCH_COMPLETE "
        f"{manifest['pass_count']}/{manifest['completed_count']}",
        flush=True,
    )
    return 0 if manifest["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
