"""Run one resumable real OpenVLA action prediction for each selected VLA82 item.

This validates model loading, prompt/image processing, and finite 7-DoF action
generation on each verified simulator frame. It does not claim closed-loop task
success.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SMOKE_MANIFEST = (
    ROOT
    / "outputs"
    / "midterm_testing_vla82"
    / "simulator_smoke_final"
    / "manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "midterm_testing_vla82" / "openvla_60_policy_probe"
)
DEFAULT_MODEL = ROOT / "models" / "openvla-7b"


def build_instruction(result: dict[str, Any]) -> str:
    object_text = (
        result.get("manipulated_object_text")
        or result.get("object_group")
        or result["object"]
    )
    task_class = result["task_class"]
    templates = {
        "PickPlaceCounterToSink": f"pick up the {object_text} from the counter and place it in the sink",
        "PickPlaceCounterToCabinet": f"pick up the {object_text} from the counter and place it in the cabinet",
        "PickPlaceCabinetToCounter": f"pick up the {object_text} from the cabinet and place it on the counter",
        "PickPlaceCounterToDrawer": f"pick up the {object_text} from the counter and place it in the drawer",
        "TurnOnStove": "turn on the stove",
        "CloseOven": "close the oven door",
        "SlideToasterOvenRack": "slide the toaster oven rack fully inward",
        "SlideOvenRack": "slide the oven rack",
        "CloseDishwasher": "close the dishwasher door",
        "OpenDishwasher": "open the dishwasher door",
        "OpenCabinet": "open the cabinet door",
        "CloseCabinet": "close the cabinet door",
        "OpenDrawer": "open the drawer",
        "CloseDrawer": "close the drawer",
    }
    return templates.get(task_class, f"perform the requested manipulation of the {object_text}")


def write_manifest(output_path: Path, results: list[dict[str, Any]]) -> None:
    completed = len(results)
    passed = sum(row["status"] == "PASS" for row in results)
    manifest = {
        "schema_version": "vla82_openvla_60_policy_probe_v1",
        "source_smoke_manifest": str(SMOKE_MANIFEST.resolve()),
        "probe_scope": "single_action_policy_inference_on_verified_simulator_frame",
        "probe_scope_zh": "在已验证仿真相机帧上执行一次真实OpenVLA动作推理",
        "completed_count": completed,
        "pass_count": passed,
        "fail_count": completed - passed,
        "all_passed": completed == 60 and passed == 60,
        "closed_loop_success_claimed": False,
        "disclaimer": (
            "PASS仅表示该条目完成图像/指令处理并生成有限的7维动作；"
            "不表示机器人已完成闭环操作，也不把功能代理当作真实物体成功。"
        ),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest_path = args.output_dir / "manifest.json"
    smoke = json.loads(SMOKE_MANIFEST.read_text(encoding="utf-8"))
    source_results = smoke["results"]
    completed_by_id: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed_by_id = {
            row["selection_id"]: row
            for row in prior.get("results", [])
            if row.get("status") == "PASS"
        }
    pending = [
        row for row in source_results if row["selection_id"] not in completed_by_id
    ]
    if not pending:
        ordered = [completed_by_id[row["selection_id"]] for row in source_results]
        write_manifest(manifest_path, ordered)
        print("OPENVLA_POLICY_PROBE_COMPLETE 60/60", flush=True)
        return 0

    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the OpenVLA policy probe")
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

    for source in pending:
        selection_id = source["selection_id"]
        instruction = build_instruction(source)
        frame_path = Path(source["camera"]["frame"])
        failures: list[dict[str, Any]] = []
        result: dict[str, Any] | None = None
        for attempt in range(1, 4):
            started = time.perf_counter()
            try:
                prompt = (
                    f"In: What action should the robot take to {instruction}?\nOut:"
                )
                inputs = processor(
                    prompt, Image.open(frame_path).convert("RGB")
                ).to("cuda:0", dtype=torch.bfloat16)
                with torch.inference_mode():
                    action = model.predict_action(
                        **inputs, unnorm_key="bridge_orig", do_sample=False
                    )
                predicted_action = [float(value) for value in action]
                if len(predicted_action) != 7 or not all(
                    math.isfinite(value) for value in predicted_action
                ):
                    raise ValueError(
                        f"invalid predicted action: {predicted_action!r}"
                    )
                result = {
                    "selection_id": selection_id,
                    "source_table": source["source_table"],
                    "task": source["task"],
                    "object": source["object"],
                    "task_class": source["task_class"],
                    "object_group": source.get("object_group"),
                    "mapping_mode": source["mapping_mode"],
                    "proxy_disclosed": source["proxy_disclosed"],
                    "image_path": str(frame_path.resolve()),
                    "instruction": instruction,
                    "predicted_action": predicted_action,
                    "inference_seconds": round(time.perf_counter() - started, 6),
                    "attempt": attempt,
                    "prior_failures": failures,
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
                "selection_id": selection_id,
                "source_table": source["source_table"],
                "task": source["task"],
                "object": source["object"],
                "task_class": source["task_class"],
                "object_group": source.get("object_group"),
                "mapping_mode": source["mapping_mode"],
                "proxy_disclosed": source["proxy_disclosed"],
                "image_path": str(frame_path.resolve()),
                "instruction": instruction,
                "predicted_action": [],
                "failures": failures,
                "status": "FAIL",
                "comprehensive_diagnosis_required": True,
            }
        completed_by_id[selection_id] = result
        ordered = [
            completed_by_id[row["selection_id"]]
            for row in source_results
            if row["selection_id"] in completed_by_id
        ]
        write_manifest(manifest_path, ordered)
        passed = sum(row["status"] == "PASS" for row in ordered)
        print(
            f"[{len(ordered):02d}/60] {selection_id} {result['status']} "
            f"(pass={passed})",
            flush=True,
        )

    final = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        f"OPENVLA_POLICY_PROBE_COMPLETE "
        f"{final['pass_count']}/{final['completed_count']}",
        flush=True,
    )
    return 0 if final["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
