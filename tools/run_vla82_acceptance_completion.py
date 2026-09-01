#!/usr/bin/env python3
"""Run the disclosed VLA82 8-task/60-object hybrid acceptance workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.vla82_acceptance.assets import (  # noqa: E402
    build_exact_asset_resolutions,
    default_digital_twin_spec,
    write_digital_twin,
)
from tools.vla82_acceptance.contracts import validate_item_report  # noqa: E402
from tools.vla82_acceptance.demo_evidence import (  # noqa: E402
    load_bindings,
    load_registry,
    materialize_demo,
    materialize_registry_object_demo,
    validate_category_coverage,
)
from tools.vla82_acceptance.orchestrator import (  # noqa: E402
    ids_to_run,
    load_run_state,
    record_attempt,
)
from tools.vla82_acceptance.recognition import run_openvla_policy_response  # noqa: E402
from tools.vla82_acceptance.reporting import (  # noqa: E402
    aggregate,
    artifact_entry,
    draw_contact_sheet,
    write_manifest,
)
from tools.vla82_acceptance.runner import run_hybrid_trial, validate_visual_change  # noqa: E402
from tools.vla82_acceptance.skill_learning import (  # noqa: E402
    extract_motion_evidence,
    learn_skill_ir,
)

REGISTRY = ROOT / "outputs/midterm_testing_vla82/vla82_midterm_registry.json"
MAPPING_PLAN = ROOT / "outputs/midterm_testing_vla82/simulator_mapping_plan.json"
BINDINGS = ROOT / "configs/vla82_acceptance/human_demo_8.json"
OUTPUT = ROOT / "outputs/midterm_testing_vla82/acceptance_completion_8x60"
STATE = OUTPUT / "run_state.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_objects() -> list[dict[str, Any]]:
    return list(load_registry(REGISTRY)["selected_objects"])


def _mapping_lookup() -> dict[str, dict[str, Any]]:
    payload = _read_json(MAPPING_PLAN)
    return {str(item["selection_id"]): dict(item) for item in payload["mappings"]}


def _merged_mapping(selection: dict[str, Any]) -> dict[str, Any]:
    mapping = _mapping_lookup().get(str(selection["selection_id"]), {})
    mapping.update(selection)
    return mapping


def _instruction(item: dict[str, Any]) -> str:
    return (
        f"执行{item['task']}任务，识别并对{item['object']}执行操作："
        f"{item.get('real_video_annotation') or item.get('operation_label', '')}"
    )


def _ensure_object_demo(item: dict[str, Any], bundle: Path) -> dict[str, Any]:
    if (bundle / "source_demo.json").is_file() and (bundle / "source_demo_preview.png").is_file():
        return _read_json(bundle / "source_demo.json")
    return materialize_registry_object_demo(item, bundle)


def _ensure_skill_ir(item: dict[str, Any], bundle: Path) -> dict[str, Any]:
    path = bundle / "skill_ir.json"
    if path.is_file():
        return _read_json(path)
    demo = _read_json(bundle / "source_demo.json")
    evidence = extract_motion_evidence(
        Path(demo["video"]),
        start_seconds=float(demo["start_seconds"]),
        stop_seconds=float(demo["stop_seconds"]),
    )
    skill = learn_skill_ir(item, evidence)
    _write_json(path, skill)
    return skill


def _recognize(
    item: dict[str, Any], bundle: Path, *, host: str, port: int, resume: bool
) -> dict[str, Any]:
    destination = bundle / "recognition.json"
    if resume and destination.is_file():
        existing = _read_json(destination)
        if existing.get("success") is True:
            return existing
    record = run_openvla_policy_response(
        expected_object=str(item["object"]),
        image_path=bundle / "source_demo_preview.png",
        instruction=_instruction(item),
        host=host,
        port=port,
    )
    _write_json(destination, record)
    _write_json(bundle / "raw_openvla_action.json", record["raw_openvla_action"])
    return record


def command_audit_demos(_args: argparse.Namespace) -> int:
    errors = validate_category_coverage(load_registry(REGISTRY), load_bindings(BINDINGS))
    print(json.dumps({"task_count": 8, "errors": errors}, ensure_ascii=False))
    return 1 if errors else 0


def command_audit_assets(_args: argparse.Namespace) -> int:
    resolutions = build_exact_asset_resolutions(
        mapping_plan=MAPPING_PLAN,
        registry=REGISTRY,
        output_root=OUTPUT / "digital_twins",
    )
    payload = [item.to_dict() for item in resolutions]
    _write_json(OUTPUT / "asset_resolutions.json", payload)
    exact = sum(item["exact_class"] is True for item in payload)
    print(json.dumps({"resolved": len(payload), "exact": exact}, ensure_ascii=False))
    return 0 if len(payload) == exact == 60 else 1


def command_run_recognition(args: argparse.Namespace) -> int:
    failures = 0
    for index, item in enumerate(_selected_objects(), 1):
        selection_id = str(item["selection_id"])
        bundle = OUTPUT / "object_operations" / selection_id
        try:
            _ensure_object_demo(item, bundle)
            recognition = _recognize(
                item, bundle, host=args.host, port=args.port, resume=args.resume
            )
            if recognition.get("success") is not True:
                failures += 1
            print(f"[{index:02d}/60] {selection_id} recognition={recognition.get('success')}")
        except Exception as error:  # batch must continue and preserve the diagnosis trail
            failures += 1
            record_attempt(
                STATE,
                selection_id=f"RECOG:{selection_id}",
                status="FAIL",
                evidence_valid=False,
                root_cause=f"{type(error).__name__}:{error}",
                details={"stage": "recognition"},
            )
            print(f"[{index:02d}/60] {selection_id} ERROR {type(error).__name__}: {error}")
    print(json.dumps({"recognition_failures": failures}, ensure_ascii=False))
    return 1 if failures else 0


def _run_one(
    item: dict[str, Any],
    bundle: Path,
    recognition: dict[str, Any],
    *,
    frame_count: int,
    bundle_kind: str,
) -> dict[str, Any]:
    mapping = _merged_mapping(item)
    twin = write_digital_twin(
        default_digital_twin_spec(mapping), OUTPUT / "execution_digital_twins"
    )
    report = run_hybrid_trial(
        mapping=mapping,
        skill_ir=_read_json(bundle / "skill_ir.json"),
        digital_twin=twin,
        raw_openvla_action=recognition["raw_openvla_action"],
        recognition=recognition,
        output_dir=bundle,
        frame_count=frame_count,
        render_size=192,
    )
    report["bundle_kind"] = bundle_kind
    report["evidence_boundary"] = (
        "OpenVLA supplies a target-conditioned visual action response; a disclosed "
        "deterministic safety supervisor executes the MuJoCo reproduction."
    )
    _write_json(bundle / "report.json", report)
    return report


def command_run_objects(args: argparse.Namespace) -> int:
    items = _selected_objects()
    state = load_run_state(STATE)
    pending = set(ids_to_run([str(item["selection_id"]) for item in items], state)) if args.resume else {
        str(item["selection_id"]) for item in items
    }
    failures = 0
    for index, item in enumerate(items, 1):
        selection_id = str(item["selection_id"])
        bundle = OUTPUT / "object_operations" / selection_id
        if selection_id not in pending:
            print(f"[{index:02d}/60] {selection_id} SKIP verified PASS")
            continue
        try:
            _ensure_object_demo(item, bundle)
            _ensure_skill_ir(item, bundle)
            recognition = _recognize(
                item, bundle, host=args.host, port=args.port, resume=True
            )
            report = _run_one(
                item,
                bundle,
                recognition,
                frame_count=args.frames,
                bundle_kind="object_operation",
            )
            passed = report["acceptance_status"] == "PASS"
            failures += 0 if passed else 1
            state = record_attempt(
                STATE,
                selection_id=selection_id,
                status="PASS" if passed else "FAIL",
                evidence_valid=passed,
                root_cause=None if passed else ",".join(report.get("acceptance_errors", [])),
                details={"stage": "object_operation", "bundle": str(bundle.resolve())},
            )
            print(f"[{index:02d}/60] {selection_id} {report['acceptance_status']}")
        except Exception as error:
            failures += 1
            state = record_attempt(
                STATE,
                selection_id=selection_id,
                status="FAIL",
                evidence_valid=False,
                root_cause=f"{type(error).__name__}:{error}",
                details={"stage": "object_operation", "bundle": str(bundle.resolve())},
            )
            action = state["items"][selection_id]["retry_decision"]["action"]
            print(f"[{index:02d}/60] {selection_id} ERROR {error}; next={action}")
    print(json.dumps({"object_failures": failures}, ensure_ascii=False))
    return 1 if failures else 0


def _task_item(binding: dict[str, Any]) -> dict[str, Any]:
    representative = next(
        item for item in _selected_objects() if item["selection_id"] == binding["selection_id"]
    )
    result = dict(representative)
    result["task"] = binding["task"]
    result["operation_label"] = binding["semantic_operation"]
    return result


def command_run_tasks(args: argparse.Namespace) -> int:
    failures = 0
    for index, binding in enumerate(load_bindings(BINDINGS), 1):
        item = _task_item(binding)
        selection_id = str(item["selection_id"])
        state_id = f"TASK:{selection_id}"
        bundle = OUTPUT / "task_learning" / selection_id
        if args.resume:
            state = load_run_state(STATE)
            if state_id not in ids_to_run([state_id], state):
                print(f"[{index}/8] {state_id} SKIP verified PASS")
                continue
        try:
            demo = materialize_demo(binding, bundle)
            evidence = extract_motion_evidence(
                Path(demo["video"]),
                start_seconds=float(demo["start_seconds"]),
                stop_seconds=float(demo["stop_seconds"]),
            )
            _write_json(bundle / "skill_ir.json", learn_skill_ir(item, evidence))
            recognition = _recognize(
                item, bundle, host=args.host, port=args.port, resume=args.resume
            )
            report = _run_one(
                item,
                bundle,
                recognition,
                frame_count=args.frames,
                bundle_kind="task_learning",
            )
            passed = report["acceptance_status"] == "PASS"
            failures += 0 if passed else 1
            record_attempt(
                STATE,
                selection_id=state_id,
                status="PASS" if passed else "FAIL",
                evidence_valid=passed,
                root_cause=None if passed else ",".join(report.get("acceptance_errors", [])),
                details={"stage": "task_learning", "bundle": str(bundle.resolve())},
            )
            print(f"[{index}/8] {state_id} {report['acceptance_status']}")
        except Exception as error:
            failures += 1
            state = record_attempt(
                STATE,
                selection_id=state_id,
                status="FAIL",
                evidence_valid=False,
                root_cause=f"{type(error).__name__}:{error}",
                details={"stage": "task_learning", "bundle": str(bundle.resolve())},
            )
            action = state["items"][state_id]["retry_decision"]["action"]
            print(f"[{index}/8] {state_id} ERROR {error}; next={action}")
    print(json.dumps({"task_failures": failures}, ensure_ascii=False))
    return 1 if failures else 0


def _load_verified_reports(parent: Path, expected: int) -> tuple[list[dict[str, Any]], list[str]]:
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    if not parent.is_dir():
        return reports, [f"missing_directory:{parent}"]
    for bundle in sorted(path for path in parent.iterdir() if path.is_dir()):
        report_path = bundle / "report.json"
        if not report_path.is_file():
            continue
        report = _read_json(report_path)
        item_errors = validate_item_report(report, bundle) + validate_visual_change(bundle)
        capture = cv2.VideoCapture(str(bundle / "rollout.mp4"))
        try:
            if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) < 1:
                item_errors.append("rollout_video_decode_failed")
        finally:
            capture.release()
        if item_errors:
            errors.extend(f"{bundle.name}:{error}" for error in item_errors)
        report["bundle_dir"] = str(bundle.resolve())
        report["verification_errors"] = item_errors
        report["acceptance_status"] = "PASS" if not item_errors else "FAIL"
        reports.append(report)
    if len(reports) != expected:
        errors.append(f"report_count:{parent.name}:{len(reports)}:{expected}")
    return reports, errors


def command_verify(_args: argparse.Namespace) -> int:
    task_reports, task_errors = _load_verified_reports(OUTPUT / "task_learning", 8)
    object_reports, object_errors = _load_verified_reports(OUTPUT / "object_operations", 60)
    manifest = aggregate(task_reports, object_reports)
    manifest["verification_errors"] = task_errors + object_errors
    manifest["evidence_interpretation"] = {
        "pure_autonomous_openvla": False,
        "recognition": (
            "No recognition failure means each exact selected-object source frame and "
            "object-conditioned instruction yielded a finite OpenVLA action; no top-1 "
            "classification accuracy is claimed."
        ),
        "reproduction": "Guarded hybrid MuJoCo reproduction with disclosed supervision.",
    }
    artifacts: list[dict[str, Any]] = []
    for report in task_reports + object_reports:
        bundle = Path(report["bundle_dir"])
        for name in ("source_demo_preview.png", "first_frame.png", "last_frame.png", "rollout.mp4"):
            if (bundle / name).is_file():
                artifacts.append(artifact_entry(bundle / name))
    manifest["evidence_artifacts"] = artifacts
    if task_errors or object_errors:
        manifest["overall_status"] = "FAIL"
    write_manifest(OUTPUT / "manifest.json", manifest)
    if task_reports:
        draw_contact_sheet(task_reports, OUTPUT / "evidence_contact_sheet_tasks.png")
    if object_reports:
        draw_contact_sheet(object_reports, OUTPUT / "evidence_contact_sheet_objects.png")
    matrix = {
        "summary": {key: manifest[key] for key in (
            "task_learning_expected", "task_learning_completed_count", "task_learning_pass_count",
            "object_operation_expected", "object_operation_completed_count", "object_operation_pass_count",
            "recognition_expected", "recognition_pass_count", "overall_status"
        )},
        "tasks": task_reports,
        "objects": object_reports,
        "verification_errors": manifest["verification_errors"],
    }
    _write_json(OUTPUT / "acceptance_matrix_data.json", matrix)
    print(json.dumps(matrix["summary"] | {"verification_error_count": len(manifest["verification_errors"])}, ensure_ascii=False))
    return 0 if manifest["overall_status"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = {
        "audit-demos": command_audit_demos,
        "audit-assets": command_audit_assets,
        "run-recognition": command_run_recognition,
        "run-objects": command_run_objects,
        "run-tasks": command_run_tasks,
        "verify": command_verify,
    }
    for name, function in commands.items():
        command = subparsers.add_parser(name)
        command.set_defaults(function=function)
        if name in {"run-recognition", "run-objects", "run-tasks"}:
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8765)
            command.add_argument("--resume", action="store_true")
        if name in {"run-objects", "run-tasks"}:
            command.add_argument("--frames", type=int, default=48)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
