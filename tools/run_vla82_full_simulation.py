"""Run strict VLA82 full-simulation preparation commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.vla82_full_sim.annotations import DEFAULT_REGISTRY_PATH, compile_all, sha256_file
from tools.vla82_full_sim.annotations import OperationSpec, predicates_for_phases
from tools.vla82_full_sim.assets import (
    DEFAULT_CATALOG_PATH,
    AssetResolutionError,
    load_asset_catalog,
    materialize_fixture_source_frame,
    materialize_custom_asset,
    resolve_asset,
    write_asset_catalog,
)
from tools.vla82_full_sim.contracts import validate_fixed_scope
from tools.vla82_full_sim.environment import (
    EnvironmentValidationError,
    build_scene_request,
    fixture_binding,
    make_environment,
)
from tools.vla82_full_sim.expert import collect_expert_episode, validated_resume_episode


OPERATION_SPECS = PROJECT_ROOT / "configs" / "vla82_full_simulation" / "operation_specs.json"
BUILD_STATE = (
    PROJECT_ROOT
    / "outputs"
    / "midterm_testing_vla82"
    / "full_simulation_objectwise_8x60"
    / "build_state.json"
)
MAPPING_PLAN = PROJECT_ROOT / "outputs" / "midterm_testing_vla82" / "simulator_mapping_plan.json"
ASSET_AUDIT_ROOT = BUILD_STATE.parent / "asset_scene_audit"
DEMO_ROOT = BUILD_STATE.parent / "expert_demos"


_RUNTIME_PHASE_OVERRIDES: dict[str, tuple[str, ...]] = {
    # The source operation.json is generic, while the real-video ledger row
    # records the concrete articulated action required by the test.
    "VLA82-009": ("push",),
}


def _runtime_spec_for_execution(spec: OperationSpec) -> OperationSpec:
    """Resolve a traceable real-video operation without rewriting source data."""
    phases = _RUNTIME_PHASE_OVERRIDES.get(spec.selection_id)
    if phases is None:
        return spec
    differences = dict(spec.source_differences)
    operation = str(differences.get("ledger_operation_label", "")).strip()
    if not operation:
        raise ValueError(f"runtime phase override lacks ledger operation provenance: {spec.selection_id}")
    return replace(
        spec,
        operation_text=operation,
        phases=phases,
        predicate_names=predicates_for_phases(phases),
    )


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_compile_specs(args: argparse.Namespace) -> int:
    specs = compile_all(DEFAULT_REGISTRY_PATH)
    errors = validate_fixed_scope(specs)
    atomic_write_json(OPERATION_SPECS, [asdict(spec) for spec in specs])
    status = "PASS" if not errors else "FAIL"
    atomic_write_json(
        BUILD_STATE,
        {
            "timestamp": datetime.now().astimezone().isoformat(),
            "task": "operation_specs",
            "status": status,
            "command": "compile-specs",
            "test_command": (
                "python -m pytest tests/test_vla82_full_sim_annotations.py "
                "tests/test_vla82_full_sim_contracts.py -q"
            ),
            "output_path": str(OPERATION_SPECS.resolve()),
            "sha256": sha256_file(OPERATION_SPECS),
            "errors": errors,
        },
    )
    print(f"compiled={len(specs)} errors={len(errors)}")
    return int(bool(errors))


def _load_compiled_specs() -> list[OperationSpec]:
    records = json.loads(OPERATION_SPECS.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("compiled operation specs must be a JSON list")
    source_specs = [
        OperationSpec(
            selection_id=str(item["selection_id"]),
            task=str(item["task"]),
            object_name=str(item["object_name"]),
            operation_text=str(item["operation_text"]),
            source_kind=str(item["source_kind"]),
            source_path=str(item["source_path"]),
            phases=tuple(item["phases"]),
            manipulated_objects=tuple(item["manipulated_objects"]),
            predicate_names=tuple(item["predicate_names"]),
            source_sha256=str(item["source_sha256"]),
            source_table=str(item.get("source_table", "")),
            source_differences=tuple(tuple(row) for row in item.get("source_differences", [])),
        )
        for item in records
    ]
    return [_runtime_spec_for_execution(spec) for spec in source_specs]


def _visible_primary_object(environment: Any, request: Any) -> bool:
    """Use renderer segmentation IDs, never metadata, to prove primary visibility."""
    raw = environment.unwrapped.env
    obj = getattr(raw, "objects", {}).get("obj")
    # ObjectPlayEnv retains the object in its MuJoCo model but does not expose
    # an ``objects`` mapping. Its documented generated object prefix remains
    # a renderer-level identifier, so visibility is still segmentation-based.
    prefixes: list[str]
    if obj is not None:
        prefixes = [str(getattr(obj, "naming_prefix", "obj_"))]
    elif getattr(raw, "fixtures", None):
        # A fixture PASS is bound to a declared fixture part, never to an
        # arbitrary stove / cabinet / kitchen geom in the rendered scene.
        binding = fixture_binding(raw, request)
        prefixes = [binding["geom"]]
    else:
        prefixes = ["MCJFObj_0_"]
    target_ids = {
        index
        for index in range(int(raw.sim.model.ngeom))
        if any(str(raw.sim.model.geom_id2name(index) or "").startswith(prefix) for prefix in prefixes if prefix)
    }
    if not target_ids:
        return False
    for camera in request.camera_names:
        segmentation = raw.sim.render(width=256, height=256, camera_name=camera, segmentation=True)
        if getattr(segmentation, "ndim", 0) == 3 and any(
            bool((segmentation[:, :, 1] == geom_id).any()) for geom_id in target_ids
        ):
            return True
    return False


def _fixture_runtime_asset(asset: Any, binding: dict[str, str]) -> Any:
    """Record concrete MuJoCo fixture identifiers before an audit can claim exactness."""
    evidence = ASSET_AUDIT_ROOT / "fixture_evidence" / f"{asset.selection_id}.json"
    source = materialize_fixture_source_frame(asset)
    atomic_write_json(
        evidence,
        {
            "selection_id": asset.selection_id,
            "semantic_class": asset.semantic_class,
            "task_class": asset.asset_path_or_group,
            "fixture_binding": binding,
            "joint_verified": bool(binding.get("joint")) or str(binding.get("joint_required", "false")).lower() != "true",
            **source,
            "collision": "native MuJoCo fixture geom",
        },
    )
    return replace(asset, exact_class=True, collision_validated=True, visible_validated=False, evidence_path=str(evidence.resolve()))


def _preview_path(selection_id: str) -> Path:
    return ASSET_AUDIT_ROOT / "previews" / f"{selection_id}.png"


def _write_two_camera_preview(path: Path, environment: Any, camera_names: tuple[str, ...]) -> None:
    import imageio.v3 as iio
    import numpy as np

    raw = environment.unwrapped.env
    frames = [np.asarray(raw.sim.render(width=256, height=256, camera_name=camera), dtype=np.uint8) for camera in camera_names]
    if any(frame.ndim != 3 or frame.shape[-1] != 3 for frame in frames):
        raise ValueError("camera preview is not RGB")
    preview = np.concatenate(frames, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, preview)


def audit_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    """Count only records actually completed by the simulator."""
    return {
        "exact": sum(bool(item.get("exact")) for item in results),
        "loadable": sum(bool(item.get("loadable")) for item in results),
        "visible": sum(bool(item.get("visible")) for item in results),
        "complete_robot": sum(bool(item.get("complete_robot")) for item in results),
        "runtime_identity": sum(bool(item.get("runtime_asset_identity", {}).get("matches")) for item in results),
        "proxies": sum(bool(item.get("proxy")) for item in results),
    }


def audit_is_complete(results: list[dict[str, Any]]) -> bool:
    counts = audit_counts(results)
    expected = {f"VLA82-{index:03d}" for index in range(1, 61)}
    return len(results) == 60 and {str(item.get("selection_id")) for item in results} == expected and counts == {
        "exact": 60,
        "loadable": 60,
        "visible": 60,
        "complete_robot": 60,
        "runtime_identity": 60,
        "proxies": 0,
    }


def reusable_audit_pass(item: dict[str, Any], *, render_requested: bool) -> bool:
    """Only a fully evidenced PASS may be reused; failures are always rerun."""
    required_true = ("exact", "loadable", "visible", "complete_robot")
    if item.get("status") != "PASS" or item.get("proxy") is not False:
        return False
    if not all(item.get(key) is True for key in required_true):
        return False
    identity = item.get("runtime_asset_identity")
    if not isinstance(identity, dict) or identity.get("matches") is not True:
        return False
    if not identity.get("selection_id") or not identity.get("asset_key"):
        return False
    if identity.get("asset_kind") == "custom_same_class":
        loaded, expected = identity.get("loaded_mjcf_path"), identity.get("expected_mjcf_path")
        if not loaded or not expected or Path(str(loaded)).resolve() != Path(str(expected)).resolve():
            return False
    if not Path(str(item.get("asset_evidence", ""))).is_file():
        return False
    try:
        evidence = json.loads(Path(str(item["asset_evidence"])).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if isinstance(evidence, dict) and "fixture_binding" in evidence:
        binding = evidence.get("fixture_binding", {})
        frame = Path(str(evidence.get("source_frame", "")))
        if not frame.is_file() or not evidence.get("source_frame_sha256"):
            return False
        if hashlib.sha256(frame.read_bytes()).hexdigest() != evidence["source_frame_sha256"]:
            return False
        if not isinstance(binding, dict) or not binding.get("geom"):
            return False
        if str(binding.get("joint_required", "true")).lower() == "true" and not binding.get("joint"):
            return False
        if not evidence.get("joint_verified", False):
            return False
    camera = item.get("audit_camera")
    if not isinstance(camera, dict) or int(camera.get("segmentation_pixels", 0)) < 4:
        return False
    if int(item.get("visibility_pixels", 0)) < 4:
        return False
    return not render_requested or Path(str(item.get("preview", ""))).is_file()


def _write_audit_checkpoint(results: list[dict[str, Any]], *, render_requested: bool, status: str) -> None:
    atomic_write_json(
        ASSET_AUDIT_ROOT / "audit.json",
        {
            "timestamp": datetime.now().astimezone().isoformat(),
            "task": "asset_scene_audit",
            "status": status,
            "counts": audit_counts(results),
            "render_requested": render_requested,
            "results": results,
        },
    )


def command_audit_assets(args: argparse.Namespace) -> int:
    if not OPERATION_SPECS.is_file():
        command_compile_specs(args)
    specs = _load_compiled_specs()
    if not DEFAULT_CATALOG_PATH.is_file():
        write_asset_catalog(specs)
    catalog = load_asset_catalog(DEFAULT_CATALOG_PATH)
    mappings = {
        str(item["selection_id"]): item
        for item in json.loads(MAPPING_PLAN.read_text(encoding="utf-8"))["mappings"]
    }
    previous: dict[str, dict[str, Any]] = {}
    checkpoint_path = ASSET_AUDIT_ROOT / "audit.json"
    if checkpoint_path.is_file():
        try:
            previous = {
                str(item["selection_id"]): item
                for item in json.loads(checkpoint_path.read_text(encoding="utf-8")).get("results", [])
                if reusable_audit_pass(item, render_requested=bool(args.render))
            }
        except (OSError, ValueError, KeyError, TypeError):
            previous = {}
    results: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        if spec.selection_id in previous:
            results.append(previous[spec.selection_id])
            print(f"[{index:02d}/60] {spec.selection_id} RESUME PASS", flush=True)
            continue
        environment = None
        try:
            mapping = mappings[spec.selection_id]
            asset = resolve_asset(spec, {**mapping, "asset": catalog[spec.selection_id]})
            if asset.kind == "custom_same_class":
                asset = materialize_custom_asset(asset)
            scene = build_scene_request(spec, asset)
            environment = make_environment(scene, seed=820000 + index)
            observation, _ = environment.reset(seed=820000 + index)
            preview = _preview_path(spec.selection_id)
            if args.render:
                _write_two_camera_preview(preview, environment, scene.camera_names)
            raw = environment.unwrapped.env
            if asset.kind == "fixture_part":
                asset = _fixture_runtime_asset(asset, fixture_binding(raw, scene))
            visible = _visible_primary_object(environment, scene)
            complete_robot = bool(
                raw.robots
                and getattr(raw.robots[0], "name", None) == "PandaOmron"
                and len(raw.robots[0].robot_joints) >= 7
            )
            results.append(
                {
                    "selection_id": spec.selection_id,
                    "status": "PASS" if visible and complete_robot else "FAIL",
                    "exact": asset.exact_class,
                    "loadable": True,
                    "visible": visible,
                    "complete_robot": complete_robot,
                    "proxy": False,
                    "preview": str(preview.resolve()) if preview.is_file() else "",
                    "asset_evidence": asset.evidence_path,
                    "manipulated_objects": list(scene.manipulated_objects),
                    "fixture_requirements": list(scene.fixture_requirements),
                    "audit_camera": getattr(environment, "audit_camera", {}),
                    "visibility_pixels": int(getattr(environment, "audit_camera", {}).get("segmentation_pixels", 0)),
                    "runtime_asset_identity": getattr(environment, "runtime_asset_identity", {}),
                }
            )
        except (AssetResolutionError, EnvironmentValidationError, OSError, ValueError, KeyError) as error:
            results.append(
                {
                    "selection_id": spec.selection_id,
                    "status": "FAIL",
                    "exact": False,
                    "loadable": False,
                    "visible": False,
                    "complete_robot": False,
                    "proxy": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            if environment is not None:
                environment.close()
        _write_audit_checkpoint(results, render_requested=bool(args.render), status="RUNNING")
        print(f"[{index:02d}/60] {spec.selection_id} {results[-1]['status']}", flush=True)
    counts = audit_counts(results)
    passed = audit_is_complete(results)
    audit = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "task": "asset_scene_audit",
        "status": "PASS" if passed else "FAIL",
        "counts": counts,
        "render_requested": bool(args.render),
        "results": results,
    }
    audit_path = ASSET_AUDIT_ROOT / "audit.json"
    atomic_write_json(audit_path, audit)
    atomic_write_json(
        BUILD_STATE,
        {
            "timestamp": audit["timestamp"],
            "task": "asset_scene_audit",
            "status": audit["status"],
            "command": "audit-assets --render" if args.render else "audit-assets",
            "counts": counts,
            "audit_path": str(audit_path.resolve()),
            "asset_catalog_sha256": sha256_file(DEFAULT_CATALOG_PATH),
        },
    )
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    return 0 if passed else 1


def _scene_for_spec(spec: OperationSpec):
    """Resolve exactly the Task-2 asset/scene contract for a source spec."""
    if not DEFAULT_CATALOG_PATH.is_file():
        write_asset_catalog(_load_compiled_specs())
    catalog = load_asset_catalog(DEFAULT_CATALOG_PATH)
    mappings = {
        str(item["selection_id"]): item
        for item in json.loads(MAPPING_PLAN.read_text(encoding="utf-8"))["mappings"]
    }
    asset = resolve_asset(spec, {**mappings[spec.selection_id], "asset": catalog[spec.selection_id]})
    if asset.kind == "custom_same_class":
        asset = materialize_custom_asset(asset)
    return build_scene_request(spec, asset)


def command_collect_demos(args: argparse.Namespace) -> int:
    """Collect source-labelled training experts, never acceptance evidence."""
    if not OPERATION_SPECS.is_file():
        command_compile_specs(args)
    requested = {part.strip() for part in str(args.selection_id or "").split(",") if part.strip()}
    specs = [spec for spec in _load_compiled_specs() if not requested or spec.selection_id in requested]
    if requested - {spec.selection_id for spec in specs}:
        raise ValueError(f"unknown selection ids: {sorted(requested - {spec.selection_id for spec in specs})}")
    reports: list[dict[str, Any]] = []
    for object_index, spec in enumerate(specs):
        scene = _scene_for_spec(spec)
        successes = 0
        attempts = 0
        next_seed = 2000
        while successes < int(args.episodes_per_object) and attempts < int(args.max_attempts_per_object):
            seed = next_seed
            next_seed += 1
            attempts += 1
            candidate = DEMO_ROOT / spec.selection_id / f"episode-{seed}.npz"
            if args.resume and validated_resume_episode(candidate, spec, scene, seed):
                successes += 1
                print(f"[{object_index + 1:02d}/{len(specs):02d}] {spec.selection_id} seed={seed} RESUME PASS", flush=True)
                continue
            report = collect_expert_episode(spec, scene, seed, DEMO_ROOT)
            reports.append(report.to_dict())
            if report.status == "PASS":
                successes += 1
            print(f"[{object_index + 1:02d}/{len(specs):02d}] {spec.selection_id} seed={seed} {report.status} steps={report.steps}", flush=True)
        reports.append({"selection_id": spec.selection_id, "summary": True, "successful_episodes": successes, "attempts": attempts, "required": int(args.episodes_per_object), "status": "PASS" if successes == int(args.episodes_per_object) else "FAIL"})
    summaries = [row for row in reports if row.get("summary")]
    passed = len(summaries) == len(specs) and all(row["status"] == "PASS" for row in summaries)
    manifest = {
        "task": "expert_demos", "status": "PASS" if passed else "FAIL", "training_expert": True,
        "objects": len(specs), "required_per_object": int(args.episodes_per_object),
        "successful_episodes": sum(int(row["successful_episodes"]) for row in summaries), "reports": reports,
    }
    atomic_write_json(DEMO_ROOT / "collection_manifest.json", manifest)
    atomic_write_json(BUILD_STATE, {"timestamp": datetime.now().astimezone().isoformat(), "task": "expert_demos", "status": manifest["status"], "manifest": str((DEMO_ROOT / "collection_manifest.json").resolve()), "successful_episodes": manifest["successful_episodes"], "training_expert": True})
    print(f"objects={len(specs)} successful_episodes={manifest['successful_episodes']} status={manifest['status']}")
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compile_specs = commands.add_parser("compile-specs", help="compile the authoritative 60 specs")
    compile_specs.set_defaults(handler=command_compile_specs)
    audit_assets = commands.add_parser("audit-assets", help="validate exact assets in complete RoboCasa scenes")
    audit_assets.add_argument("--render", action="store_true", help="write one two-camera PNG per selection")
    audit_assets.set_defaults(handler=command_audit_assets)
    collect_demos = commands.add_parser("collect-demos", help="collect step-only expert training demonstrations")
    collect_demos.add_argument("--episodes-per-object", type=int, default=6)
    collect_demos.add_argument("--max-attempts-per-object", type=int, default=24)
    collect_demos.add_argument("--selection-id", default="", help="optional comma-separated fixed VLA82 IDs")
    collect_demos.add_argument("--resume", action="store_true")
    collect_demos.set_defaults(handler=command_collect_demos)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
