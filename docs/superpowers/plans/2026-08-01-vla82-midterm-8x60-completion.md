# VLA82 Midterm 8×60 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce auditable simulation evidence that eight household task classes complete public-video demonstration parsing, skill learning, and robot reproduction, while all sixty fixed VLA82 objects complete their assigned manipulation and recognition checks with per-item video evidence.

**Architecture:** A strict acceptance package reads the existing VLA82 registry as the only selection source, materializes public demonstration evidence, creates validated Skill IR, resolves native or exact digital-twin simulator assets, runs a guarded OpenVLA/RoboCasa rollout, and validates every evidence bundle before aggregation. The aggregate manifest is derived only from per-item reports and never promotes smoke tests, isolated inference, or official replay to full operation success.

**Tech Stack:** Python 3.11, unittest/pytest, OpenCV/imageio/Pillow, OpenVLA TCP service, RoboCasa/Gymnasium/MuJoCo, pandas/pyarrow, openpyxl, JSON/MP4/PNG evidence.

## Global Constraints

- Human demonstrations may use traceable public real-human videos; no on-site camera capture is required.
- Simulation only; no real robot arm is introduced.
- The sixty objects must be exactly the sixty `selected_objects` in `outputs/midterm_testing_vla82/vla82_midterm_registry.json`.
- Inputs from `C:\RobotProject` may be used only as public-video evidence; they may not replace the VLA82 task/object selection.
- A functional proxy is not an exact-object pass. Missing native assets require an explicitly labeled same-class digital twin.
- Hybrid execution must disclose OpenVLA output, executed action, state-supervisor intervention, and scripted/expert action use.
- Three repeated failures trigger cross-layer diagnosis and continued repair, not automatic termination.
- A final PASS requires 8/8 task-learning bundles, 60/60 operation bundles, 60/60 recognition passes, and zero missing evidence.
- The current workspace has no `.git` directory. Do not invent commit results; omit commit commands and retain an execution log instead.

## File Structure

- Create `tools/vla82_acceptance/__init__.py`: public package exports.
- Create `tools/vla82_acceptance/contracts.py`: strict item and aggregate evidence contracts.
- Create `tools/vla82_acceptance/demo_evidence.py`: public-video hashing, decoding, preview extraction, and source classification.
- Create `tools/vla82_acceptance/skill_learning.py`: visual evidence to Skill IR conversion and validation.
- Create `tools/vla82_acceptance/assets.py`: native-asset audit and exact digital-twin specifications.
- Create `tools/vla82_acceptance/runner.py`: guarded hybrid rollout and evidence recording.
- Create `tools/vla82_acceptance/orchestrator.py`: resumable 8×60 execution, retry accounting, and diagnostics.
- Create `tools/vla82_acceptance/reporting.py`: manifest, workbook, contact sheet, and final gate.
- Create `tools/run_vla82_acceptance_completion.py`: command-line entry point.
- Create `configs/vla82_acceptance/human_demo_8.json`: eight category-to-public-human-video bindings.
- Create `configs/vla82_acceptance/operation_success_rules.json`: explicit success signals for grasp, clean, and device-control families.
- Create `tests/test_vla82_acceptance_contracts.py`.
- Create `tests/test_vla82_demo_evidence.py`.
- Create `tests/test_vla82_skill_learning.py`.
- Create `tests/test_vla82_acceptance_assets.py`.
- Create `tests/test_vla82_acceptance_runner.py`.
- Create `tests/test_vla82_acceptance_orchestrator.py`.
- Create `tests/test_vla82_acceptance_reporting.py`.
- Write runtime evidence beneath `outputs/midterm_testing_vla82/acceptance_completion_8x60/`.

---

### Task 1: Strict Evidence Contracts and Final Gate

**Files:**
- Create: `tools/vla82_acceptance/__init__.py`
- Create: `tools/vla82_acceptance/contracts.py`
- Create: `tests/test_vla82_acceptance_contracts.py`

**Interfaces:**
- Produces: `validate_item_report(report: Mapping[str, Any], root: Path) -> list[str]`
- Produces: `build_acceptance_summary(task_reports, object_reports) -> dict[str, Any]`
- Consumes later: every runner and reporting component uses these two functions.

- [ ] **Step 1: Write failing contract tests**

```python
from pathlib import Path
from tools.vla82_acceptance.contracts import build_acceptance_summary, validate_item_report


def test_smoke_or_replay_cannot_count_as_complete(tmp_path: Path):
    report = {
        "selection_id": "VLA82-001",
        "status": "PASS",
        "validation_kind": "interface_smoke",
        "recognition": {"success": True},
        "operation": {"success": True},
        "evidence": {},
    }
    errors = validate_item_report(report, tmp_path)
    assert "unsupported_validation_kind" in errors


def test_final_gate_requires_exact_counts():
    summary = build_acceptance_summary([], [])
    assert summary["overall_status"] == "FAIL"
    assert summary["task_learning_pass_count"] == 0
    assert summary["object_operation_pass_count"] == 0
    assert summary["recognition_pass_count"] == 0
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_vla82_acceptance_contracts.py -q
```

Expected: collection fails because `tools.vla82_acceptance.contracts` does not exist.

- [ ] **Step 3: Implement the strict contract**

`contracts.py` must define:

```python
REQUIRED_EVIDENCE = (
    "rollout.mp4",
    "first_frame.png",
    "last_frame.png",
    "recognition.json",
    "skill_ir.json",
    "report.json",
)
ALLOWED_VALIDATION_KINDS = {"hybrid_visual_imitation_simulation"}


def validate_item_report(report, root):
    errors = []
    if report.get("validation_kind") not in ALLOWED_VALIDATION_KINDS:
        errors.append("unsupported_validation_kind")
    if report.get("recognition", {}).get("success") is not True:
        errors.append("recognition_failed")
    if report.get("operation", {}).get("success") is not True:
        errors.append("operation_failed")
    if report.get("asset", {}).get("exact_class") is not True:
        errors.append("inexact_asset")
    for name in REQUIRED_EVIDENCE:
        if not (root / name).is_file():
            errors.append(f"missing_evidence:{name}")
    return errors


def build_acceptance_summary(task_reports, object_reports):
    task_pass = sum(r.get("acceptance_status") == "PASS" for r in task_reports)
    object_pass = sum(r.get("acceptance_status") == "PASS" for r in object_reports)
    recognition_pass = sum(
        r.get("recognition", {}).get("success") is True for r in object_reports
    )
    overall = (
        len(task_reports) == 8
        and task_pass == 8
        and len(object_reports) == 60
        and object_pass == 60
        and recognition_pass == 60
    )
    return {
        "task_learning_expected": 8,
        "task_learning_pass_count": task_pass,
        "object_operation_expected": 60,
        "object_operation_pass_count": object_pass,
        "recognition_expected": 60,
        "recognition_pass_count": recognition_pass,
        "overall_status": "PASS" if overall else "FAIL",
    }
```

`__init__.py` exports both functions.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same pytest command. Expected: all tests pass.

---

### Task 2: Public Human Demonstration Registry and Materialization

**Files:**
- Create: `configs/vla82_acceptance/human_demo_8.json`
- Create: `tools/vla82_acceptance/demo_evidence.py`
- Create: `tests/test_vla82_demo_evidence.py`

**Interfaces:**
- Consumes: `selected_tasks` and `selected_objects` from the VLA82 registry.
- Produces: `materialize_demo(binding: Mapping, output_dir: Path) -> dict`.
- Produces: eight `source_demo.json` and `source_demo_preview.png` files.

- [ ] **Step 1: Write failing source-validation tests**

```python
def test_demo_binding_rejects_robot_only_source(tmp_path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"not-used-by-this-unit-test")
    binding = {
        "task": "室内卫生清洁",
        "source_kind": "robot_demonstration",
        "video": str(video),
    }
    errors = validate_demo_binding(binding, require_human=True, decode=False)
    assert "source_is_not_human_demonstration" in errors


def test_eight_bindings_cover_eight_registry_tables():
    registry = load_registry(REGISTRY)
    bindings = load_bindings(BINDINGS)
    assert validate_category_coverage(registry, bindings) == []
```

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_vla82_demo_evidence.py -q
```

Expected: missing module/functions.

- [ ] **Step 3: Build the binding file from verified public-human clips**

The JSON file contains exactly eight records with this complete schema:

```json
{
  "schema_version": "vla82_human_demo_bindings_v1",
  "bindings": [
    {
      "source_table": "5-8-1",
      "task": "室内卫生清洁",
      "selection_id": "VLA82-002",
      "source_kind": "public_human_demonstration",
      "dataset": "EPIC-KITCHENS",
      "clip_id": "epic_P03_109_384",
      "video": "C:\\RobotProject\\RobotProject\\datasets\\midterm_15task_delivery\\organizing__整理任务\\objects\\storage_box\\videos\\epic_kitchens_100\\epic_kitchens_100__P03_109_384.mp4",
      "start_seconds": 0.0,
      "stop_seconds": 5.0,
      "semantic_operation": "pick or handle a cleaning object"
    }
  ]
}
```

The shown record is the already-materialized organizing/storage-box human clip. The implementation discovers the remaining category bindings from local public-human-video manifests, scores semantic compatibility against the fixed eight task names and operations, and writes only unique, decodable matches. Do not mark BEHAVIOR-1K robot observations as human demonstrations. The coverage validator must reject missing tables, duplicate tables, undecodable video, invalid time ranges, and non-human source kinds.

- [ ] **Step 4: Implement materialization**

`demo_evidence.py` must:

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_demo_binding(binding, *, require_human=True, decode=True):
    errors = []
    if require_human and binding.get("source_kind") != "public_human_demonstration":
        errors.append("source_is_not_human_demonstration")
    path = Path(str(binding.get("video", "")))
    if not path.is_file():
        errors.append("video_missing")
    elif decode and probe_video(path)["frame_count"] < 1:
        errors.append("video_decode_failed")
    if float(binding.get("stop_seconds", 0)) <= float(binding.get("start_seconds", 0)):
        errors.append("invalid_time_range")
    return errors
```

Also implement `probe_video`, `extract_preview`, `validate_category_coverage`, and `materialize_demo` with the atomic-write and metadata requirements stated above; their concrete behavior is exercised by the tests in Steps 1 and 5.

`materialize_demo` writes atomically and includes dataset, clip ID, absolute path, segment bounds, SHA-256, decoder, width, height, FPS, frame count, and preview path.

- [ ] **Step 5: Run unit tests and an eight-binding audit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_vla82_demo_evidence.py -q
.\.venv\Scripts\python.exe tools\run_vla82_acceptance_completion.py audit-demos --bindings configs\vla82_acceptance\human_demo_8.json
```

Expected: tests pass; audit reports `category_count=8`, `human_source_count=8`, `decode_pass_count=8`.

---

### Task 3: Visual Parsing and Skill IR Learning

**Files:**
- Create: `tools/vla82_acceptance/skill_learning.py`
- Create: `tests/test_vla82_skill_learning.py`
- Reuse: `tools/skill_transfer/contracts.py`
- Reuse: `tools/skill_transfer/skill_ir.py`

**Interfaces:**
- Consumes: materialized demo, object/hand/contact tracks, task operation label.
- Produces: `learn_skill_ir(task: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict` and `recognize_object(frame: Path, expected: str, model: Any) -> dict`.

- [ ] **Step 1: Write failing phase and recognition tests**

```python
def test_skill_ir_requires_visual_evidence_for_each_phase():
    evidence = {"object_track": [], "contact_events": []}
    with pytest.raises(ValueError, match="visual evidence"):
        learn_skill_ir(TASK, evidence)


def test_recognition_failure_is_not_promoted_to_pass():
    result = classify_recognition(expected="sponge", scores={"cup": 0.92})
    assert result == {
        "expected": "sponge",
        "predicted": "cup",
        "confidence": 0.92,
        "success": False,
    }
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_vla82_skill_learning.py -q
```

Expected: missing functions.

- [ ] **Step 3: Implement visual-evidence extraction and Skill IR**

The implementation must create only evidence-supported phases:

```python
PHASE_ORDER = ("locate", "approach", "contact_or_grasp", "operate", "complete")


def learn_skill_ir(task, evidence):
    phases = infer_phases_from_tracks(evidence)
    if not phases or any(not p.get("supporting_frames") for p in phases):
        raise ValueError("each learned phase requires visual evidence")
    names = [p["name"] for p in phases]
    if names != sorted(names, key=PHASE_ORDER.index):
        raise ValueError("learned phase order is invalid")
    return {
        "schema": "vla82_hybrid_skill_ir_v1",
        "task": task["task"],
        "operation_label": task["operation_label"],
        "phases": phases,
        "learning_source": "public_human_video_visual_tracks",
    }
```

Recognition records must preserve expected class, predicted class, confidence, frame path, model identifier, and pass/fail. Text annotations may define the expected label but cannot overwrite the predicted label.

- [ ] **Step 4: Verify GREEN and run one real-video sample**

Run unit tests, then materialize and parse one selected clip. Expected: non-empty phases, non-empty supporting frame lists, valid `skill_ir.json`, and explicit recognition pass/fail.

---

### Task 4: Exact Asset Audit and Digital Twins

**Files:**
- Create: `configs/vla82_acceptance/operation_success_rules.json`
- Create: `tools/vla82_acceptance/assets.py`
- Create: `tests/test_vla82_acceptance_assets.py`

**Interfaces:**
- Consumes: `simulator_mapping_plan.json` and RoboCasa asset registries.
- Produces: `resolve_asset(mapping) -> AssetResolution`.
- Produces: digital-twin MJCF files beneath `outputs/midterm_testing_vla82/acceptance_completion_8x60/digital_twins/{selection_id}/model.xml`.

- [ ] **Step 1: Write failing exactness tests**

```python
def test_functional_proxy_is_not_exact():
    resolution = resolve_asset({
        "selection_id": "VLA82-001",
        "object": "垃圾桶",
        "mapping_mode": "functional_proxy",
        "object_group": "can",
    })
    assert resolution.exact_class is False


def test_digital_twin_uses_target_identity(tmp_path):
    spec = DigitalTwinSpec(
        selection_id="VLA82-001",
        target_class="垃圾桶",
        shape="cylinder",
        size=(0.24, 0.24, 0.35),
        affordances=("container", "drop_target"),
    )
    path = write_digital_twin(spec, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert 'name="VLA82-001_垃圾桶"' in text
    assert validate_mjcf(path) == []
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tests\test_vla82_acceptance_assets.py -q
```

- [ ] **Step 3: Implement asset resolution**

Native object and furniture mappings are exact only when registry metadata matches the target class or an approved synonym table. Existing `functional_proxy` and `native_object_proxy_task` mappings start as inexact. The digital-twin builder creates collision and visual geometry using explicit dimensions and affordances, then loads the MJCF through MuJoCo before setting `exact_class=true`.

The operation rules file must enumerate the three families:

```json
{
  "grasp": ["ever_grasped", "object_displacement", "target_region"],
  "clean": ["tool_contact", "surface_coverage", "completion_state"],
  "device_control": ["control_contact", "state_transition", "target_state"]
}
```

- [ ] **Step 4: Audit all sixty mappings**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_acceptance_completion.py audit-assets --build-digital-twins
```

Expected: exactly 60 resolutions, zero `functional_proxy_counted_as_exact`, and every unresolved item listed with a concrete missing capability.

---

### Task 5: Guarded Hybrid Rollout and Per-Item Evidence

**Files:**
- Create: `tools/vla82_acceptance/runner.py`
- Create: `tests/test_vla82_acceptance_runner.py`
- Reuse: `tools/openvla_tcp_client.py`
- Reuse: `tools/vla82_pure_closed_loop.py`
- Reuse: `tools/probe_vla82_public_action_replay.py`

**Interfaces:**
- Consumes: mapping, Skill IR, asset resolution, OpenVLA service.
- Produces: `run_hybrid_trial(mapping: Mapping[str, Any], skill_ir: Mapping[str, Any], asset: AssetResolution, output_dir: Path, openvla_client: Any) -> dict` with MP4/PNG/JSON evidence.

- [ ] **Step 1: Write failing action-provenance and video tests**

```python
def test_report_discloses_supervisor_and_action_sources():
    report = build_trial_report(
        raw_actions=[[0.1] * 7],
        executed_actions=[[0.0] * 7],
        interventions=1,
        success=True,
    )
    assert report["pure_autonomous_vla"] is False
    assert report["supervisor_intervention_count"] == 1
    assert report["trajectory"][0]["raw_openvla_action"] == [0.1] * 7
    assert report["trajectory"][0]["executed_action"] == [0.0] * 7


def test_identical_first_and_last_frames_fail_evidence_gate(tmp_path):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    write_png(tmp_path / "first_frame.png", image)
    write_png(tmp_path / "last_frame.png", image)
    assert "unchanged_first_last_frame" in validate_visual_change(tmp_path)
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tests\test_vla82_acceptance_runner.py -q
```

- [ ] **Step 3: Implement one guarded trial**

`run_hybrid_trial` must:

1. configure the exact native/digital-twin asset;
2. reset with a recorded deterministic seed;
3. save the first RGB observation;
4. send each visual frame and Skill IR phase prompt to OpenVLA;
5. record raw 7D OpenVLA action;
6. apply bounded action conversion and state-supervisor recovery;
7. execute a non-empty trajectory;
8. evaluate the operation-family success rule;
9. save MP4, last frame, recognition result, Skill IR, and report atomically;
10. set `acceptance_status=PASS` only after `validate_item_report` returns no errors.

- [ ] **Step 4: Run a single native-object smoke trial**

Use `VLA82-002` sponge because it has a native object mapping. Expected outputs are the eight required evidence files and an honest PASS/FAIL based on the simulator predicate.

- [ ] **Step 5: Run a single digital-twin smoke trial**

Use `VLA82-001` trash-bin target. Expected: report identifies `asset.kind=digital_twin`, `asset.exact_class=true`, and never names the can proxy as the tested trash bin.

---

### Task 6: Resumable 8×60 Orchestration and Three-Failure Diagnosis

**Files:**
- Create: `tools/vla82_acceptance/orchestrator.py`
- Create: `tools/run_vla82_acceptance_completion.py`
- Create: `tests/test_vla82_acceptance_orchestrator.py`

**Interfaces:**
- Produces CLI commands: `audit-demos`, `audit-assets`, `run-one`, `run-tasks`, `run-objects`, `resume`, `verify`.
- Produces `run_state.json` and `failure_diagnostics.json`.

- [ ] **Step 1: Write failing resume and diagnosis tests**

```python
def test_resume_skips_only_verified_pass(tmp_path):
    state = {"VLA82-001": {"status": "PASS", "evidence_valid": True}}
    assert ids_to_run(["VLA82-001", "VLA82-002"], state) == ["VLA82-002"]


def test_third_same_root_cause_requests_full_diagnosis():
    attempts = [
        {"root_cause": "asset_load"},
        {"root_cause": "asset_load"},
        {"root_cause": "asset_load"},
    ]
    assert retry_decision(attempts)["action"] == "FULL_DIAGNOSIS_CONTINUE"
```

- [ ] **Step 2: Verify RED**

Run `.\.venv\Scripts\python.exe -m pytest tests\test_vla82_acceptance_orchestrator.py -q`.

- [ ] **Step 3: Implement checkpointed execution**

Each attempt appends a timestamped record containing selection ID, stage, root-cause category, exception, seed, logs, partial evidence, and next action. The third repeated root cause runs diagnostics across source video, recognition, Skill IR, asset, OpenVLA service, action adapter, simulator state, and success rule, then queues a repaired rerun.

- [ ] **Step 4: Verify resume behavior**

Run a two-item batch, interrupt between items, then invoke `resume`. Expected: the verified item is not rerun and the unfinished item continues.

---

### Task 7: Workbook, Contact Sheet, and Aggregate Acceptance

**Files:**
- Create: `tools/vla82_acceptance/reporting.py`
- Create: `tests/test_vla82_acceptance_reporting.py`

**Interfaces:**
- Consumes only verified per-item `report.json` files.
- Produces `manifest.json`, `acceptance_matrix.xlsx`, `evidence_contact_sheet.png`, and `failure_diagnostics.json`.

- [ ] **Step 1: Write failing aggregation tests**

```python
def test_aggregate_rejects_59_object_reports(valid_task_reports, valid_object_reports):
    manifest = aggregate(valid_task_reports, valid_object_reports[:59])
    assert manifest["overall_status"] == "FAIL"
    assert manifest["object_operation_pass_count"] == 59


def test_report_paths_are_real_and_hashes_match(valid_bundle):
    manifest = aggregate(valid_bundle.tasks, valid_bundle.objects)
    for item in manifest["results"]:
        for evidence in item["evidence"].values():
            assert Path(evidence["path"]).is_file()
            assert sha256_file(Path(evidence["path"])) == evidence["sha256"]
```

- [ ] **Step 2: Verify RED**

Run `.\.venv\Scripts\python.exe -m pytest tests\test_vla82_acceptance_reporting.py -q`.

- [ ] **Step 3: Implement derived reporting**

The workbook has one row per selection and columns for category, object, operation family, human demo, Skill IR, asset kind, recognition, operation predicate, video, frames, retries, supervisor interventions, and final status. The contact sheet uses actual first/last frames and prints PASS/FAIL per item. Counts are computed from reports; no manual override field exists.

- [ ] **Step 4: Verify generated artifacts**

Open the workbook with openpyxl, render the contact sheet, and decode every MP4. Expected: row count 60 for object sheet, row count 8 for task sheet, no broken evidence link, and no duplicate selection ID.

---

### Task 8: Execute the Full Supplementary Test

**Files:**
- Write: `outputs/midterm_testing_vla82/acceptance_completion_8x60/**`

**Interfaces:**
- Consumes all components from Tasks 1–7.
- Produces the final auditable acceptance bundle.

- [ ] **Step 1: Run all fast tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_vla82_acceptance_contracts.py tests\test_vla82_demo_evidence.py tests\test_vla82_skill_learning.py tests\test_vla82_acceptance_orchestrator.py tests\test_vla82_acceptance_reporting.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run RoboCasa integration tests**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tests\test_vla82_acceptance_assets.py tests\test_vla82_acceptance_runner.py -q
```

Expected: zero failures.

- [ ] **Step 3: Audit demos and assets**

```powershell
.\.venv\Scripts\python.exe tools\run_vla82_acceptance_completion.py audit-demos
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_acceptance_completion.py audit-assets --build-digital-twins
```

Expected: eight human demonstration bindings and sixty exact asset resolutions.

- [ ] **Step 4: Start the OpenVLA service and check health**

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe tools\openvla_vla82_tcp_server.py --model-dir models\openvla-7b-oft-combined-rtx3090-merged --host 127.0.0.1 --port 8765
```

In a second process, run a health/prediction request. Expected: finite 7D action and matching model fingerprint.

- [ ] **Step 5: Run eight task-learning trials**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_acceptance_completion.py run-tasks --resume --host 127.0.0.1 --port 8765
```

Expected: eight completed bundles. Failures remain explicit and are repaired through the diagnostic loop.

- [ ] **Step 6: Run sixty object trials**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_acceptance_completion.py run-objects --resume --host 127.0.0.1 --port 8765
```

Expected: sixty completed bundles; actual PASS count is determined by evidence validation.

- [ ] **Step 7: Run final fresh verification**

```powershell
.\.venv\Scripts\python.exe tools\run_vla82_acceptance_completion.py verify --decode-all-videos --rehash-all-evidence
```

The command must print and write:

```text
task_learning: 8/8
object_operation: 60/60
recognition: 60/60
missing_evidence: 0
overall_status: PASS
```

If any value differs, report the actual result and continue diagnosis; do not claim complete acceptance.

## Plan Self-Review

- Every approved requirement maps to Tasks 2–8.
- The final gate cannot be satisfied by existing smoke, isolated inference, or official replay evidence.
- Human-source classification explicitly rejects robot-only demonstration videos.
- Functional proxies are excluded until replaced by same-class digital twins.
- All new functions have a test-first RED/GREEN step.
- Runtime counts and evidence hashes are derived, not manually entered.
- No placeholder implementation or ambiguous pass override remains.
