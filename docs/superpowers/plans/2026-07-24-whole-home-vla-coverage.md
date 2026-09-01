# Whole-Home VLA Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a truthful, whole-home 15-task/120-object VLA coverage ledger and workbook in which every counted row has a verified demonstration or simulator-generation path.

**Architecture:** A Python survey layer normalizes local RoboCasa and Bridge evidence plus registered external-dataset metadata into dataset-agnostic task, object, and evidence records. A deterministic selection layer chooses non-overlapping household-service categories across room zones and emits JSON/CSV. The existing artifact-tool workbook renderer consumes only these normalized outputs.

**Tech Stack:** Python 3.11 standard library and pytest; existing `tools.vla_metadata_survey` package; Node.js with `@oai/artifact-tool` for XLSX rendering.

## Global Constraints

- A row counts toward the final 15x120 claim only when `trainability` is `ready` or `generatable`.
- `catalog_only` entries remain visible for audit but are excluded from headline coverage.
- Each selected row records `dataset_id`, `room_zone`, `task_id`, `object_id`, and a concrete evidence source.
- Task categories must be semantically non-overlapping at household-service level; `arranging_buffet` is excluded.
- Do not overwrite source datasets or existing OpenVLA adapters.

---

### Task 1: Add normalized whole-home coverage contracts

**Files:**
- Create: `tools/vla_household_coverage.py`
- Create: `tests/test_vla_household_coverage.py`

**Interfaces:**
- Produces `EvidenceStatus = Literal["ready", "generatable", "catalog_only"]`.
- Produces `validate_coverage_row(row: dict) -> None`.
- Produces `eligible_rows(rows: list[dict]) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

```python
from tools.vla_household_coverage import eligible_rows, validate_coverage_row

def test_catalog_only_row_is_not_trainable():
    row = {"dataset_id": "robocasa", "room_zone": "kitchen", "task_id": "t",
           "object_id": "cup", "trainability": "catalog_only", "evidence": "registry"}
    validate_coverage_row(row)
    assert eligible_rows([row]) == []

def test_eligible_row_requires_a_concrete_route():
    row = {"dataset_id": "robocasa", "room_zone": "kitchen", "task_id": "t",
           "object_id": "cup", "trainability": "generatable", "evidence": "task_code"}
    validate_coverage_row(row)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tests/test_vla_household_coverage.py -v`

Expected: FAIL because `tools.vla_household_coverage` does not exist.

- [ ] **Step 3: Implement the contract**

```python
ELIGIBLE = {"ready", "generatable"}
REQUIRED = {"dataset_id", "room_zone", "task_id", "object_id", "trainability", "evidence"}

def validate_coverage_row(row: dict) -> None:
    missing = REQUIRED - row.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if row["trainability"] not in ELIGIBLE | {"catalog_only"}:
        raise ValueError("invalid trainability")
    if row["trainability"] in ELIGIBLE and not row["evidence"]:
        raise ValueError("eligible row requires evidence")

def eligible_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["trainability"] in ELIGIBLE]
```

- [ ] **Step 4: Run the focused tests**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tests/test_vla_household_coverage.py -v`

Expected: PASS.

### Task 2: Build source manifests and a room-aware evidence ledger

**Files:**
- Modify: `tools/vla_household_coverage.py`
- Modify: `tests/test_vla_household_coverage.py`
- Create: `outputs/vla_metadata_survey/whole_home_source_manifest.json`
- Create: `outputs/vla_metadata_survey/whole_home_coverage_ledger.jsonl`

**Interfaces:**
- Produces `build_local_ledger(robocasa_tasks, robocasa_objects, bridge_summary) -> list[dict]`.
- Produces `write_source_manifest(path, sources) -> None`.

- [ ] **Step 1: Write the failing room and provenance test**

```python
def test_ledger_marks_task_code_as_generatable_and_registry_only_as_catalog():
    rows = build_local_ledger(
        [{"task_id": "clean::wipe", "task_family_native": "clean", "manipulated_objects": ["cloth"]}],
        [{"canonical_name": "cloth"}, {"canonical_name": "ornament"}],
        {"dataset_id": "bridge_v2", "total_episodes": 1},
    )
    assert rows[0]["trainability"] == "generatable"
    assert rows[0]["room_zone"] == "kitchen"
    assert any(row["object_id"] == "ornament" and row["trainability"] == "catalog_only" for row in rows)
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tests/test_vla_household_coverage.py::test_ledger_marks_task_code_as_generatable_and_registry_only_as_catalog -v`

Expected: FAIL because `build_local_ledger` is not defined.

- [ ] **Step 3: Implement ledger construction and external source placeholders**

```python
def build_local_ledger(tasks, objects, bridge_summary):
    used = {name for task in tasks for name in task.get("manipulated_objects", [])}
    rows = []
    for task in tasks:
        for name in task.get("manipulated_objects", []):
            rows.append({"dataset_id": "robocasa", "room_zone": "kitchen",
                         "task_id": task["task_id"], "object_id": name,
                         "trainability": "generatable", "evidence": "local_task_code"})
    for obj in objects:
        if obj["canonical_name"] not in used:
            rows.append({"dataset_id": "robocasa", "room_zone": "kitchen",
                         "task_id": "unassigned", "object_id": obj["canonical_name"],
                         "trainability": "catalog_only", "evidence": "local_registry"})
    return rows
```

Record RLBench, BEHAVIOR, DROID, and Bridge in `whole_home_source_manifest.json` with `metadata_status` set to `pending_download`, `local_metadata`, or `verified`—never infer a trajectory count from an object list.

- [ ] **Step 4: Run the tests and generate the ledger**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tests/test_vla_household_coverage.py -v`

Expected: PASS; generated rows contain source provenance and explicit trainability.

### Task 3: Audit and download only metadata needed to close non-kitchen gaps

**Files:**
- Create: `tools/fetch_household_dataset_metadata.py`
- Modify: `tools/vla_household_coverage.py`
- Modify: `tests/test_vla_household_coverage.py`
- Create: `datasets/metadata_cache/rlbench/manifest.json`
- Create: `datasets/metadata_cache/behavior/manifest.json`
- Create: `datasets/metadata_cache/droid/manifest.json`

**Interfaces:**
- Produces `parse_external_manifest(dataset_id: str, data: dict) -> list[dict]`.
- CLI: `python -m tools.fetch_household_dataset_metadata --dataset rlbench --metadata-only`.

- [ ] **Step 1: Write the failing external-manifest test**

```python
def test_external_manifest_needs_task_object_and_action_route():
    rows = parse_external_manifest("rlbench", {
        "records": [{"task_id": "wipe_table", "object_id": "cloth",
                     "room_zone": "living_room", "action_route": "demo_archive"}]
    })
    assert rows[0]["trainability"] == "ready"
```

- [ ] **Step 2: Run the test and confirm failure**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tests/test_vla_household_coverage.py::test_external_manifest_needs_task_object_and_action_route -v`

Expected: FAIL because `parse_external_manifest` is not defined.

- [ ] **Step 3: Implement metadata-only acquisition**

The fetcher must first retrieve a small official manifest or task index, validate checksums when supplied, and write only metadata files into `datasets/metadata_cache/<dataset_id>`. It must refuse full trajectory download unless passed `--download-trajectories` with an explicit object/task filter.

- [ ] **Step 4: Run parser tests and metadata-only commands**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tests/test_vla_household_coverage.py -v`

Expected: PASS; all external rows identify a non-kitchen room zone and whether demo actions are actually available.

### Task 4: Select non-overlapping whole-home categories and 120 eligible objects

**Files:**
- Modify: `tools/vla_household_coverage.py`
- Modify: `tests/test_vla_household_coverage.py`
- Create: `outputs/vla_metadata_survey/whole_home_selection.json`

**Interfaces:**
- Produces `select_whole_home_coverage(rows: list[dict], task_limit: int = 15, object_limit: int = 120) -> dict`.

- [ ] **Step 1: Write the selection tests**

```python
def test_selection_excludes_catalog_only_and_overlapping_buffet_category():
    selection = select_whole_home_coverage(FIXTURE_ROWS)
    assert "arranging_buffet" not in selection["task_ids"]
    assert len(selection["objects"]) == 120
    assert all(row["trainability"] != "catalog_only" for row in selection["rows"])
```

- [ ] **Step 2: Run test and confirm failure**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tests/test_vla_household_coverage.py::test_selection_excludes_catalog_only_and_overlapping_buffet_category -v`

Expected: FAIL because `select_whole_home_coverage` is not defined.

- [ ] **Step 3: Implement deterministic constrained selection**

Rank eligible rows by direct demonstration evidence, then simulated generation route, then object/task diversity. Enforce at least one selected task from each verified room-zone group available in the manifests and reject semantically duplicated category pairs through an explicit exclusion map containing `arranging_buffet` and `setting_the_table`.

- [ ] **Step 4: Run all coverage tests and generate selection JSON**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tests/test_vla_household_coverage.py -v`

Expected: PASS; output contains exactly 15 categories and 120 eligible objects, or terminates with a precise deficit report rather than padding with catalog-only rows.

### Task 5: Render and verify the revised workbook

**Files:**
- Modify: `outputs/vla_metadata_survey/build_task_object_workbook.mjs`
- Modify: `tests/test_vla_metadata_survey.py`
- Create: `outputs/vla_metadata_survey/VLA_全屋家政_15任务_120物体_训练覆盖表.xlsx`

**Interfaces:**
- Consumes `whole_home_selection.json`.
- Main-sheet columns: `task_id`, `task_name`, `room_zone`, `object`, `物体中文名称`, `dataset_id`, `trainability`, `evidence`, `demonstration_count`, `collector_or_archive`.

- [ ] **Step 1: Write a failing workbook-source test**

```python
def test_whole_home_selection_has_no_catalog_only_rows():
    selection = json.loads(Path("outputs/vla_metadata_survey/whole_home_selection.json").read_text())
    assert all(row["trainability"] != "catalog_only" for row in selection["rows"])
```

- [ ] **Step 2: Run the test and confirm failure before rendering**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tests/test_vla_metadata_survey.py -v`

Expected: FAIL until the constrained selection output exists.

- [ ] **Step 3: Update the workbook renderer**

Use the existing blue header, banded rows, frozen header and first columns. Add a `训练状态` sheet that separates `ready`, `generatable`, and excluded `catalog_only` candidates; show blank demonstration counts as `未采集` rather than zero.

- [ ] **Step 4: Render and verify**

Run: `node outputs/vla_metadata_survey/build_task_object_workbook.mjs`

Expected: XLSX export succeeds, the main table contains 120 eligible rows, and the summary reports the per-room-zone distribution and no `catalog_only` headline rows.

- [ ] **Step 5: Run final validation**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tests/test_vla_household_coverage.py tests/test_vla_metadata_survey.py -v`

Expected: PASS.
