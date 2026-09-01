# Household VLA Data Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an evidence-backed indoor household VLA coverage registry and replacement task/object workbook from public robot demonstrations.

**Architecture:** Separate source auditing, download selection, conversion-status verification, and workbook generation. The audit writes normalized JSON evidence records; the workbook consumes only validated records and preserves gaps instead of inventing coverage.

**Tech Stack:** Python 3.10, JSON/JSONL, Hugging Face dataset metadata, local filesystem audit, artifact-tool/OpenPyXL spreadsheet validation.

## Global Constraints

- Indoor household work only; no self-collected data.
- Each final category is a distinct service outcome, not an atomic motion or fixture type.
- Each final category needs at least five ordinary objects, each backed by RGB, task text, and an action record or documented conversion path.
- Appliance controls, household cleaning/waste, storage fixtures, and organization must not become duplicate categories.
- Never claim direct OpenVLA training for a source needing action-space conversion.
- Preserve free disk space by downloading only selected shards tied to an evidence gap.

---

### Task 1: Build a normalized local-source audit

**Files:**
- Create: `tools/audit_household_vla_sources.py`
- Create: `outputs/household_vla_evidence/source_audit.json`
- Test: `tools/tests/test_audit_household_vla_sources.py`

**Interfaces:**
- Consumes: local BEHAVIOR manifests, Bridge/RT-1 directories, RoboCasa directories, and AgiBot task metadata.
- Produces: `list[dict]` records with `dataset`, `instruction`, `rgb_status`, `action_status`, `conversion_status`, `environment`, and `eligible`.

- [ ] **Step 1: Write the failing test**

```python
from tools.audit_household_vla_sources import classify_record

def test_classify_record_requires_rgb_text_and_actions():
    record = classify_record("x", "put cup away", True, True, "direct", "home")
    assert record["eligible"] is True
    assert classify_record("x", "put cup away", True, False, "missing", "home")["eligible"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tools/tests/test_audit_household_vla_sources.py -v`

Expected: FAIL because `classify_record` is not defined.

- [ ] **Step 3: Write minimal implementation**

```python
def classify_record(dataset, instruction, rgb_status, action_status, conversion_status, environment):
    return {
        "dataset": dataset, "instruction": instruction,
        "rgb_status": rgb_status, "action_status": action_status,
        "conversion_status": conversion_status, "environment": environment,
        "eligible": bool(instruction and rgb_status and action_status and environment == "home"),
    }
```

- [ ] **Step 4: Run test and write source audit**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tools/tests/test_audit_household_vla_sources.py -v; C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.audit_household_vla_sources --output outputs/household_vla_evidence/source_audit.json`

Expected: test PASS and JSON contains all audited sources.

### Task 2: Audit candidate public sources and make a minimal download manifest

**Files:**
- Create: `tools/select_household_vla_downloads.py`
- Create: `outputs/household_vla_evidence/download_manifest.json`
- Test: `tools/tests/test_select_household_vla_downloads.py`

**Interfaces:**
- Consumes: `source_audit.json` and verified official source metadata.
- Produces: `download_manifest.json` entries with `dataset`, `task_gap`, `remote_path`, `estimated_bytes`, and `reason`.

- [ ] **Step 1: Write the failing test**

```python
from tools.select_household_vla_downloads import select_under_budget

def test_select_under_budget_keeps_only_eligible_gaps():
    rows = [{"eligible": True, "estimated_bytes": 10, "task_gap": "hygiene"},
            {"eligible": False, "estimated_bytes": 1, "task_gap": "outdoor"}]
    assert select_under_budget(rows, 10) == [rows[0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tools/tests/test_select_household_vla_downloads.py -v`

Expected: FAIL because `select_under_budget` is not defined.

- [ ] **Step 3: Write minimal implementation**

```python
def select_under_budget(rows, budget_bytes):
    selected, used = [], 0
    for row in rows:
        if row["eligible"] and used + row["estimated_bytes"] <= budget_bytes:
            selected.append(row)
            used += row["estimated_bytes"]
    return selected
```

- [ ] **Step 4: Run validation and create manifest**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tools/tests/test_select_household_vla_downloads.py -v; C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.select_household_vla_downloads --audit outputs/household_vla_evidence/source_audit.json --output outputs/household_vla_evidence/download_manifest.json --min-free-gb 220`

Expected: manifest contains no outdoor, procurement, child-only, or unsupported source.

### Task 3: Download and verify selected shards

**Files:**
- Create: `tools/download_household_vla_shards.py`
- Create: `outputs/household_vla_evidence/download_verification.json`
- Test: `tools/tests/test_download_household_vla_shards.py`

**Interfaces:**
- Consumes: `download_manifest.json`.
- Produces: per-shard file count, RGB/text/action availability, checksum or size, and remaining disk space.

- [ ] **Step 1: Write the failing test**

```python
from tools.download_household_vla_shards import verify_triplet

def test_verify_triplet_requires_all_modalities(tmp_path):
    assert verify_triplet(tmp_path / "a.mp4", tmp_path / "a.json", tmp_path / "a.parquet") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tools/tests/test_download_household_vla_shards.py -v`

Expected: FAIL because `verify_triplet` is not defined.

- [ ] **Step 3: Write minimal implementation**

```python
def verify_triplet(rgb_path, text_path, action_path):
    return all(path.is_file() and path.stat().st_size > 0 for path in (rgb_path, text_path, action_path))
```

- [ ] **Step 4: Run selected download and verification**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.download_household_vla_shards --manifest outputs/household_vla_evidence/download_manifest.json --root datasets --verification outputs/household_vla_evidence/download_verification.json --min-free-gb 220`

Expected: all accepted rows are verified; failures remain gaps and do not enter training records.

### Task 4: Generate and validate the replacement household workbook

**Files:**
- Create: `tools/build_household_vla_evidence_workbook.py`
- Create: `outputs/household_vla_evidence/家庭VLA_证据任务物体操作表_重构版.xlsx`
- Test: `tools/tests/test_build_household_vla_evidence_workbook.py`

**Interfaces:**
- Consumes: source audit, download verification, and selected task/object evidence records.
- Produces: workbook sheets `任务物体操作`, `数据源审计`, `训练转换状态`, and `覆盖缺口`.

- [ ] **Step 1: Write the failing test**

```python
from tools.build_household_vla_evidence_workbook import category_is_valid

def test_category_needs_five_nonduplicate_objects():
    assert category_is_valid(["cup", "plate", "bowl", "fork", "knife"]) is True
    assert category_is_valid(["cup", "cup", "plate"]) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tools/tests/test_build_household_vla_evidence_workbook.py -v`

Expected: FAIL because `category_is_valid` is not defined.

- [ ] **Step 3: Write minimal implementation**

```python
def category_is_valid(objects):
    return len(set(objects)) >= 5
```

- [ ] **Step 4: Build and inspect workbook**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tools/tests/test_build_household_vla_evidence_workbook.py -v; C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.build_household_vla_evidence_workbook --audit outputs/household_vla_evidence/source_audit.json --verification outputs/household_vla_evidence/download_verification.json --output outputs/household_vla_evidence/家庭VLA_证据任务物体操作表_重构版.xlsx`

Expected: all final categories have five or more distinct everyday objects; each row has source instruction and training/conversion status; unsupported categories are only in `覆盖缺口`.

### Task 5: Publish the data-to-training handoff

**Files:**
- Create: `docs/HOUSEHOLD_VLA_DATA_HANDOFF.md`
- Test: manual JSON and workbook cross-check.

**Interfaces:**
- Consumes: all Task 1-4 artifacts.
- Produces: exact training-ready source list, required conversion commands, and excluded-source rationale.

- [ ] **Step 1: Write handoff content**

Include direct OXE training, explicit BEHAVIOR action conversion, selected RoboCasa/AIRoA adapters, and the statement that human-video-only sources are not robot-action training data.

- [ ] **Step 2: Cross-check counts**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -c "import json; a=json.load(open('outputs/household_vla_evidence/source_audit.json',encoding='utf-8')); print(len(a))"`

Expected: the handoff references only audit records present in the JSON.

- [ ] **Step 3: Commit when repository control is available**

The current workspace is not a Git repository; preserve all artifacts but do not fabricate a commit.
