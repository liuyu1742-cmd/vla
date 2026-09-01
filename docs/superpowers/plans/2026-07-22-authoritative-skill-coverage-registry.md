# Authoritative Skill Coverage Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable registry that separates the Word document's 15 service-task classes, the workbook's 15 dataset groups and 123 candidate objects, and a sparse task-object applicability matrix without treating all `15 × 123` pairs as valid.

**Architecture:** Read the source `.xlsx` without modifying it, normalize its 138 source relations into deterministic JSON registries, and combine them with versioned catalogs for the 15 service tasks and 15 action primitives. Generate all 1,845 task-object pairs with explicit applicability and validation states; only semantically applicable objects with passed execution evidence may count toward the 120-object acceptance target.

**Tech Stack:** Python 3.10+ standard library (`dataclasses`, `enum`, `json`, `zipfile`, `xml.etree.ElementTree`, `hashlib`), `unittest`, PowerShell, JSON.

## Global Constraints

- The source Word and Excel files are read-only; no command may save, rename, move, or rewrite either source.
- The final task taxonomy is the 15 service-task classes in Chapter 5.
- The workbook contains 15 dataset groups, 138 task-object relations, and 123 unique candidate objects; these are source facts, not execution-success claims.
- A candidate object counts toward the `>=120` target only after it has at least one applicable service-task edge and passed closed-loop evidence.
- Every non-applicable pair must remain explicit in the generated matrix; absence must not be interpreted as applicability.
- Generated JSON is UTF-8, deterministic, sorted, and includes source path, SHA-256, sheet, row, and generator version.
- Registry tooling must not import OpenVLA, Torch, RoboCasa, or add dependencies to either runtime environment.
- Simulator truth may be referenced only by later execution evidence, never by the source-registry builder.
- No generated report may claim real-robot completion before real hardware exists.

---

## Locked File Structure

```text
tools/skill_coverage/__init__.py
tools/skill_coverage/contracts.py
tools/skill_coverage/xlsx_reader.py
tools/skill_coverage/registry_builder.py
tools/skill_coverage/matrix_builder.py
tools/build_authoritative_skill_coverage.py
tools/validate_skill_coverage.py
data/skill_coverage/service_tasks.json
data/skill_coverage/action_primitives.json
data/skill_coverage/applicability_policy.json
data/skill_coverage/generated/dataset_tasks.json
data/skill_coverage/generated/object_candidates.json
data/skill_coverage/generated/source_relations.json
data/skill_coverage/generated/task_object_matrix.json
data/skill_coverage/generated/coverage_summary.json
tests/test_skill_coverage_contracts.py
tests/test_skill_coverage_xlsx_reader.py
tests/test_skill_coverage_registry_builder.py
tests/test_skill_coverage_matrix_builder.py
tests/test_build_authoritative_skill_coverage.py
tests/test_authoritative_skill_coverage_snapshot.py
docs/SKILL_COVERAGE_REGISTRY.md
```

The source readers, pure builders, policy data, generated snapshots, and validation CLI remain separate so a source change can be detected without changing classification logic.

---

### Task 1: Registry Contracts and Validation States

**Files:**
- Create: `tools/skill_coverage/__init__.py`
- Create: `tools/skill_coverage/contracts.py`
- Create: `tests/test_skill_coverage_contracts.py`

**Interfaces:**
- Produces: `Applicability`, `ValidationState`, `validate_object_candidate(record)`, `validate_matrix_entry(record)`.
- Consumed by: registry builder, matrix builder, validation CLI.

- [ ] **Step 1: Write the failing contract tests**

```python
import unittest


class SkillCoverageContractTests(unittest.TestCase):
    def test_candidate_requires_source_rows(self):
        from tools.skill_coverage.contracts import validate_object_candidate

        with self.assertRaisesRegex(ValueError, "source_rows"):
            validate_object_candidate(
                {
                    "object_id": "water_cup",
                    "display_name": "水杯",
                    "dataset_task_groups": ["food_serving"],
                    "source_rows": [],
                }
            )

    def test_not_applicable_pair_cannot_pass(self):
        from tools.skill_coverage.contracts import validate_matrix_entry

        with self.assertRaisesRegex(ValueError, "not_applicable"):
            validate_matrix_entry(
                {
                    "service_task_id": "security_entry",
                    "object_id": "water_cup",
                    "applicability": "not_applicable",
                    "validation_state": "passed",
                }
            )

    def test_needs_review_pair_cannot_count_as_passed(self):
        from tools.skill_coverage.contracts import validate_matrix_entry

        with self.assertRaisesRegex(ValueError, "needs_review"):
            validate_matrix_entry(
                {
                    "service_task_id": "item_delivery",
                    "object_id": "water_cup",
                    "applicability": "needs_review",
                    "validation_state": "passed",
                }
            )
```

- [ ] **Step 2: Run the test and verify the module is missing**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest tests.test_skill_coverage_contracts -v
```

Expected: `ModuleNotFoundError: No module named 'tools.skill_coverage'`.

- [ ] **Step 3: Implement the contracts**

```python
from __future__ import annotations

from enum import Enum
from typing import Any


class Applicability(str, Enum):
    DIRECT = "direct"
    DEVICE_CONTROL = "device_control"
    COMPOSITE_RESOURCE = "composite_resource"
    NOT_APPLICABLE = "not_applicable"
    NEEDS_REVIEW = "needs_review"


class ValidationState(str, Enum):
    UNTESTED = "untested"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


def _require_text(record: dict[str, Any], key: str) -> None:
    if not isinstance(record.get(key), str) or not record[key].strip():
        raise ValueError(f"{key} is required")


def validate_object_candidate(record: dict[str, Any]) -> dict[str, Any]:
    _require_text(record, "object_id")
    _require_text(record, "display_name")
    if not record.get("dataset_task_groups"):
        raise ValueError("dataset_task_groups are required")
    if not record.get("source_rows"):
        raise ValueError("source_rows are required")
    return record


def validate_matrix_entry(record: dict[str, Any]) -> dict[str, Any]:
    _require_text(record, "service_task_id")
    _require_text(record, "object_id")
    applicability = Applicability(record["applicability"])
    validation = ValidationState(record["validation_state"])
    if validation is ValidationState.PASSED and applicability in {
        Applicability.NOT_APPLICABLE,
        Applicability.NEEDS_REVIEW,
    }:
        raise ValueError(f"{applicability.value} pair cannot be passed")
    return record
```

Create `tools/skill_coverage/__init__.py` with exports for the two enums and validators.

- [ ] **Step 4: Run the contract tests**

Expected: 3 tests pass.

- [ ] **Step 5: Commit the isolated contract change**

```powershell
git add tools/skill_coverage tests/test_skill_coverage_contracts.py
git commit -m "feat: define skill coverage registry contracts"
```

---

### Task 2: Dependency-Free Workbook Reader

**Files:**
- Create: `tools/skill_coverage/xlsx_reader.py`
- Create: `tests/test_skill_coverage_xlsx_reader.py`

**Interfaces:**
- Produces: `read_sheet_table(path: Path, sheet_name: str) -> list[dict[str, str]]`.
- Consumed by: `build_authoritative_skill_coverage.py`.

- [ ] **Step 1: Write a failing test with an in-memory-style minimal XLSX ZIP fixture**

The test creates a temporary ZIP containing `xl/workbook.xml`, workbook relationships, shared strings, and one worksheet. It must assert that `task_id`, Chinese names, and numeric cells are returned as strings without changing the file hash.

```python
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path


class XlsxReaderTests(unittest.TestCase):
    def test_reads_shared_strings_and_preserves_source(self):
        from tools.skill_coverage.xlsx_reader import read_sheet_table

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.xlsx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "xl/workbook.xml",
                    '<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
                    '<sheet name="任务物体数据表" sheetId="1" r:id="rId1"/></sheets></workbook>',
                )
                archive.writestr(
                    "xl/_rels/workbook.xml.rels",
                    '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
                )
                strings = ["task_id", "task_name", "object", "物体中文名称", "food_serving", "送餐饮水", "water_cup", "水杯"]
                archive.writestr(
                    "xl/sharedStrings.xml",
                    '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                    + "".join(f"<si><t>{value}</t></si>" for value in strings)
                    + "</sst>",
                )
                archive.writestr(
                    "xl/worksheets/sheet1.xml",
                    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
                    '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>'
                    '<c r="C1" t="s"><v>2</v></c><c r="D1" t="s"><v>3</v></c></row>'
                    '<row r="2"><c r="A2" t="s"><v>4</v></c><c r="B2" t="s"><v>5</v></c>'
                    '<c r="C2" t="s"><v>6</v></c><c r="D2" t="s"><v>7</v></c></row>'
                    '</sheetData></worksheet>',
                )
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = read_sheet_table(path, "任务物体数据表")
            after = hashlib.sha256(path.read_bytes()).hexdigest()

        self.assertEqual(rows[0]["object"], "water_cup")
        self.assertEqual(rows[0]["物体中文名称"], "水杯")
        self.assertEqual(before, after)
```

- [ ] **Step 2: Run the test and verify failure because the reader is missing**

- [ ] **Step 3: Implement the standard-library XLSX reader**

Implementation requirements:

```python
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def _column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    result = 0
    for letter in letters:
        result = result * 26 + ord(letter) - ord("A") + 1
    return result - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.text or "" for node in item.findall(f".//{{{MAIN}}}t")) for item in root]


def _sheet_target(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = next(
        (node for node in workbook.findall(f".//{{{MAIN}}}sheet") if node.attrib.get("name") == sheet_name),
        None,
    )
    if sheet is None:
        raise ValueError(f"sheet not found: {sheet_name}")
    relation_id = sheet.attrib[f"{{{OFFICE_REL}}}id"]
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relation = next(node for node in relationships.findall(f"{{{PACKAGE_REL}}}Relationship") if node.attrib["Id"] == relation_id)
    target = relation.attrib["Target"].replace("\\", "/")
    return target if target.startswith("xl/") else f"xl/{target}"


def _cell_text(cell: ET.Element, strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(f".//{{{MAIN}}}t"))
    value = cell.find(f"{{{MAIN}}}v")
    raw = "" if value is None or value.text is None else value.text
    return strings[int(raw)] if cell_type == "s" and raw else raw


def read_sheet_table(path: Path, sheet_name: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(path, "r") as archive:
        strings = _shared_strings(archive)
        root = ET.fromstring(archive.read(_sheet_target(archive, sheet_name)))
    rows: list[list[str]] = []
    for row in root.findall(f".//{{{MAIN}}}row"):
        values: list[str] = []
        for cell in row.findall(f"{{{MAIN}}}c"):
            index = _column_index(cell.attrib["r"])
            values.extend([""] * (index + 1 - len(values)))
            values[index] = _cell_text(cell, strings).strip()
        rows.append(values)
    if not rows:
        return []
    headers = rows[0]
    return [
        {header: values[index] if index < len(values) else "" for index, header in enumerate(headers) if header}
        for values in rows[1:]
        if any(values)
    ]
```

- [ ] **Step 4: Run the XLSX reader tests**

Expected: source hash before and after is identical and the Chinese sheet content is decoded correctly.

- [ ] **Step 5: Commit**

```powershell
git add tools/skill_coverage/xlsx_reader.py tests/test_skill_coverage_xlsx_reader.py
git commit -m "feat: read authoritative workbook without dependencies"
```

---

### Task 3: Canonical Service Tasks, Primitives, and Mapping Policy

**Files:**
- Create: `data/skill_coverage/service_tasks.json`
- Create: `data/skill_coverage/action_primitives.json`
- Create: `data/skill_coverage/applicability_policy.json`
- Extend test: `tests/test_skill_coverage_contracts.py`

**Interfaces:**
- Produces: immutable, versioned catalogs used by the matrix builder.
- Source references: Word Chapter 3 Table 3-1 and Chapter 5 Table 5-1.

- [ ] **Step 1: Add failing catalog integrity tests**

Tests must assert exactly 15 unique service task IDs, exactly 15 unique primitive IDs, all policy targets exist, and every workbook dataset group has one conservative primary service-task mapping.

- [ ] **Step 2: Create the 15 service-task catalog**

Use these exact stable IDs and Chinese names:

```json
[
  ["appliance_management", "家电综合管理任务"],
  ["environment_adjustment", "环境调节任务"],
  ["security_entry", "智慧安防任务（入口安全）"],
  ["security_environment", "智慧安防任务（环境安全）"],
  ["cleaning_service", "清洁服务任务"],
  ["cooking_assistance", "烹饪辅助任务"],
  ["maintenance_management", "养护管理任务"],
  ["item_delivery", "物品递送任务"],
  ["organization_storage", "整理收纳任务"],
  ["energy_management", "能源管理任务"],
  ["facility_monitoring", "设施监控任务"],
  ["scene_linkage", "场景联动任务"],
  ["health_care", "健康关怀任务"],
  ["entertainment_service", "娱乐服务任务"],
  ["other_comprehensive", "其他综合任务"]
]
```

Each stored object also includes `ordinal`, `source_section`, `backend_classes`, and `acceptance_kind` (`physical`, `device_state`, `monitoring`, or `composite`).

- [ ] **Step 3: Create the 15 core primitive catalog**

Use: `握取、捏取、托举、轻放、推入、悬挂、擦拭、拖洗、折叠、堆叠、排列、推送、搬运、旋转、递送`. Normalize the source typo `托洗` to `拖洗` while retaining `source_label: "托洗"` for auditability.

- [ ] **Step 4: Create the conservative applicability policy**

Primary mappings assign each dataset group to one service task only:

```json
{
  "cleaning": "cleaning_service",
  "organizing": "organization_storage",
  "smart_cooking": "cooking_assistance",
  "appliance_management": "appliance_management",
  "security_monitoring": "security_entry",
  "laundry": "cleaning_service",
  "waste_disposal": "cleaning_service",
  "clothing_care": "organization_storage",
  "window_care": "cleaning_service",
  "bedroom_service": "organization_storage",
  "food_serving": "item_delivery",
  "object_fetching": "item_delivery",
  "elderly_assistance": "health_care",
  "maintenance_management": "maintenance_management",
  "entertainment_service": "entertainment_service"
}
```

Add exact-object secondary mappings only where the relationship is defensible, including:

- `security_camera` keeps its entrance-security route and also maps to `security_environment`.
- `fire_alarm` and `fire_extinguisher` explicitly override `security_entry` to `not_applicable` and map to `security_environment` as `direct`.
- `air_conditioner`, `fan`, `electric_curtain`, `window`, `blind`, `ceiling_light`, `bedroom_lamp` → `environment_adjustment`.
- Major appliances and lights → `energy_management` as `device_control`.
- `robot_vacuum`, `fire_alarm`, `washing_machine`, `air_conditioner`, `refrigerator`, `smart_lock` → `facility_monitoring`.
- `ceiling_light`, `air_conditioner`, `television`, `electric_curtain`, `smart_lock`, `speaker` → `scene_linkage` as `composite_resource`.
- `speaker`, `mobile_phone` → `other_comprehensive` as `device_control`.

All unlisted pairs default to `not_applicable`; no wildcard may mark all objects applicable to a service task.

- [ ] **Step 5: Run catalog integrity tests and commit**

Expected: exactly 15/15 catalogs and all 15 service tasks have at least one primary or secondary object route.

---

### Task 4: Normalize 138 Relations into 15 Dataset Groups and 123 Candidates

**Files:**
- Create: `tools/skill_coverage/registry_builder.py`
- Create: `tests/test_skill_coverage_registry_builder.py`

**Interfaces:**
- Consumes: rows from `read_sheet_table`.
- Produces: `build_source_registries(rows) -> {dataset_tasks, object_candidates, source_relations}`.

- [ ] **Step 1: Write failing duplicate-object and provenance tests**

The fixture includes `remote_control` in three task groups. Assert one candidate object, three source rows, three dataset groups, and three unchanged source relations.

- [ ] **Step 2: Implement strict row normalization**

```python
REQUIRED_COLUMNS = (
    "task_id", "task_name", "object", "物体中文名称",
    "images", "videos", "annotations", "metadata", "object_dir",
)


def as_int(value: str, field: str, source_row: int) -> int:
    try:
        return int(float(value or "0"))
    except ValueError as error:
        raise ValueError(f"row {source_row}: invalid {field}={value!r}") from error


def build_source_registries(rows: list[dict[str, str]]) -> dict[str, list[dict]]:
    # Enumerate source rows from Excel row 2, preserve each relation, verify that
    # repeated object IDs have the same Chinese name, aggregate task groups and
    # source rows, and sort every output by stable ID.
    dataset_tasks: dict[str, dict] = {}
    candidates: dict[str, dict] = {}
    relations: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    for offset, row in enumerate(rows, start=2):
        missing = [key for key in REQUIRED_COLUMNS if key not in row]
        if missing:
            raise ValueError(f"row {offset}: missing columns {missing}")
        task_id = row["task_id"].strip()
        task_name = row["task_name"].strip()
        object_id = row["object"].strip()
        display_name = row["??????"].strip()
        if not all((task_id, task_name, object_id, display_name)):
            raise ValueError(f"row {offset}: blank identifier or name")
        pair = (task_id, object_id)
        if pair in seen_pairs:
            raise ValueError(f"row {offset}: duplicate relation {pair}")
        seen_pairs.add(pair)

        counts = {
            key: as_int(row[key], key, offset)
            for key in ("images", "videos", "annotations", "metadata")
        }
        if any(value < 0 for value in counts.values()):
            raise ValueError(f"row {offset}: negative media count")
        relation = {
            "source_row": offset,
            "dataset_task_id": task_id,
            "dataset_task_name": task_name,
            "object_id": object_id,
            "display_name": display_name,
            **counts,
            "object_dir": row["object_dir"].strip(),
        }
        relations.append(relation)

        task = dataset_tasks.setdefault(
            task_id,
            {"dataset_task_id": task_id, "display_name": task_name, "source_rows": []},
        )
        if task["display_name"] != task_name:
            raise ValueError(f"row {offset}: inconsistent task name for {task_id}")
        task["source_rows"].append(offset)

        candidate = candidates.setdefault(
            object_id,
            {
                "object_id": object_id,
                "display_name": display_name,
                "dataset_task_groups": [],
                "source_rows": [],
            },
        )
        if candidate["display_name"] != display_name:
            raise ValueError(f"row {offset}: inconsistent object name for {object_id}")
        candidate["dataset_task_groups"].append(task_id)
        candidate["source_rows"].append(offset)

    for task in dataset_tasks.values():
        task["source_rows"].sort()
        task["relation_count"] = len(task["source_rows"])
    for candidate in candidates.values():
        candidate["dataset_task_groups"] = sorted(set(candidate["dataset_task_groups"]))
        candidate["source_rows"].sort()

    return {
        "dataset_tasks": sorted(dataset_tasks.values(), key=lambda item: item["dataset_task_id"]),
        "object_candidates": sorted(candidates.values(), key=lambda item: item["object_id"]),
        "source_relations": sorted(relations, key=lambda item: item["source_row"]),
    }
```

The implementation must raise on missing columns, blank IDs, inconsistent Chinese names, negative counts, or duplicate `(task_id, object_id)` relations. It must not infer service-task applicability.

- [ ] **Step 3: Run builder tests**

Expected: duplicate IDs are deduplicated only in `object_candidates`; source relations remain lossless.

- [ ] **Step 4: Commit**

```powershell
git add tools/skill_coverage/registry_builder.py tests/test_skill_coverage_registry_builder.py
git commit -m "feat: normalize task object source registries"
```

---

### Task 5: Sparse 15-by-123 Applicability Matrix

**Files:**
- Create: `tools/skill_coverage/matrix_builder.py`
- Create: `tests/test_skill_coverage_matrix_builder.py`

**Interfaces:**
- Consumes: service-task catalog, candidate objects, applicability policy.
- Produces: `build_task_object_matrix(service_tasks, candidates, policy) -> list[dict]` and `summarize_coverage(service_tasks, candidates, matrix, acceptance_target=120) -> dict`.

- [ ] **Step 1: Write failing sparse-matrix tests**

Tests must prove:

- Water cup is applicable to `item_delivery` and not applicable to `security_entry`.
- Fire alarm is applicable to `security_environment` and `facility_monitoring`, and explicitly not applicable to `security_entry`.
- Air conditioner is applicable to appliance, environment, energy, facility, and scene tasks but not cooking.
- Every candidate has exactly 15 matrix entries.
- Applicability alone leaves `validation_state="untested"` and does not increase `validated_object_count`.

- [ ] **Step 2: Implement explicit matrix generation**

```python
def build_task_object_matrix(service_tasks, candidates, policy):
    entries = []
    primary = policy["primary_by_dataset_task"]
    secondary = policy["secondary_by_object"]
    for candidate in candidates:
        applicable = {
            primary[group]: "direct"
            for group in candidate["dataset_task_groups"]
        }
        for rule in secondary.get(candidate["object_id"], []):
            applicable[rule["service_task_id"]] = rule["applicability"]
        for task in service_tasks:
            status = applicable.get(task["service_task_id"], "not_applicable")
            entries.append(
                {
                    "service_task_id": task["service_task_id"],
                    "object_id": candidate["object_id"],
                    "applicability": status,
                    "validation_state": "untested",
                    "evidence": [],
                }
            )
    return sorted(entries, key=lambda item: (item["service_task_id"], item["object_id"]))
```

`summarize_coverage` reports separate counts for candidate, semantically mapped, validated, failed, blocked, and target gap. It must never use semantically mapped count as validated count.

- [ ] **Step 3: Run the matrix tests**

Expected: all non-applicable examples remain explicit and `validated_object_count == 0` before evidence is loaded.

- [ ] **Step 4: Commit**

```powershell
git add tools/skill_coverage/matrix_builder.py tests/test_skill_coverage_matrix_builder.py
git commit -m "feat: build sparse task object applicability matrix"
```

---

### Task 6: Deterministic Builder CLI and Source-Immutability Guard

**Files:**
- Create: `tools/build_authoritative_skill_coverage.py`
- Create: `tests/test_build_authoritative_skill_coverage.py`

**Interfaces:**
- CLI consumes: `--xlsx`, `--sheet`, `--output-dir`, exact expected counts.
- CLI produces: five deterministic generated JSON files plus SHA-256 provenance.

- [ ] **Step 1: Write failing CLI tests**

Test a miniature workbook, assert output filenames and source hash stability, and assert the command fails if expected counts do not match actual counts.

- [ ] **Step 2: Implement an argument-driven `main(argv=None) -> int`**

Required arguments/defaults:

```text
--xlsx PATH                         required
--sheet 任务物体数据表              default
--output-dir data/skill_coverage/generated
--expected-dataset-tasks 15
--expected-relations 138
--expected-candidates 123
```

The CLI computes the source SHA-256 before reading, builds registries and matrix, computes it again, aborts if the hash changed, and writes JSON via a temporary file followed by `Path.replace()` inside the output directory. It never writes beside the source workbook.

- [ ] **Step 3: Run the CLI unit tests**

- [ ] **Step 4: Run the authoritative build**

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.build_authoritative_skill_coverage `
  --xlsx "C:\RobotProject\RobotProject\docs\家庭服务任务物体数据表.xlsx" `
  --sheet "任务物体数据表" `
  --output-dir "C:\OpenVLA-Simulator\data\skill_coverage\generated" `
  --expected-dataset-tasks 15 `
  --expected-relations 138 `
  --expected-candidates 123
```

Expected summary:

```json
{
  "dataset_task_count": 15,
  "source_relation_count": 138,
  "candidate_object_count": 123,
  "matrix_entry_count": 1845,
  "validated_object_count": 0,
  "acceptance_target": 120,
  "remaining_validation_gap": 120
}
```

- [ ] **Step 5: Verify original workbook hash and timestamp did not change**

```powershell
Get-FileHash -Algorithm SHA256 "C:\RobotProject\RobotProject\docs\家庭服务任务物体数据表.xlsx"
Get-Item "C:\RobotProject\RobotProject\docs\家庭服务任务物体数据表.xlsx" | Select-Object Length,LastWriteTime
```

- [ ] **Step 6: Commit generated source snapshots and CLI**

```powershell
git add tools/build_authoritative_skill_coverage.py tests/test_build_authoritative_skill_coverage.py data/skill_coverage/generated
git commit -m "feat: generate authoritative skill coverage snapshot"
```

---

### Task 7: Validation CLI, Snapshot Test, and PyCharm Documentation

**Files:**
- Create: `tools/validate_skill_coverage.py`
- Create: `tests/test_authoritative_skill_coverage_snapshot.py`
- Create: `docs/SKILL_COVERAGE_REGISTRY.md`

**Interfaces:**
- Consumes: generated registries, matrix, later execution-evidence paths.
- Produces: exit code 0 only when structural/source checks pass; acceptance remains false until 120 objects have passed evidence.

- [ ] **Step 1: Write the failing snapshot test**

```python
import json
import unittest
from pathlib import Path


class AuthoritativeSnapshotTests(unittest.TestCase):
    def test_authoritative_counts_and_sparse_matrix(self):
        root = Path(__file__).resolve().parents[1] / "data" / "skill_coverage" / "generated"
        summary = json.loads((root / "coverage_summary.json").read_text(encoding="utf-8"))
        matrix = json.loads((root / "task_object_matrix.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["dataset_task_count"], 15)
        self.assertEqual(summary["source_relation_count"], 138)
        self.assertEqual(summary["candidate_object_count"], 123)
        self.assertEqual(len(matrix), 15 * 123)
        self.assertEqual(summary["validated_object_count"], 0)
        self.assertTrue(any(item["applicability"] == "not_applicable" for item in matrix))
```

- [ ] **Step 2: Implement validation CLI**

It validates contracts, unique IDs, exactly 15 entries per candidate, policy references, source SHA-256, evidence paths, and separates structural success from project acceptance. Output must contain both:

```text
STRUCTURAL_VALIDATION: PASS
PROJECT_ACCEPTANCE: NOT_READY validated_objects=0 target=120
```

- [ ] **Step 3: Document PyCharm and terminal commands**

`docs/SKILL_COVERAGE_REGISTRY.md` must state:

- Interpreter: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe`.
- Working directory: `C:\OpenVLA-Simulator`.
- Builder module: `tools.build_authoritative_skill_coverage`.
- Validator module: `tools.validate_skill_coverage`.
- Original source files are read-only.
- `candidate_object_count=123` is not the same as `validated_object_count`.
- How a later RoboCasa/device report is attached as evidence without editing generated source relations.

- [ ] **Step 4: Run complete registry tests**

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest `
  tests.test_skill_coverage_contracts `
  tests.test_skill_coverage_xlsx_reader `
  tests.test_skill_coverage_registry_builder `
  tests.test_skill_coverage_matrix_builder `
  tests.test_build_authoritative_skill_coverage `
  tests.test_authoritative_skill_coverage_snapshot -v
```

Expected: all tests pass.

- [ ] **Step 5: Run structural validation**

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.validate_skill_coverage `
  --registry-dir "C:\OpenVLA-Simulator\data\skill_coverage\generated"
```

Expected: structural pass and project acceptance not ready with an explicit gap of 120, because execution evidence belongs to later phases.

- [ ] **Step 6: Commit**

```powershell
git add tools/validate_skill_coverage.py tests/test_authoritative_skill_coverage_snapshot.py docs/SKILL_COVERAGE_REGISTRY.md
git commit -m "docs: add auditable skill coverage workflow"
```

---

## Self-Review Checklist

- Spec coverage: the plan distinguishes all three 15-class taxonomies, 138 relations, 123 candidates, sparse applicability, `>=120` passed objects, source immutability, and evidence states.
- Scope boundary: human-video pose/contact parsing and OpenVLA execution are intentionally excluded from this first plan and remain governed by `docs/superpowers/specs/2026-07-22-human-video-skill-ir-transfer-design.md`.
- Type consistency: applicability and validation strings are defined once in Task 1 and reused unchanged.
- Count consistency: expected authoritative outputs are 15 dataset tasks, 138 relations, 123 candidates, and 1,845 explicit pairs.
- Placeholder scan: implementation must not contain `TODO`, `TBD`, `pass`, ellipses, or implicit “appropriate handling” language.
- Truthfulness: the first generated snapshot must report zero validated objects until real simulator/device evidence is linked.

## Completion Gate

This plan is complete only when the original Excel SHA-256 is unchanged, all registry tests pass, structural validation passes, generated counts are exactly `15/138/123/1845`, every candidate has at least one conservative semantic mapping, non-applicable pairs are explicit, and project acceptance remains honestly marked `NOT_READY` until execution evidence exists.
