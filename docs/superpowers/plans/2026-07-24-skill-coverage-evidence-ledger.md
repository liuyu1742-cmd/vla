# Skill Coverage Evidence Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已验证的 `organizing::toy` L3 hybrid 闭环证据安全、可重放地合并进 15 类任务 / 123 个候选物体覆盖注册表。

**Architecture:** 使用非生成的 evidence ledger 保存规范化证据记录；纯函数负责验证 acceptance bundle、映射 relation、叠加矩阵并重算汇总；CLI 负责原子写入。权威注册表构建入口使用同名包包装 legacy 模块，在默认输出目录重建时自动重放 ledger。

**Tech Stack:** Python 3.10/3.11、标准库 `json/hashlib/pathlib/tempfile`、现有 skill-coverage contracts、`unittest`。

## Global Constraints

- 不修改 `C:\RobotProject` 或权威 Excel/Word。
- 只有真实闭环成功证据才能产生 `validation_state=passed`。
- `organizing::toy` 必须映射到 `organization_storage::toy`。
- hybrid 证据必须记录 `pure_autonomous_vla=false` 和 final-contact servo 使用。
- smoke、训练损失、目录存在和失败报告不得计入覆盖。
- 导入失败时 ledger、matrix、summary 不得部分更新。

---

### Task 1: 验收包与 relation 映射纯函数

**Files:**
- Create: `tests/test_skill_coverage_evidence_merge.py`
- Create: `tools/skill_coverage/evidence_merge.py`

**Interfaces:**
- Consumes: acceptance bundle path、Task-2 contract、applicability policy、project root。
- Produces: `validate_hybrid_acceptance_bundle(path, contract, policy, project_root) -> dict[str, Any]`。

- [ ] **Step 1: Write the failing tests**

```python
def test_maps_organizing_toy_to_organization_storage(self):
    record = validate_hybrid_acceptance_bundle(
        self.bundle_path, self.contract, self.policy, self.project_root
    )
    self.assertEqual("organizing::toy", record["relation_key"])
    self.assertEqual("organization_storage", record["service_task_id"])
    self.assertEqual("L3", record["evidence_level"])
    self.assertFalse(record["pure_autonomous_vla"])

def test_rejects_bundle_without_three_native_successes(self):
    bundle = self.valid_bundle()
    bundle["aggregate"]["successful_seeds"] = 2
    with self.assertRaisesRegex(ValueError, "successful seeds"):
        self.validate(bundle)

def test_rejects_autonomous_claim_for_hybrid_bundle(self):
    bundle = self.valid_bundle()
    bundle["evaluation_mode"]["pure_autonomous_vla"] = True
    with self.assertRaisesRegex(ValueError, "pure autonomous"):
        self.validate(bundle)
```

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest tests.test_skill_coverage_evidence_merge -v
```

Expected: import failure because `tools.skill_coverage.evidence_merge` does not exist.

- [ ] **Step 3: Implement the minimal validator**

```python
def validate_hybrid_acceptance_bundle(
    bundle_path: Path,
    contract: list[dict[str, Any]],
    policy: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    bundle = load_json(bundle_path)
    require_hybrid_schema_and_three_successes(bundle)
    relation = index_contract(contract)[bundle["relation_key"]]
    dataset_task_id, object_id = bundle["relation_key"].split("::", 1)
    service_task_id = policy["primary_by_dataset_task"][dataset_task_id]
    reports = validate_referenced_reports(bundle, project_root)
    return normalized_ledger_record(
        bundle, bundle_path, reports, dataset_task_id, service_task_id, object_id
    )
```

The helper must validate exact report relation/seed/split/fingerprints, native predicates,
coverage flags and referenced file existence before returning.

- [ ] **Step 4: Run tests to verify GREEN**

Run the Task 1 command. Expected: all Task 1 tests pass.

---

### Task 2: Ledger merge and deterministic matrix overlay

**Files:**
- Extend: `tests/test_skill_coverage_evidence_merge.py`
- Extend: `tools/skill_coverage/evidence_merge.py`

**Interfaces:**
- Consumes: existing ledger、baseline matrix、service tasks、object candidates。
- Produces:
  - `merge_ledger_record(ledger, record) -> dict[str, Any]`
  - `overlay_evidence_ledger(matrix, ledger) -> list[dict[str, Any]]`
  - `coverage_with_preserved_metadata(summary, service_tasks, candidates, matrix) -> dict[str, Any]`

- [ ] **Step 1: Add failing overlay tests**

```python
def test_overlay_marks_only_organization_storage_toy_passed(self):
    updated = overlay_evidence_ledger(self.matrix, self.ledger)
    passed = [item for item in updated if item["validation_state"] == "passed"]
    self.assertEqual(1, len(passed))
    self.assertEqual(("organization_storage", "toy"),
                     (passed[0]["service_task_id"], passed[0]["object_id"]))

def test_overlay_rejects_not_applicable_target(self):
    bad = copy.deepcopy(self.ledger)
    bad["records"][0]["service_task_id"] = "cleaning_service"
    with self.assertRaisesRegex(ValueError, "not applicable"):
        overlay_evidence_ledger(self.matrix, bad)

def test_same_relation_replaces_instead_of_duplicates(self):
    merged = merge_ledger_record(self.ledger, self.newer_record)
    self.assertEqual(1, len(merged["records"]))
```

- [ ] **Step 2: Verify RED**

Run the Task 1 test command. Expected: missing overlay functions.

- [ ] **Step 3: Implement deterministic ledger and overlay**

```python
def merge_ledger_record(ledger, record):
    by_relation = {item["relation_key"]: dict(item) for item in ledger["records"]}
    by_relation[record["relation_key"]] = dict(record)
    return {
        "schema_version": "skill_coverage_evidence_ledger_v1",
        "records": [by_relation[key] for key in sorted(by_relation)],
    }

def overlay_evidence_ledger(matrix, ledger):
    updated = copy.deepcopy(matrix)
    index = {(x["service_task_id"], x["object_id"]): x for x in updated}
    for record in ledger["records"]:
        entry = index[(record["service_task_id"], record["object_id"])]
        require_applicable(entry)
        entry["validation_state"] = "passed"
        entry["evidence"] = [ledger_evidence_reference(record)]
    return updated
```

- [ ] **Step 4: Verify GREEN**

Run the Task 1 test command. Expected: all tests pass.

---

### Task 3: 原子证据导入 CLI

**Files:**
- Create: `tools/merge_skill_coverage_evidence.py`
- Create: `tests/test_merge_skill_coverage_evidence_cli.py`

**Interfaces:**
- Consumes: `--evidence`, `--registry-dir`, `--ledger`, `--contract`, `--policy`。
- Produces: updated ledger/matrix/summary and JSON result。

- [ ] **Step 1: Write a failing end-to-end CLI test**

```python
def test_cli_writes_ledger_matrix_and_summary_atomically(self):
    exit_code = main(self.arguments)
    self.assertEqual(0, exit_code)
    self.assertEqual(1, self.load_summary()["validated_object_count"])
    passed = [x for x in self.load_matrix() if x["validation_state"] == "passed"]
    self.assertEqual("organization_storage", passed[0]["service_task_id"])
```

Add a failure test that corrupts one report hash and asserts the three target files are
byte-for-byte unchanged.

- [ ] **Step 2: Verify RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest tests.test_merge_skill_coverage_evidence_cli -v
```

Expected: import failure because the CLI does not exist.

- [ ] **Step 3: Implement staged validation and atomic writes**

```python
record = validate_hybrid_acceptance_bundle(...)
next_ledger = merge_ledger_record(current_ledger, record)
next_matrix = overlay_evidence_ledger(current_matrix, next_ledger)
next_summary = coverage_with_preserved_metadata(...)
validate_in_memory_shapes(next_ledger, next_matrix, next_summary)
atomic_write_json(ledger_path, next_ledger)
atomic_write_json(matrix_path, next_matrix)
atomic_write_json(summary_path, next_summary)
```

All validation and serialization must finish before the first target replacement.

- [ ] **Step 4: Verify GREEN**

Run Task 3 tests and Task 1 tests. Expected: all pass.

---

### Task 4: 重建注册表时重放 ledger

**Files:**
- Create: `tools/build_authoritative_skill_coverage/__init__.py`
- Create: `tools/build_authoritative_skill_coverage/__main__.py`
- Create: `tests/test_build_authoritative_skill_coverage_with_evidence.py`

**Interfaces:**
- Consumes: legacy builder output and default `data/skill_coverage/evidence_ledger.json`。
- Produces: the existing builder API `main(argv) -> int`, with evidence replay for the
  default registry or explicit `--evidence-ledger`。

- [ ] **Step 1: Write failing rebuild persistence test**

```python
def test_rebuild_replays_evidence_ledger(self):
    exit_code = main([
        "--xlsx", str(self.xlsx),
        "--output-dir", str(self.output),
        "--evidence-ledger", str(self.ledger),
        "--expected-dataset-tasks", "2",
        "--expected-relations", "2",
        "--expected-candidates", "1",
    ])
    self.assertEqual(1, self.summary()["validated_object_count"])
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest tests.test_build_authoritative_skill_coverage_with_evidence -v
```

Expected: parser rejects `--evidence-ledger`.

- [ ] **Step 3: Implement same-name package wrapper**

Load `tools/build_authoritative_skill_coverage.py` through `importlib`, preserve its public
helpers, build to a staging directory, apply ledger there, then replace the requested
output files. Custom test outputs do not use the project ledger unless
`--evidence-ledger` is supplied.

- [ ] **Step 4: Verify GREEN and legacy compatibility**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest tests.test_build_authoritative_skill_coverage_with_evidence tests.test_build_authoritative_skill_coverage -v
```

Expected: new persistence tests and existing builder tests all pass.

---

### Task 5: 导入真实 organizing::toy 证据并独立验收

**Files:**
- Create: `data/skill_coverage/evidence_ledger.json`
- Update through CLI: `data/skill_coverage/generated/task_object_matrix.json`
- Update through CLI: `data/skill_coverage/generated/coverage_summary.json`

**Interfaces:**
- Consumes: `outputs/formal_skill_dagger_r3_hybrid_v3_eval/acceptance_summary.json`。
- Produces: first audited L3 coverage record。

- [ ] **Step 1: Run all evidence-merge tests**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest tests.test_skill_coverage_evidence_merge tests.test_merge_skill_coverage_evidence_cli tests.test_build_authoritative_skill_coverage_with_evidence tests.test_build_authoritative_skill_coverage -v
```

Expected: all pass.

- [ ] **Step 2: Import the real bundle**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.merge_skill_coverage_evidence --evidence outputs\formal_skill_dagger_r3_hybrid_v3_eval\acceptance_summary.json
```

Expected: `organizing::toy -> organization_storage::toy`, validated objects `1`,
validated tasks `1`, remaining gap `119`.

- [ ] **Step 3: Run independent registry validation**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.validate_skill_coverage --registry-dir data\skill_coverage\generated
```

Expected:

```text
STRUCTURAL_VALIDATION: PASS
PROJECT_ACCEPTANCE: NOT_READY validated_objects=1 target=120 validated_tasks=1/15
```

- [ ] **Step 4: Rebuild and prove persistence**

Run the authoritative builder with the locked Excel and then rerun Step 3. Expected:
the same `1 / 1 / 119` coverage values, proving that evidence survives regeneration.

- [ ] **Step 5: Run regression suite and compile**

Run all skill-coverage, Task-2 contract, formal evidence and new ledger tests, followed by:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m compileall -q tools tests
```

Expected: zero failures and compile exit code `0`.
