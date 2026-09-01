# VLA82 Full-Simulation Object-Wise Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an auditable RoboCasa/MuJoCo acceptance system in which eight household task classes are learned from human-video Skill IR, sixty fixed objects are each trained with an independent lightweight skill adapter, and 600 held-out full-arm rollouts achieve 600/600 top-1 recognition and 600/600 operation success.

**Architecture:** A frozen shared visual backbone feeds a separately trained adapter for each VLA82 selection ID. An answer-free 60-class recognizer selects the adapter, the adapter predicts phase and operation targets, and a standard IK/safety controller drives a complete RoboCasa robot; annotation-compiled physics predicates decide success. Atomic state, DAgger repair, strict evidence contracts, and a source-preserving workbook report prevent simplified or scripted evidence from being accepted.

**Tech Stack:** Python 3.11; RoboCasa/robosuite/MuJoCo; OpenVLA/OFT visual features; PyTorch; Ultralytics YOLO; OpenCV/imageio; NumPy; pytest; Node.js and `@oai/artifact-tool` for the final workbook.

## Global Constraints

- The fixed source scope is exactly the 60 IDs in `outputs/midterm_testing_vla82/vla82_midterm_registry.json` and the eight selected task tables; do not add, remove, or substitute an item.
- Resolve operation meaning in this order: `operation.json`, `instruction.json`, ledger `真实视频标注`, then ledger `表内操作标签` as context only.
- Use a complete RoboCasa/robosuite robot, gripper, camera, scene, and physics; abstract gantries and direct writes to object or fixture terminal state are forbidden.
- Use a shared frozen visual backbone plus sixty independently trained adapter checkpoints. Each adapter needs its own dataset manifest, training metrics, checkpoint, and SHA256.
- Recognition requests may contain the image and the full 60-class vocabulary, but may not contain the expected class, selection ID, task answer, or target-conditioned instruction.
- Standard IK, collision protection, action limiting, and joint limiting are allowed. A per-selection hard-coded full trajectory is forbidden.
- Every physical predicate must come from the source annotation and cover every phase of composite and multi-object operations.
- Held-out acceptance seeds are `1000` through `1009` for each object and may never enter training. DAgger must reproduce the failure distribution with new repair seeds at or above `4000`; a failed object reruns all ten acceptance seeds after repair.
- Final gates are exactly 8/8 tasks, 60/60 objects, 600/600 top-1 recognitions, and 600/600 physical operations.
- After the same root cause occurs three times, run full-layer diagnosis and continue; do not stop the overall test.
- Use `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe` for simulation/tests and `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe` for OpenVLA feature caching/training.
- The workspace is not a Git repository. Replace commit steps with atomic file writes, versioned output directories, and a passing-test checkpoint recorded in `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/build_state.json`.

---

## File Structure

- `tools/vla82_full_sim/contracts.py`: strict 8/60/600 evidence and anti-shortcut gates.
- `tools/vla82_full_sim/annotations.py`: source annotation loading, precedence, operation-family compilation, and differences.
- `tools/vla82_full_sim/assets.py`: exact native/custom asset resolution and multi-object scene requirements.
- `tools/vla82_full_sim/predicates.py`: operation-specific physical predicate evaluators.
- `tools/vla82_full_sim/environment.py`: complete-robot RoboCasa scene factory and observable state extraction.
- `tools/vla82_full_sim/expert.py`: operation-spec-driven expert/DAgger demonstration collection.
- `tools/vla82_full_sim/adapters.py`: independent object adapter model, training, loading, and fingerprinting.
- `tools/vla82_full_sim/recognition.py`: answer-free 60-class dataset, inference, and top-1 validation.
- `tools/vla82_full_sim/task_learning.py`: eight human-video Skill IR learning and task-to-object binding.
- `tools/vla82_full_sim/rollout.py`: recognizer → adapter → IK → simulator closed-loop execution and evidence.
- `tools/vla82_full_sim/orchestrator.py`: resumable 60-object training, DAgger repair, ten-seed reruns, and three-failure diagnosis.
- `tools/vla82_full_sim/reporting.py`: manifest, contact sheet, artifact hashes, and report-data export.
- `tools/run_vla82_full_simulation.py`: one CLI with audit, collect, train, recognize, rollout, repair, and verify commands.
- `configs/vla82_full_simulation/acceptance.json`: seeds, thresholds, output root, and forbidden shortcuts.
- `configs/vla82_full_simulation/object_assets.json`: exact asset/fixture and manipulated-object definitions for all 60 IDs.
- `configs/vla82_full_simulation/operation_specs.json`: compiled 60 source-backed operation specifications.
- `.codex_artifact_work/build_vla82_full_sim_ledger.mjs`: source-preserving workbook report builder.
- `tests/test_vla82_full_sim_*.py`: focused unit/integration contracts for every module.

---

### Task 1: Compile the authoritative 60 operation specifications

**Files:**
- Create: `tools/vla82_full_sim/__init__.py`
- Create: `tools/vla82_full_sim/annotations.py`
- Create: `tools/vla82_full_sim/contracts.py`
- Create: `tools/run_vla82_full_simulation.py`
- Create: `tests/test_vla82_full_sim_annotations.py`
- Create: `tests/test_vla82_full_sim_contracts.py`
- Create at runtime: `configs/vla82_full_simulation/operation_specs.json`

**Interfaces:**
- Consumes: `load_registry(path: Path) -> dict` from `tools.vla82_acceptance.demo_evidence` and source `operation.json` / `instruction.json` files under each registry `data_path`.
- Produces: `OperationSpec`, `compile_operation_spec(selection: Mapping[str, Any]) -> OperationSpec`, `compile_all(registry_path: Path) -> list[OperationSpec]`, `validate_fixed_scope(specs) -> list[str]`, and `validate_no_shortcuts(report, bundle) -> list[str]`.

- [ ] **Step 1: Write failing annotation-precedence and fixed-scope tests**

```python
def test_operation_json_overrides_ledger_label(tmp_path):
    data = tmp_path / "obj"
    data.mkdir()
    (data / "operation.json").write_text(
        json.dumps({"instruction": "抓取—擦拭—放回"}, ensure_ascii=False), "utf-8"
    )
    selection = {
        "selection_id": "VLA82-002", "task": "室内卫生清洁", "object": "海绵",
        "operation_label": "台面→橱柜", "real_video_annotation": "仅抓取",
        "data_path": str(data),
    }
    spec = compile_operation_spec(selection)
    assert spec.source_kind == "operation_json"
    assert spec.operation_text == "抓取—擦拭—放回"
    assert spec.phases == ("grasp", "wipe", "return")


def test_fixed_scope_requires_exactly_the_registry_sixty():
    registry = json.loads(REGISTRY.read_text("utf-8"))
    specs = [compile_operation_spec(item) for item in registry["selected_objects"]]
    assert validate_fixed_scope(specs) == []
    assert "selection_count:59" in validate_fixed_scope(specs[:-1])
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tests\test_vla82_full_sim_annotations.py tests\test_vla82_full_sim_contracts.py -q
```

Expected: collection fails with `ModuleNotFoundError: tools.vla82_full_sim`.

- [ ] **Step 3: Implement source-backed operation compilation**

```python
@dataclass(frozen=True)
class OperationSpec:
    selection_id: str
    task: str
    object_name: str
    operation_text: str
    source_kind: str
    source_path: str
    phases: tuple[str, ...]
    manipulated_objects: tuple[str, ...]
    predicate_names: tuple[str, ...]
    source_sha256: str


PHRASE_PHASES = {
    "擦拭": "wipe", "刷洗": "scrub", "喷": "spray", "投放": "deposit",
    "放入": "insert", "叠放": "stack", "并排": "arrange", "旁": "arrange",
    "拉开": "pull", "拉出": "pull", "推入": "push", "关闭": "close",
    "打开": "open", "旋钮": "turn_knob", "按压": "press", "合盖": "close_lid",
    "抓取": "grasp", "放回": "return", "摆放": "place",
    "pick": "grasp", "place": "place", "close": "close", "open": "open",
    "push": "push", "pull": "pull", "turn": "turn_knob", "press": "press",
}


def compile_operation_spec(selection: Mapping[str, Any]) -> OperationSpec:
    data_path = Path(str(selection["data_path"]))
    operation = _read_instruction(data_path / "operation.json")
    instruction = _read_instruction(data_path / "instruction.json")
    candidates = (
        (operation, "operation_json", data_path / "operation.json"),
        (instruction, "instruction_json", data_path / "instruction.json"),
        (str(selection.get("real_video_annotation", "")).strip(), "ledger_real_video_annotation", REGISTRY_SOURCE),
        (str(selection.get("operation_label", "")).strip(), "ledger_operation_label", REGISTRY_SOURCE),
    )
    text, kind, source = next(item for item in candidates if item[0])
    phases = parse_ordered_phases(text)
    return OperationSpec(
        selection_id=str(selection["selection_id"]), task=str(selection["task"]),
        object_name=str(selection["object"]), operation_text=text, source_kind=kind,
        source_path=str(source.resolve()), phases=phases,
        manipulated_objects=parse_manipulated_objects(text, selection),
        predicate_names=predicates_for_phases(phases), source_sha256=sha256_file(source),
    )
```

- [ ] **Step 4: Add anti-shortcut evidence gates**

```python
FORBIDDEN_EXECUTION_KINDS = {"abstract_gantry", "direct_state_write", "scripted_full_trajectory"}


def validate_no_shortcuts(report: Mapping[str, Any], bundle: Path) -> list[str]:
    errors = []
    if report.get("complete_robot_model") is not True:
        errors.append("complete_robot_missing")
    if report.get("execution_kind") in FORBIDDEN_EXECUTION_KINDS:
        errors.append("forbidden_execution_kind")
    if report.get("terminal_state_directly_written") is not False:
        errors.append("terminal_state_write_not_disproved")
    if report.get("adapter_checkpoint_sha256") is None:
        errors.append("adapter_fingerprint_missing")
    return errors
```

- [ ] **Step 5: Implement the first CLI command**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compile_specs = commands.add_parser("compile-specs")
    compile_specs.set_defaults(handler=command_compile_specs)
    return parser


def command_compile_specs(args: argparse.Namespace) -> int:
    specs = compile_all(REGISTRY)
    errors = validate_fixed_scope(specs)
    atomic_write_json(OPERATION_SPECS, [asdict(spec) for spec in specs])
    print(f"compiled={len(specs)} errors={len(errors)}")
    return int(bool(errors))
```

- [ ] **Step 6: Run GREEN tests and compile all 60 specs**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tests\test_vla82_full_sim_annotations.py tests\test_vla82_full_sim_contracts.py -q
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py compile-specs
```

Expected: tests pass and command prints `compiled=60 errors=0`.

- [ ] **Step 7: Record the atomic checkpoint**

Write `build_state.json` with task `operation_specs`, status `PASS`, test command, output hash, and timestamp. Verify `operation_specs.json` contains exactly 60 unique IDs.

---

### Task 2: Resolve exact assets and complete-robot scenes

**Files:**
- Create: `tools/vla82_full_sim/assets.py`
- Create: `tools/vla82_full_sim/environment.py`
- Create: `configs/vla82_full_simulation/object_assets.json`
- Modify: `tools/run_vla82_full_simulation.py`
- Create: `tests/test_vla82_full_sim_assets.py`
- Create: `tests/test_vla82_full_sim_environment.py`

**Interfaces:**
- Consumes: `OperationSpec` and existing `outputs/midterm_testing_vla82/simulator_mapping_plan.json`.
- Produces: `AssetSpec`, `SceneRequest`, `resolve_asset(spec, mapping) -> AssetSpec`, `build_scene_request(spec, asset) -> SceneRequest`, and `make_environment(request, seed) -> gym.Env`.

- [ ] **Step 1: Write failing exact-asset and full-robot tests**

```python
def test_functional_proxy_can_never_be_final_exact_asset():
    mapping = {"selection_id": "VLA82-050", "mapping_mode": "functional_proxy", "object_group": "can"}
    spec = fixture_operation_spec("VLA82-050", "订书机", ("place",))
    asset = resolve_asset(spec, mapping)
    assert asset.exact_class is True
    assert asset.kind != "functional_proxy"
    assert asset.semantic_class == "订书机"


def test_scene_request_contains_complete_robot_and_all_annotated_objects():
    spec = fixture_operation_spec(
        "VLA82-016", "书籍", ("place",), manipulated=("书籍", "凉鞋", "凉鞋")
    )
    request = build_scene_request(spec, fixture_asset("书籍"))
    assert request.robot_name == "PandaOmron"
    assert request.camera_names == ("robot0_agentview_left", "robot0_eye_in_hand")
    assert request.manipulated_objects == ("书籍", "凉鞋", "凉鞋")
```

- [ ] **Step 2: Run RED tests**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tests\test_vla82_full_sim_assets.py tests\test_vla82_full_sim_environment.py -q
```

Expected: import fails because `assets.py` and `environment.py` do not exist.

- [ ] **Step 3: Implement exact native/custom asset resolution**

```python
@dataclass(frozen=True)
class AssetSpec:
    selection_id: str
    semantic_class: str
    kind: Literal["robocasa_native", "custom_same_class", "fixture_part"]
    asset_path_or_group: str
    exact_class: bool
    collision_validated: bool
    visible_validated: bool
    affordances: tuple[str, ...]


def resolve_asset(spec: OperationSpec, mapping: Mapping[str, Any]) -> AssetSpec:
    native = resolve_native_exact_group(spec.object_name, mapping)
    if native is not None:
        return validate_native_asset(spec, native)
    fixture = resolve_exact_fixture_part(spec.object_name, mapping)
    if fixture is not None:
        return validate_fixture_part(spec, fixture)
    model = build_source_textured_same_class_asset(spec)
    return validate_custom_asset(spec, model)
```

Custom assets must use the exact class, source-video-derived appearance, class-specific dimensions and collision bodies. `exact_class=False` is a hard failure.

- [ ] **Step 4: Implement operation-aware scene requests**

```python
@dataclass(frozen=True)
class SceneRequest:
    selection_id: str
    task_class: str
    robot_name: str
    camera_names: tuple[str, ...]
    primary_asset: AssetSpec
    manipulated_objects: tuple[str, ...]
    fixture_requirements: tuple[str, ...]
    forbidden_direct_state_writes: bool = True


def make_environment(request: SceneRequest, seed: int):
    env = build_registered_or_custom_robocasa_task(request, seed)
    raw = env.unwrapped.env
    assert raw.robots and len(raw.robots[0].robot_joints) >= 7
    assert set(request.camera_names).issubset(raw.camera_names)
    return env
```

`environment.py` also defines the state contract consumed by predicates:

```python
@dataclass(frozen=True)
class PhysicsSnapshot:
    step: int
    robot_qpos: np.ndarray
    gripper_qpos: np.ndarray
    body_poses: dict[str, np.ndarray]
    joint_positions: dict[str, float]
    contacts: tuple[tuple[str, str, float], ...]
    dirt_fraction: float
    spray_coverage: float
    dispensed_amount: float
```

- [ ] **Step 5: Register `audit-assets` and run audits for all 60**

Add `audit-assets` to `build_parser()` with `--render`; its handler loads all compiled specs, resolves each asset, resets each complete-robot scene, renders both cameras, and exits nonzero unless exact/loadable/visible/complete-robot counts all equal 60.

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py audit-assets --render
```

Expected: `exact=60 loadable=60 visible=60 complete_robot=60 proxies=0` and sixty preview images.

- [ ] **Step 6: Run GREEN tests and checkpoint**

Run the two test files again. Record `asset_scene_audit=PASS` and hashes for `object_assets.json` plus the 60 previews.

---

### Task 3: Implement annotation-specific physical predicates

**Files:**
- Create: `tools/vla82_full_sim/predicates.py`
- Create: `tests/test_vla82_full_sim_predicates.py`

**Interfaces:**
- Consumes: `OperationSpec` plus `PhysicsSnapshot` captured by `environment.py`.
- Produces: `PredicateResult`, `evaluate_operation(spec, initial, history, final) -> PredicateResult`, and individual evaluators registered under phase names.

- [ ] **Step 1: Write failing tests for every physical operation family**

```python
@pytest.mark.parametrize(
    ("phase", "snapshot", "expected_metric"),
    [
        ("deposit", deposit_inside_bin_snapshot(), "inside_target_volume"),
        ("wipe", wipe_contact_coverage_snapshot(), "dirt_reduction"),
        ("scrub", scrub_contact_coverage_snapshot(), "dirt_reduction"),
        ("spray", spray_trigger_coverage_snapshot(), "spray_coverage"),
        ("insert", inserted_and_released_snapshot(), "containment"),
        ("stack", stable_stack_snapshot(), "stable_support"),
        ("arrange", keyboard_mouse_relation_snapshot(), "spatial_relation"),
        ("open", opened_hinge_snapshot(), "joint_target"),
        ("close", closed_hinge_snapshot(), "joint_target"),
        ("pull", pulled_slider_snapshot(), "joint_target"),
        ("push", pushed_slider_snapshot(), "joint_target"),
        ("turn_knob", rotated_knob_snapshot(), "joint_target"),
        ("press", pressed_dispenser_snapshot(), "dispensed_amount"),
        ("close_lid", closed_laptop_snapshot(), "joint_target"),
    ],
)
def test_each_operation_family_requires_physical_contact_and_metric(phase, snapshot, expected_metric):
    result = evaluate_phase(phase, snapshot.initial, snapshot.history, snapshot.final)
    assert result.success is True
    assert expected_metric in result.metrics
    assert result.contact_verified is True
```

Add negative tests that identical terminal positions without contact fail, and composite operations fail when one phase is missing.

- [ ] **Step 2: Run RED tests**

Expected: import fails for `tools.vla82_full_sim.predicates`.

- [ ] **Step 3: Implement the predicate registry**

```python
@dataclass(frozen=True)
class PredicateResult:
    success: bool
    phase_results: tuple[dict[str, Any], ...]
    metrics: dict[str, float | bool | str]
    contact_verified: bool
    errors: tuple[str, ...]


PREDICATES = {
    "grasp": evaluate_grasp, "place": evaluate_place, "return": evaluate_place,
    "deposit": evaluate_deposit, "wipe": evaluate_wipe, "scrub": evaluate_wipe,
    "spray": evaluate_spray, "insert": evaluate_insert, "stack": evaluate_stack,
    "arrange": evaluate_arrangement, "open": evaluate_hinge,
    "close": evaluate_hinge, "close_lid": evaluate_hinge,
    "pull": evaluate_slider, "push": evaluate_slider,
    "turn_knob": evaluate_knob, "press": evaluate_press,
}


def evaluate_operation(spec, initial, history, final):
    results = tuple(PREDICATES[phase](spec, initial, history, final) for phase in spec.phases)
    return combine_ordered_phase_results(spec, results, history)
```

- [ ] **Step 4: Enforce ordered composite and multi-object completion**

`combine_ordered_phase_results` must compare observed phase-transition timestamps with `spec.phases`, require every `spec.manipulated_objects` entry to have its own result, and reject final-state-only success.

- [ ] **Step 5: Run GREEN tests and checkpoint**

Expected: all predicate tests pass, including negative no-contact and missing-phase cases.

---

### Task 4: Collect operation-spec-driven expert demonstrations

**Files:**
- Create: `tools/vla82_full_sim/expert.py`
- Modify: `tools/run_vla82_full_simulation.py`
- Create: `tests/test_vla82_full_sim_expert.py`
- Reuse: `tools/collect_vla82_targeted_expert.py`, `tools/pick_place_oracle/`, and RoboCasa public actions only as training-label sources.

**Interfaces:**
- Consumes: `SceneRequest`, `OperationSpec`, and predicate evaluators.
- Produces: `collect_expert_episode(spec, scene, seed, output_dir) -> EpisodeReport` and `.npz` episodes containing `primary`, `wrist`, `proprio`, `actions`, `phases`, `contacts`, and `success`.

- [ ] **Step 1: Write a failing demonstration-schema test**

```python
def test_expert_episode_contains_visual_action_contact_and_phase_labels(tmp_path):
    report = collect_expert_episode(fixture_spec(), fixture_scene(), 82002, tmp_path)
    episode = np.load(report.episode_path, allow_pickle=False)
    assert episode["primary"].shape[0] == episode["actions"].shape[0]
    assert episode["wrist"].shape[0] == episode["actions"].shape[0]
    assert episode["proprio"].shape[1] == 8
    assert set(np.unique(episode["phases"])) >= {"grasp", "wipe", "return"}
    assert episode["contacts"].any()
    assert report.predicate_success is True
```

- [ ] **Step 2: Run RED and confirm missing implementation**

- [ ] **Step 3: Implement primitive experts selected by compiled phase**

```python
EXPERTS = {
    "grasp": PickPlaceExpert,
    "place": PickPlaceExpert,
    "return": PickPlaceExpert,
    "deposit": DepositExpert,
    "wipe": SurfaceCoverageExpert,
    "scrub": SurfaceCoverageExpert,
    "spray": SprayExpert,
    "insert": InsertionExpert,
    "stack": StackExpert,
    "arrange": RelativePlacementExpert,
    "open": ArticulationExpert,
    "close": ArticulationExpert,
    "close_lid": ArticulationExpert,
    "pull": ArticulationExpert,
    "push": ArticulationExpert,
    "turn_knob": KnobExpert,
    "press": PressExpert,
}


def collect_expert_episode(spec, scene, seed, output_dir):
    env = make_environment(scene, seed)
    controller = CompositeExpert([EXPERTS[phase](spec, env) for phase in spec.phases])
    return record_successful_episode(env, controller, spec, output_dir)
```

Experts may read privileged state only while producing training labels. Their rollouts must be marked `training_expert=True` and can never be accepted as learned-policy rollouts.

- [ ] **Step 4: Register `collect-demos` and collect six initial demonstrations per object**

Add `collect-demos` with `--episodes-per-object` and `--resume`. Resume skips only episodes whose NPZ schema, predicate result, and manifest hash validate.

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py collect-demos --episodes-per-object 6 --resume
```

Expected: 360 successful episodes, 60 manifests, no episode with false source predicate.

- [ ] **Step 5: Run GREEN tests and checkpoint dataset hashes**

Record each object's six episode hashes and the global manifest hash. A missing or unsuccessful expert episode blocks training for that object.

---

### Task 5: Train sixty independent skill adapters

**Files:**
- Create: `tools/vla82_full_sim/adapters.py`
- Create: `tools/cache_vla82_full_sim_features.py`
- Create: `tools/train_vla82_object_adapters.py`
- Create: `tests/test_vla82_full_sim_adapters.py`

**Interfaces:**
- Consumes: shared OpenVLA/OFT image tokens, proprioception, phase labels, operation targets, and expert actions.
- Produces: `ObjectSkillAdapter`, `train_adapter(selection_id, dataset, output_dir) -> TrainingReport`, `load_adapter(path)`, and independent `adapter.pt`/`metrics.json`/`adapter.sha256` files.

- [ ] **Step 1: Write failing model and independence tests**

```python
def test_adapter_predicts_phase_target_and_action():
    model = ObjectSkillAdapter(feature_dim=32, proprio_dim=8, phase_count=5)
    output = model(torch.zeros(2, 32), torch.zeros(2, 8))
    assert output.phase_logits.shape == (2, 5)
    assert output.target_delta.shape == (2, 7)


def test_two_object_adapters_have_independent_checkpoints(tmp_path):
    a = train_adapter("VLA82-002", tiny_dataset(seed=2), tmp_path / "a")
    b = train_adapter("VLA82-003", tiny_dataset(seed=3), tmp_path / "b")
    assert a.checkpoint_sha256 != b.checkpoint_sha256
    assert a.training_selection_id == "VLA82-002"
    assert b.training_selection_id == "VLA82-003"
```

- [ ] **Step 2: Run RED tests**

- [ ] **Step 3: Implement the lightweight adapter and loss**

```python
class ObjectSkillAdapter(nn.Module):
    def __init__(self, feature_dim: int, proprio_dim: int, phase_count: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + proprio_dim, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(),
        )
        self.phase_head = nn.Linear(256, phase_count)
        self.target_head = nn.Linear(256, 7)

    def forward(self, visual, proprio):
        hidden = self.trunk(torch.cat((visual, proprio), dim=-1))
        return AdapterOutput(self.phase_head(hidden), torch.tanh(self.target_head(hidden)))


def adapter_loss(output, phase, target):
    return F.cross_entropy(output.phase_logits, phase) + F.smooth_l1_loss(output.target_delta, target)
```

The adjacent result types are exact:

```python
@dataclass(frozen=True)
class AdapterOutput:
    phase_logits: torch.Tensor
    target_delta: torch.Tensor


@dataclass(frozen=True)
class TrainingReport:
    training_selection_id: str
    train_samples: int
    validation_samples: int
    best_validation_loss: float
    checkpoint_path: str
    checkpoint_sha256: str
    dataset_manifest_sha256: str
```

- [ ] **Step 4: Cache shared visual features once**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe tools\cache_vla82_full_sim_features.py --input datasets\vla82_full_sim_expert --output datasets\vla82_full_sim_features --resume
```

Expected: every episode frame has a finite feature vector and the cache records the shared model hash.

- [ ] **Step 5: Train all sixty adapters independently**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe tools\train_vla82_object_adapters.py --features datasets\vla82_full_sim_features --output models\vla82_full_sim_object_adapters --selection all --resume
```

Expected: sixty checkpoint directories, sixty distinct training-selection IDs, finite validation metrics, and no copied checkpoint hash without an explicit identical-data failure.

- [ ] **Step 6: Run GREEN tests and checkpoint**

Record the shared-backbone hash plus a table of 60 adapter hashes and training metrics.

---

### Task 6: Train and enforce answer-free 60-class visual recognition

**Files:**
- Create: `tools/vla82_full_sim/recognition.py`
- Create: `tools/collect_vla82_60class_detection_dataset.py`
- Create: `tools/train_vla82_60class_detector.py`
- Create: `tests/test_vla82_full_sim_recognition.py`

**Interfaces:**
- Consumes: exact assets, segmentation masks used only for training labels, and `models/vision/yolov8s-worldv2.pt` as initialization.
- Produces: `RecognitionRequest`, `RecognitionResult`, `validate_answer_free_request`, dataset splits, detector checkpoint, and `recognize(image, vocabulary) -> RecognitionResult`.

- [ ] **Step 1: Write failing anti-answer-injection and top-1 tests**

```python
def test_recognition_request_rejects_expected_answer_fields():
    request = {"image": "frame.png", "vocabulary": ALL_60, "expected": "海绵"}
    assert "forbidden_field:expected" in validate_answer_free_request(request)


def test_recognition_prompt_cannot_contain_selection_or_expected_class():
    request = RecognitionRequest(image="frame.png", vocabulary=ALL_60, prompt="VLA82-002 海绵")
    errors = validate_answer_free_request(request, expected_class="海绵")
    assert "expected_class_leak" in errors
    assert "selection_id_leak" in errors


def test_top1_is_model_prediction_not_expected_label():
    result = classify_top1([("海绵", 0.2), ("清洁刷", 0.8)])
    assert result.predicted_class == "清洁刷"
```

- [ ] **Step 2: Run RED tests**

- [ ] **Step 3: Implement the request contract and inference wrapper**

```python
@dataclass(frozen=True)
class RecognitionRequest:
    image: str
    vocabulary: tuple[str, ...]
    prompt: str = "识别图中主要可操作物体"


def validate_answer_free_request(request, expected_class=None):
    payload = asdict(request) if is_dataclass(request) else dict(request)
    errors = [f"forbidden_field:{key}" for key in ("expected", "selection_id", "target") if key in payload]
    prompt = str(payload.get("prompt", ""))
    if expected_class and expected_class in prompt:
        errors.append("expected_class_leak")
    if re.search(r"VLA82-\d{3}", prompt):
        errors.append("selection_id_leak")
    return errors
```

```python
@dataclass(frozen=True)
class RecognitionResult:
    predicted_class: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    vocabulary_sha256: str
    expected_class_was_not_supplied: bool
    request_sha256: str
```

- [ ] **Step 4: Collect train/validation/acceptance-separated renders**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\collect_vla82_60class_detection_dataset.py --train-seeds 2000:2011 --validation-seeds 3000:3003 --acceptance-seeds 1000:1009
```

The collector may use simulator segmentation to generate training labels, but acceptance inference receives RGB only. Split overlap is a hard failure.

- [ ] **Step 5: Train and validate the 60-class detector**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\train_vla82_60class_detector.py --data datasets\vla82_60class_detection\dataset.yaml --initial models\vision\yolov8s-worldv2.pt --output models\vla82_60class_detector
```

Expected before acceptance: validation top-1 60/60 classes and zero class-index mismatch. Do not inspect acceptance labels during model selection.

- [ ] **Step 6: Run GREEN tests and checkpoint**

Record detector weight hash, class-order hash, train/validation seed hashes, and proof that acceptance seeds were excluded.

---

### Task 7: Build eight human-video Skill IR task-learning bundles

**Files:**
- Create: `tools/vla82_full_sim/task_learning.py`
- Modify: `tools/run_vla82_full_simulation.py`
- Create: `tests/test_vla82_full_sim_task_learning.py`
- Reuse: `configs/vla82_acceptance/human_demo_8.json`, `tools/vla82_acceptance/demo_evidence.py`, and motion/contact extractors under `tools/human_video_skill_ir/`.

**Interfaces:**
- Consumes: eight public human demonstration bindings and compiled operation specs.
- Produces: `learn_task_skill_ir(binding, representative_spec) -> TaskSkillIR` and eight bundles with visual observations, contact/motion evidence, ordered phases, and representative object adapter binding.

- [ ] **Step 1: Write failing eight-category and evidence tests**

```python
def test_eight_task_irs_cover_unique_source_tables_and_have_visual_evidence():
    results = learn_all_task_irs(BINDINGS, OPERATION_SPECS)
    assert len(results) == 8
    assert len({item.source_table for item in results}) == 8
    assert all(item.motion_evidence_frames for item in results)
    assert all(item.ordered_phases for item in results)
    assert all(item.representative_adapter_sha256 for item in results)
```

- [ ] **Step 2: Run RED tests**

- [ ] **Step 3: Implement task-level Skill IR learning**

```python
@dataclass(frozen=True)
class TaskSkillIR:
    source_table: str
    task: str
    source_video: str
    motion_evidence_frames: tuple[int, ...]
    contact_events: tuple[dict[str, Any], ...]
    ordered_phases: tuple[str, ...]
    representative_selection_id: str
    representative_adapter_sha256: str


def learn_task_skill_ir(binding, representative_spec):
    evidence = extract_human_video_evidence(binding)
    phases = align_evidence_to_operation(evidence, representative_spec)
    adapter_hash = sha256_file(
        ADAPTER_ROOT / representative_spec.selection_id / "adapter.pt"
    )
    return TaskSkillIR(
        source_table=str(binding["source_table"]),
        task=str(binding["task"]),
        source_video=str(Path(binding["video"]).resolve()),
        motion_evidence_frames=tuple(evidence.motion_frame_indices),
        contact_events=tuple(evidence.contact_events),
        ordered_phases=tuple(phases),
        representative_selection_id=representative_spec.selection_id,
        representative_adapter_sha256=adapter_hash,
    )
```

- [ ] **Step 4: Register `learn-tasks` and materialize eight task bundles**

Add `learn-tasks` to the CLI; it loads all eight bindings, validates unique source tables, learns each `TaskSkillIR`, and atomically writes the eight bundles.

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py learn-tasks
```

Expected: `tasks=8 learned=8 missing_visual_evidence=0`.

- [ ] **Step 5: Run GREEN tests and checkpoint**

Record hashes for eight source videos, eight Skill IR files, and eight representative adapter bindings.

---

### Task 8: Execute learned full-arm rollouts without terminal-state scripting

**Files:**
- Create: `tools/vla82_full_sim/rollout.py`
- Modify: `tools/run_vla82_full_simulation.py`
- Create: `tests/test_vla82_full_sim_rollout.py`

**Interfaces:**
- Consumes: `OperationSpec`, `SceneRequest`, recognizer, object adapter, and physical predicates.
- Produces: `run_learned_rollout(spec, scene, seed, artifacts) -> RolloutReport` with video, trajectory, recognition result, adapter fingerprint, robot-joint movement, contacts, and predicate results.

- [ ] **Step 1: Write failing rollout integrity tests**

```python
def test_rollout_uses_recognized_adapter_and_complete_robot(tmp_path):
    report = run_learned_rollout(fixture_spec(), fixture_scene(), 1000, tmp_path)
    assert report.complete_robot_model is True
    assert report.recognition.expected_class_was_not_supplied is True
    assert report.recognition.predicted_class == report.object_name
    assert report.adapter_selection_source == "recognizer_top1"
    assert report.robot_joint_motion_norm > 0.1
    assert report.gripper_motion_norm > 0.01
    assert report.terminal_state_directly_written is False
    assert report.predicate.success is True


def test_rollout_rejects_expert_or_scripted_actions(tmp_path):
    report = fixture_report(execution_kind="training_expert")
    assert "learned_policy_not_used" in validate_acceptance_rollout(report, tmp_path)
```

- [ ] **Step 2: Run RED tests**

- [ ] **Step 3: Implement the learned control loop**

```python
def run_learned_rollout(spec, scene, seed, artifacts):
    env = make_environment(scene, seed)
    obs, _ = env.reset(seed=seed)
    recognition = recognizer.recognize(rgb(obs), ALL_60_CLASSES)
    adapter = adapter_store.load_for_class(recognition.predicted_class)
    recorder = PhysicsRecorder(env, spec)
    for decision in range(MAX_DECISIONS):
        visual = shared_backbone.encode(rgb(obs))
        output = adapter(visual, proprio(obs))
        action = ik_safety_controller.command(output.phase_logits, output.target_delta, obs)
        obs, _, terminated, truncated, info = env.step(action)
        recorder.append(obs, action, output, info)
        if recorder.operation_complete or terminated or truncated:
            break
    return recorder.finalize(recognition, adapter)
```

`RolloutReport` is serialized with this fixed acceptance surface:

```python
@dataclass(frozen=True)
class RolloutReport:
    selection_id: str
    object_name: str
    seed: int
    status: str
    complete_robot_model: bool
    recognition: RecognitionResult
    adapter_selection_source: str
    adapter_checkpoint_sha256: str
    robot_joint_motion_norm: float
    gripper_motion_norm: float
    terminal_state_directly_written: bool
    predicate: PredicateResult
    evidence: dict[str, str]
```

- [ ] **Step 4: Add anti-direct-write instrumentation**

Record qpos changes before and after each `env.step`. Any object or fixture terminal-state change outside `env.step`, reset, or MuJoCo integration marks `terminal_state_directly_written=True` and fails the trial.

- [ ] **Step 5: Write evidence atomically**

Each seed directory must contain `recognition.json`, `trajectory.npz`, `trajectory.json`, `rollout.mp4`, `first_frame.png`, `contact_frame.png`, `last_frame.png`, `predicate.json`, and `report.json`. Decode the video and validate frame differences before writing PASS.

- [ ] **Step 6: Register `rollout`, run GREEN tests, and execute one real pilot**

Add `rollout` with required `--selection-id` and `--seed`. The handler must exit nonzero unless recognition, adapter invocation, complete-robot motion, physical predicate, and evidence validation all pass.

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tests\test_vla82_full_sim_rollout.py -q
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py rollout --selection-id VLA82-002 --seed 1000
```

Expected: a complete-arm video, correct image-only recognition, learned adapter fingerprint, physical contact, and PASS predicate.

---

### Task 9: Add resumable DAgger repair and ten-seed rerun semantics

**Files:**
- Create: `tools/vla82_full_sim/orchestrator.py`
- Modify: `tools/run_vla82_full_simulation.py`
- Create: `tests/test_vla82_full_sim_orchestrator.py`
- Create: `configs/vla82_full_simulation/acceptance.json`

**Interfaces:**
- Consumes: demonstration collector, trainer, recognizer, rollout runner, and atomic state.
- Produces: `run_object_cycle(selection_id)`, `rerun_required_seeds`, `retry_decision`, `diagnose_layers`, and resumable CLI state.

- [ ] **Step 1: Write failing rerun and three-failure tests**

```python
def test_any_failed_seed_requires_all_ten_to_rerun():
    prior = {seed: "PASS" for seed in range(1000, 1010)}
    prior[1004] = "FAIL"
    assert rerun_required_seeds(prior) == list(range(1000, 1010))


def test_three_same_root_causes_trigger_full_diagnosis_and_continue():
    attempts = [{"root_cause": "recognition_wrong"}] * 3
    decision = retry_decision(attempts)
    assert decision.action == "FULL_DIAGNOSIS_CONTINUE"
    assert decision.stop_entire_test is False
    assert set(decision.layers) == {
        "video", "annotation", "asset", "recognition", "adapter", "ik",
        "control", "contact", "predicate", "render", "evidence",
    }
```

- [ ] **Step 2: Run RED tests**

- [ ] **Step 3: Implement atomic state and DAgger cycles**

```python
ACCEPTANCE_SEEDS = tuple(range(1000, 1010))


def run_object_cycle(selection_id, services, state_path):
    while True:
        results = [services.rollout(selection_id, seed) for seed in ACCEPTANCE_SEEDS]
        if all(item.status == "PASS" for item in results):
            return record_object_pass(selection_id, results, state_path)
        failures = [item for item in results if item.status != "PASS"]
        diagnosis = diagnose_failures(failures)
        repair_seeds = allocate_repair_seeds(selection_id, failures, minimum=4000)
        services.collect_dagger(selection_id, failures, repair_seeds)
        services.retrain_adapter(selection_id)
        if diagnosis.recognition_related:
            services.augment_and_retrain_recognizer(failures)
```

The loop never converts a failure into PASS without a new learned rollout. Seeds 1000–1009 are never written to a training manifest; DAgger receives only newly allocated repair seeds at or above 4000. It preserves every attempt under `attempts/round-*`.

- [ ] **Step 4: Implement CLI and resume behavior**

The CLI subcommands must be `compile-specs`, `audit-assets`, `collect-demos`, `train-adapters`, `train-recognizer`, `learn-tasks`, `rollout`, `run-object`, `run-all`, `repair`, and `verify`. `--resume` skips only objects with a valid final 10/10 evidence contract.

- [ ] **Step 5: Run GREEN tests and checkpoint**

Expected: rerun semantics, diagnosis layers, state atomicity, and resume behavior pass.

---

### Task 10: Build strict 8/60/600 reporting and update the ledger

**Files:**
- Create: `tools/vla82_full_sim/reporting.py`
- Create: `tests/test_vla82_full_sim_reporting.py`
- Create: `.codex_artifact_work/build_vla82_full_sim_ledger.mjs`
- Create at runtime: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/manifest.json`
- Create at runtime: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/acceptance_matrix.xlsx`

**Interfaces:**
- Consumes: eight task reports, sixty object reports, six hundred seed reports, model hashes, evidence paths, and the original ledger.
- Produces: strict aggregate manifest, contact sheet, evidence inventory, and a source-preserving successor workbook.

- [ ] **Step 1: Write failing strict-count reporting tests**

```python
def test_manifest_pass_requires_exact_8_60_600():
    manifest = build_manifest(task_passes(8), object_passes(60, seeds=10))
    assert manifest["overall_status"] == "PASS"
    assert manifest["task_pass"] == 8
    assert manifest["object_pass"] == 60
    assert manifest["recognition_pass"] == 600
    assert manifest["operation_pass"] == 600


def test_one_missing_video_or_seed_forces_fail(tmp_path):
    reports = object_passes(60, seeds=10, root=tmp_path)
    (tmp_path / "VLA82-060/seed-1009/rollout.mp4").unlink()
    manifest = build_manifest(task_passes(8), reports)
    assert manifest["overall_status"] == "FAIL"
    assert "missing_evidence:VLA82-060:1009:rollout.mp4" in manifest["errors"]
```

- [ ] **Step 2: Run RED tests**

- [ ] **Step 3: Implement strict manifest and contact sheets**

`build_manifest` must verify exact IDs, exact seeds, top-1 correctness, operation predicates, adapter fingerprints, complete-robot motion, anti-shortcut fields, video decode, image differences, and hashes. Draw one 60-row contact sheet containing first/contact/last frames and status.

- [ ] **Step 4: Build the workbook with `@oai/artifact-tool`**

The builder must import the original ledger, preserve its existing sheets and formatting, and add versioned sheets:

- `完整仿真总览`: formula-driven 8/8, 60/60, 600/600 cards;
- `逐物体学习`: one row per selection ID with annotation, dataset hash, adapter hash, training result, 10/10 result, representative video;
- `600次物理闭环`: one row per object-seed with top-1 class, adapter, predicate, video, and PASS/FAIL;
- `物理操作判据`: every compiled phase and threshold;
- `证据清单`: artifact path, size, and SHA256.

The successor workbook path is `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/acceptance_matrix.xlsx`; do not overwrite the source ledger.

- [ ] **Step 5: Verify formulas and visual layout**

Use `workbook.inspect` on the summary and representative ranges, scan `#REF!|#DIV/0!|#VALUE!|#NAME?|#N/A`, render all added sheets, visually inspect them, and export one `.xlsx` only.

- [ ] **Step 6: Run GREEN tests and checkpoint**

Expected: reporting tests pass; manifest and workbook agree exactly on 8, 60, 600, and overall status.

---

### Task 11: Run an operation-family pilot gate before the full batch

**Files:**
- Modify only if pilot defects require tested fixes: modules created in Tasks 1–10.
- Runtime output: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/pilot/`

**Interfaces:**
- Consumes the complete pipeline.
- Produces one learned rollout for every distinct operation family before spending time on all 600 trials.

- [ ] **Step 1: Select the fixed pilot IDs from real annotations**

Use:

```text
VLA82-001 deposit
VLA82-002 wipe/return
VLA82-004 spray/place
VLA82-007 turn_knob
VLA82-009 push
VLA82-010 pull
VLA82-016 composite multi-object place
VLA82-024 multi-object insert
VLA82-027 close slider
VLA82-028 close hinge
VLA82-048 stack
VLA82-049 place/close_lid
VLA82-053 relative arrange
VLA82-060 press/place
```

- [ ] **Step 2: Execute one held-out seed for each pilot**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py run-object --selection-id VLA82-001,VLA82-002,VLA82-004,VLA82-007,VLA82-009,VLA82-010,VLA82-016,VLA82-024,VLA82-027,VLA82-028,VLA82-048,VLA82-049,VLA82-053,VLA82-060 --seeds 1000
```

- [ ] **Step 3: Diagnose failures with RED regression tests before fixes**

For every pilot failure, add the smallest test reproducing its root cause to the responsible module test, observe RED, patch production code, observe GREEN, and rerun the failed pilot plus any other IDs sharing the root cause.

- [ ] **Step 4: Pass the pilot gate**

Expected: 14/14 top-1 recognition, 14/14 learned physical execution, all operation families represented, all videos show the complete robot.

- [ ] **Step 5: Record the pilot checkpoint**

Write the pilot manifest and do not start the 600 batch until it is PASS.

---

### Task 12: Execute all sixty learning cycles and the final 600-trial acceptance

**Files:**
- Runtime outputs only under `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/`.
- Modify production files only through a failing regression test as required by TDD.

**Interfaces:**
- Consumes all completed components and pilot-gated models.
- Produces the final 8/8, 60/60, 600/600 evidence package.

- [ ] **Step 1: Run all object cycles with resume enabled**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py run-all --seeds 1000:1009 --resume
```

Expected behavior: each object runs all ten seeds; failed objects collect DAgger data, retrain, and rerun all ten. Same-root failure three triggers full diagnosis and continues.

- [ ] **Step 2: Generate reports only after all object state records are PASS**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py verify
```

Expected: `tasks=8/8 objects=60/60 recognition=600/600 operation=600/600 missing_evidence=0 overall=PASS`.

- [ ] **Step 3: Run the complete automated test suite**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tests\test_vla82_full_sim_annotations.py tests\test_vla82_full_sim_contracts.py tests\test_vla82_full_sim_assets.py tests\test_vla82_full_sim_environment.py tests\test_vla82_full_sim_predicates.py tests\test_vla82_full_sim_expert.py tests\test_vla82_full_sim_adapters.py tests\test_vla82_full_sim_recognition.py tests\test_vla82_full_sim_task_learning.py tests\test_vla82_full_sim_rollout.py tests\test_vla82_full_sim_orchestrator.py tests\test_vla82_full_sim_reporting.py -q
```

Expected: all tests pass with no warnings or errors.

- [ ] **Step 4: Independently reconcile the final evidence counts**

Verify exactly:

```text
8 task Skill IR bundles
60 object adapter checkpoints
60 object training reports
600 recognition reports
600 trajectory reports
600 decodable rollout videos
600 first frames
600 contact frames
600 last frames
60 representative high-quality videos
1 strict manifest
1 source-preserving acceptance workbook
```

- [ ] **Step 5: Visually inspect evidence before declaring completion**

Open the 60-object contact sheet, the 8-task contact sheet, at least one representative video from each operation family, and all added workbook sheets. Any abstract gantry, missing robot, wrong object, skipped phase, clipped sheet, or inconsistent count forces FAIL and repair.

- [ ] **Step 6: Freeze the final acceptance package**

Write `reproduction_commands.txt`, a recursive artifact manifest with size/SHA256, environment package versions, GPU identity, shared-backbone hash, detector hash, sixty adapter hashes, and the final timestamp. Mark the root `manifest.json` PASS only after every preceding check succeeds.

---

## Execution Order and Fastest Safe Path

1. Tasks 1–3 establish source truth and physics predicates.
2. Tasks 4–6 create demonstrations, adapters, and independent recognition.
3. Tasks 7–10 connect task learning, full-arm execution, repair, and reporting.
4. Task 11 gates every operation family with fourteen pilots.
5. Task 12 runs the full 600 only after the pilot gate, avoiding expensive batch-wide debugging.

Estimated critical path on the available RTX 3090 and local RoboCasa environment is 4–9 working days, dominated by exact asset/scene repair, expert collection, DAgger rounds, and 600 full-arm videos. The plan prioritizes shared feature caching, lightweight adapters, pilot gating, and resumability; it does not replace required learning or physical evidence with shortcuts.
