# VLA82 Material Friction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic material-aware MuJoCo friction to VLA82 objects and use it to run the six simplest outstanding object operations without weakening strict PASS criteria.

**Architecture:** A new pure `friction.py` module owns semantic material profiles and applies them only to selected collision geoms. Asset MJCF generation and live RoboCasa models consume the same profile; the collector records before/after values in every episode diagnostic. Existing expert controllers and predicates remain unchanged except for targeted root-cause fixes proven necessary by failed physical traces.

**Tech Stack:** Python 3.11, NumPy, MuJoCo/RoboSuite/RoboCasa, pytest, OpenCV, JSON/NPZ evidence.

## Global Constraints

- Never write object `qpos`, `qvel`, body pose, fixture joint state, or terminal state to obtain success.
- Never lower or bypass an existing predicate, contact requirement, stable-release window, or source annotation.
- Apply friction only to collision geoms belonging to the selected movable object; do not change visual geoms, robot geoms, fixtures, or support surfaces.
- Use deterministic profile version `vla82-material-friction-v1` and record every applied value.
- Use seed `2000` for the first diagnostic episode of each candidate.
- A result counts only when JSON has `status="PASS"` and `predicate_success=true`, NPZ exists, and the exported video is non-empty.
- The current workspace is not detected as a Git worktree, so each task ends with a diff/test checkpoint instead of a commit.

---

### Task 1: Pure material profile registry

**Files:**
- Create: `tools/vla82_full_sim/friction.py`
- Create: `tests/test_vla82_material_friction.py`

**Interfaces:**
- Produces: `FrictionProfile`, `profile_for(selection_id: str, semantic_class: str) -> FrictionProfile`, `format_friction(profile: FrictionProfile) -> str`.
- `FrictionProfile` fields: `version`, `material`, `source`, `sliding`, `torsional`, `rolling`; property `values -> tuple[float, float, float]`.

- [ ] **Step 1: Write failing mapping and fallback tests**

```python
from tools.vla82_full_sim.friction import PROFILE_VERSION, profile_for


def test_first_batch_profiles_are_material_specific():
    expected = {
        "VLA82-006": ("wood", (0.82, 0.008, 0.0003)),
        "VLA82-014": ("smooth_ceramic_glass", (0.40, 0.004, 0.0001)),
        "VLA82-018": ("paper_cardboard", (0.65, 0.006, 0.0002)),
        "VLA82-037": ("metal", (0.45, 0.004, 0.0001)),
        "VLA82-041": ("metal", (0.45, 0.004, 0.0001)),
        "VLA82-045": ("paper_cardboard", (0.65, 0.006, 0.0002)),
    }
    for selection_id, (material, values) in expected.items():
        profile = profile_for(selection_id, "")
        assert profile.version == PROFILE_VERSION
        assert profile.material == material
        assert profile.source == "selection_id"
        assert profile.values == values


def test_unknown_object_uses_declared_semantic_fallback():
    profile = profile_for("VLA82-999", "unknown utensil")
    assert profile.material == "plastic"
    assert profile.source == "semantic_fallback"
    assert profile.values == (0.55, 0.005, 0.0002)
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tests/test_vla82_material_friction.py -q`

Expected: collection fails because `tools.vla82_full_sim.friction` does not exist.

- [ ] **Step 3: Implement immutable profiles and deterministic selection mapping**

```python
from dataclasses import dataclass

PROFILE_VERSION = "vla82-material-friction-v1"


@dataclass(frozen=True)
class FrictionProfile:
    version: str
    material: str
    source: str
    sliding: float
    torsional: float
    rolling: float

    @property
    def values(self) -> tuple[float, float, float]:
        return (self.sliding, self.torsional, self.rolling)


_BASE = {
    "smooth_ceramic_glass": (0.40, 0.004, 0.0001),
    "metal": (0.45, 0.004, 0.0001),
    "plastic": (0.55, 0.005, 0.0002),
    "paper_cardboard": (0.65, 0.006, 0.0002),
    "wood": (0.82, 0.008, 0.0003),
    "rubber_sponge": (1.00, 0.012, 0.0005),
}
_SELECTION_MATERIAL = {
    "VLA82-001": "plastic", "VLA82-002": "rubber_sponge", "VLA82-003": "plastic",
    "VLA82-004": "plastic", "VLA82-005": "metal", "VLA82-006": "wood",
    "VLA82-007": "metal", "VLA82-008": "metal", "VLA82-009": "metal",
    "VLA82-010": "metal", "VLA82-011": "smooth_ceramic_glass",
    "VLA82-012": "smooth_ceramic_glass", "VLA82-013": "smooth_ceramic_glass",
    "VLA82-014": "smooth_ceramic_glass", "VLA82-015": "metal",
    "VLA82-016": "paper_cardboard", "VLA82-017": "plastic",
    "VLA82-018": "paper_cardboard", "VLA82-019": "metal",
    "VLA82-020": "smooth_ceramic_glass", "VLA82-021": "plastic",
    "VLA82-022": "wood", "VLA82-023": "plastic", "VLA82-024": "plastic",
    "VLA82-025": "paper_cardboard", "VLA82-026": "smooth_ceramic_glass",
    "VLA82-027": "plastic", "VLA82-028": "metal", "VLA82-029": "metal",
    "VLA82-030": "wood", "VLA82-031": "wood", "VLA82-032": "metal",
    "VLA82-033": "metal", "VLA82-034": "metal", "VLA82-035": "metal",
    "VLA82-036": "smooth_ceramic_glass", "VLA82-037": "metal",
    "VLA82-038": "metal", "VLA82-039": "wood",
    "VLA82-040": "smooth_ceramic_glass", "VLA82-041": "metal",
    "VLA82-042": "plastic", "VLA82-043": "metal",
    "VLA82-044": "smooth_ceramic_glass", "VLA82-045": "paper_cardboard",
    "VLA82-046": "paper_cardboard", "VLA82-047": "plastic",
    "VLA82-048": "paper_cardboard", "VLA82-049": "metal", "VLA82-050": "metal",
    "VLA82-051": "wood", "VLA82-052": "plastic", "VLA82-053": "plastic",
    "VLA82-054": "plastic", "VLA82-055": "paper_cardboard", "VLA82-056": "plastic",
    "VLA82-057": "plastic", "VLA82-058": "plastic", "VLA82-059": "plastic",
    "VLA82-060": "plastic",
}


def profile_for(selection_id: str, semantic_class: str) -> FrictionProfile:
    material = _SELECTION_MATERIAL.get(selection_id, "plastic")
    source = "selection_id" if selection_id in _SELECTION_MATERIAL else "semantic_fallback"
    return FrictionProfile(PROFILE_VERSION, material, source, *_BASE[material])


def format_friction(profile: FrictionProfile) -> str:
    return " ".join(f"{value:.6g}" for value in profile.values)
```

Add a coverage assertion `assert set(_SELECTION_MATERIAL) == {f"VLA82-{index:03d}" for index in range(1, 61)}` so accidental omissions fail at import time. The explicit table classifies the selected 60 consistently; `semantic_fallback` is reserved for non-registry diagnostics.

- [ ] **Step 4: Run focused tests**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tests/test_vla82_material_friction.py -q`

Expected: all Task 1 tests PASS.

---

### Task 2: Apply profiles to custom MJCF and live collision geoms

**Files:**
- Modify: `tools/vla82_full_sim/friction.py`
- Modify: `tools/vla82_full_sim/assets.py:426-525`
- Modify: `tools/vla82_full_sim/environment.py:1220-1258`
- Modify: `tools/vla82_full_sim/environment.py:1524-1570`
- Modify: `tests/test_vla82_material_friction.py`
- Modify: `tests/test_vla82_full_sim_assets.py`

**Interfaces:**
- Consumes: `profile_for()` and `format_friction()` from Task 1.
- Produces: `apply_object_friction(raw: Any, geom_names: Sequence[str], profile: FrictionProfile) -> dict[str, object]` and `environment.friction_evidence`.

- [ ] **Step 1: Add failing tests for collision-only mutation and evidence**

```python
def test_apply_object_friction_changes_only_contact_enabled_geoms():
    raw = fake_raw_model(
        names=("obj_visual", "obj_collision", "robot_collision"),
        contype=(0, 1, 1), conaffinity=(0, 1, 1),
        friction=((0.95, .3, .1), (0.95, .3, .1), (1.0, .005, .0001)),
    )
    evidence = apply_object_friction(
        raw, ("obj_visual", "obj_collision"), profile_for("VLA82-006", "rolling pin")
    )
    assert evidence["geom_names"] == ["obj_collision"]
    assert evidence["before"]["obj_collision"] == [0.95, 0.3, 0.1]
    assert evidence["after"]["obj_collision"] == [0.82, 0.008, 0.0003]
    assert raw.sim.model.geom_friction[2].tolist() == [1.0, .005, .0001]
```

- [ ] **Step 2: Run the focused test and verify it fails on the missing function**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tests/test_vla82_material_friction.py -q`

Expected: FAIL because `apply_object_friction` is not defined.

- [ ] **Step 3: Implement collision-geom filtering and evidence generation**

```python
def apply_object_friction(raw, geom_names, profile):
    model = raw.sim.model
    before, after, applied = {}, {}, []
    for name in geom_names:
        geom_id = int(model.geom_name2id(name))
        if geom_id < 0 or not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
            continue
        old = np.asarray(model.geom_friction[geom_id], dtype=float).copy()
        model.geom_friction[geom_id] = np.asarray(profile.values, dtype=float)
        before[name] = old.tolist()
        after[name] = list(profile.values)
        applied.append(name)
    if not applied:
        raise ValueError("selected object has no contact-enabled collision geom")
    return {
        "version": profile.version, "material": profile.material,
        "material_source": profile.source, "geom_names": applied,
        "before": before, "after": after, "application_stage": "post_load_pre_step",
    }
```

- [ ] **Step 4: Use the same profile in custom asset MJCF generation**

In `_xml_for_asset()`, replace the generic non-sponge friction string with `format_friction(profile_for(asset.selection_id, asset.semantic_class))`. In `asset_evidence.json`, store the same `material`, `material_source`, `friction`, and `friction_profile_version` rather than a generic hard-coded value.

- [ ] **Step 5: Apply live friction before runtime fingerprinting**

In `make_environment()`, after asset identity validation and before `scene_runtime_fingerprint()`:

```python
contract = build_physics_capture_contract(environment, request)
profile = profile_for(request.selection_id, request.primary_asset.semantic_class)
environment.friction_evidence = apply_object_friction(
    raw, contract.object_geom_names.get("obj", ()), profile
)
```

Add `"friction_evidence": getattr(environment, "friction_evidence", {})` to the runtime fingerprint payload so the scene hash binds the applied coefficients.

- [ ] **Step 6: Run profile, asset, environment, and no-shortcut tests**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tests/test_vla82_material_friction.py tests/test_vla82_full_sim_assets.py tests/test_vla82_full_sim_environment.py tests/test_vla82_full_sim_contracts.py -q`

Expected: all selected tests PASS; no assertion indicates robot/fixture/visual friction mutation.

---

### Task 3: Persist friction evidence in every episode

**Files:**
- Modify: `tools/vla82_full_sim/expert.py:5112-5204`
- Modify: `tests/test_vla82_full_sim_expert.py`

**Interfaces:**
- Consumes: `environment.friction_evidence` from Task 2.
- Produces: top-level diagnostic JSON field `friction_evidence` with actual applied live-model values.

- [ ] **Step 1: Add a failing collector serialization test**

Extend the existing mocked collector test so its environment exposes:

```python
environment.friction_evidence = {
    "version": "vla82-material-friction-v1",
    "material": "wood",
    "material_source": "selection_id",
    "geom_names": ["obj_collision"],
    "before": {"obj_collision": [0.95, 0.3, 0.1]},
    "after": {"obj_collision": [0.82, 0.008, 0.0003]},
    "application_stage": "post_load_pre_step",
}
```

Assert the written diagnostic contains that exact dictionary even when the episode fails its predicate.

- [ ] **Step 2: Run the focused test and confirm the field is missing**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tests/test_vla82_full_sim_expert.py -q`

Expected: the new assertion FAILS because `friction_evidence` is absent.

- [ ] **Step 3: Capture and serialize immutable friction evidence**

Initialize `friction_evidence: dict[str, Any] = {}` before the collector `try`, set it immediately after `make_environment()`, and add it to `_atomic_json()`:

```python
friction_evidence = dict(getattr(environment, "friction_evidence", {}))
...
_atomic_json(diagnostic_path, {
    **report.to_dict(),
    "runtime_scene": runtime_scene,
    "friction_evidence": friction_evidence,
    ...
})
```

- [ ] **Step 4: Run collector and contract regression tests**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tests/test_vla82_full_sim_expert.py tests/test_vla82_predicate_stability.py tests/test_vla82_full_sim_contracts.py -q`

Expected: all selected tests PASS.

---

### Task 4: Run first-batch strict simulations and diagnose failures

**Files:**
- Modify only if traces prove a defect: `tools/vla82_full_sim/expert.py`
- Modify only if a test exposes a spawn/binding defect: `tools/vla82_full_sim/environment.py`
- Add focused regression tests under: `tests/test_vla82_*.py`
- Generate: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-*/episode-2000.json`
- Generate on PASS: matching `.npz` files.

**Interfaces:**
- Consumes: friction-enabled `make_environment()` and unchanged strict predicates.
- Produces: honest PASS/FAIL diagnostics for `006,014,018,037,041,045`.

- [ ] **Step 1: Verify the simulator runtime before long runs**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -c "import cv2, mujoco, numpy, robosuite; print('runtime: PASS')"`

Expected: `runtime: PASS`.

- [ ] **Step 2: Run one fixed-seed attempt per first-batch object**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py collect-demos --episodes-per-object 1 --max-attempts-per-object 1 --selection-id VLA82-006,VLA82-014,VLA82-018,VLA82-037,VLA82-041,VLA82-045`

Expected: one diagnostic JSON per ID; PASS episodes additionally have NPZ. A failed process exit is acceptable at this diagnostic stage, but missing JSON is not.

- [ ] **Step 3: Classify each failure from physical traces**

For every FAIL, extract `errors`, `controller_trace`, contact snapshots, object displacement and final predicate. Fix only the earliest broken physical stage: reach, contact, two-pad grasp, lift, collision-free transport, target entry, release, or stable settle.

- [ ] **Step 4: For each code defect, add a reproducing test before the minimal fix**

Examples of acceptable assertions are exact geom binding, target location, sign of controller displacement, or stable-release dwell. Do not assert a hard-coded PASS and do not modify `evaluate_operation()` thresholds to match a failed trace.

- [ ] **Step 5: Re-run the affected object after every tested fix**

Run the Task 4 Step 2 command with `--selection-id` narrowed to the affected ID.

Expected: JSON reports true physical state. Continue diagnosis after three failures rather than stopping or inflating friction.

- [ ] **Step 6: Run the full focused regression set**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tests/test_vla82_material_friction.py tests/test_vla82_full_sim_assets.py tests/test_vla82_full_sim_environment.py tests/test_vla82_full_sim_expert.py tests/test_vla82_full_sim_predicates.py tests/test_vla82_full_sim_contracts.py -q`

Expected: all selected tests PASS.

---

### Task 5: Export visible evidence and summarize only verified results

**Files:**
- Generate: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-*/episode-2000.primary.mp4`
- Generate: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-*/episode-2000.wrist.mp4`
- Generate: per-PASS start/contact/end PNGs beside each episode.
- Update after verification: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/collection_manifest.json`

**Interfaces:**
- Consumes: only strict-PASS JSON/NPZ pairs from Task 4.
- Produces: reviewer-visible videos/images and a count derived from verified files.

- [ ] **Step 1: Verify every proposed PASS pair before export**

Read each JSON and require `status == "PASS"`, `predicate_success is True`, non-empty `friction_evidence.geom_names`, and matching NPZ. Reject any entry lacking one condition.

- [ ] **Step 2: Export both recorded cameras**

Run the following over the six candidate directories; the JSON guard ensures only verified PASS episodes are exported:

```powershell
$ids = '006','014','018','037','041','045'
foreach ($id in $ids) {
    $stem = "outputs\midterm_testing_vla82\full_simulation_objectwise_8x60\expert_demos\VLA82-$id\episode-2000"
    $result = Get-Content "$stem.json" -Raw | ConvertFrom-Json
    if ($result.status -eq 'PASS' -and $result.predicate_success -eq $true -and (Test-Path "$stem.npz")) {
        C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\export_vla82_episode_video.py "$stem.npz" --camera primary --fps 12
        C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\export_vla82_episode_video.py "$stem.npz" --camera wrist --fps 12
    }
}
```

Expected: two non-empty MP4 files; the exporter refuses non-PASS JSON.

- [ ] **Step 3: Save start, first-contact, and end frames**

Load frames from the PASS NPZ, choose the first snapshot whose contact set includes an object/gripper pair, and save the corresponding primary frame plus frame 0 and the final frame. Record selected frame indices in the episode JSON evidence metadata.

- [ ] **Step 4: Visually inspect each exported primary video**

Require the robot, selected object, source surface and target region to remain visible during the decisive action. If occluded, regenerate only the camera view; do not rerun or alter the successful trajectory unless the recorded stream itself is unusable.

- [ ] **Step 5: Recompute the strict PASS count from disk**

Count only episode JSON files satisfying the strict fields and having their matching NPZ and MP4. Report new PASS IDs, unchanged PASS IDs, remaining failures, friction values and absolute evidence paths without claiming 60/60 unless all 60 meet these checks.
