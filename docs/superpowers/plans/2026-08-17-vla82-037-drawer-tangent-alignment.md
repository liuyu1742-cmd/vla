# VLA82-037 Drawer Tangent Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the reset-time mobile base with the tongs across the drawer width while preserving the native front standoff, then begin VLA82-037 with arm-only approach instead of pushing the open drawer.

**Architecture:** Add one pure projection helper and one source-drawer reset hook in `environment.py`. The hook obtains the live slide-joint world axis, preserves the native base projection along that axis, and replaces only the perpendicular projection with the sampled object's projection. `expert.py` then selects `approach` from authoritative `source_fixture == "drawer"` semantics, preventing pre-grasp base translation.

**Tech Stack:** Python 3.11, NumPy, MuJoCo/RoboCasa, `unittest`, existing VLA82 strict predicate and replay pipeline.

## Global Constraints

- Keep the existing `0.39 m` VLA82-037 arm-reach threshold unchanged.
- Keep the native base projection along the drawer slide axis unchanged.
- Change only the reset-time base projection perpendicular to the slide axis.
- Do not change drawer, object, joint, qpos, pose, contact, or success state.
- Do not add runtime base motion before the grasp.
- Do not weaken strict phase, contact, grasp, transport, release, or support predicates.
- Do not affect counter-to-drawer, cabinet-source, or non-drawer-source tasks.
- Run one full VLA82-037 physical attempt only after the geometry, semantic, smoke, and regression gates pass.
- This workspace is not a Git repository; omit commits and preserve all existing user files.

---

### Task 1: Compute a Slide-Axis-Preserving Drawer Source Anchor

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Modify: `tools/vla82_full_sim/environment.py:592-646`

**Interfaces:**
- Consumes: drawer center, native base anchor, sampled object center, and live drawer slide axis in world coordinates.
- Produces: `source_drawer_robot_anchor(*, drawer_center: Sequence[float], native_anchor: Sequence[float], object_center: Sequence[float], slide_axis_world: Sequence[float]) -> np.ndarray`.
- Raises: `EnvironmentValidationError` for a non-finite or zero-length horizontal slide axis.

- [ ] **Step 1: Write RED tests for projection invariants**

Add `source_drawer_robot_anchor` to the existing environment imports and add:

```python
def test_drawer_source_anchor_preserves_slide_standoff_and_aligns_width(self) -> None:
    drawer = np.array((1.925, -0.300, 0.775))
    native = np.array((1.651, -0.943, 0.700))
    obj = np.array((1.928, -0.584, 0.743))
    slide = np.array((0.0, 1.0, 0.0))

    aligned = source_drawer_robot_anchor(
        drawer_center=drawer,
        native_anchor=native,
        object_center=obj,
        slide_axis_world=slide,
    )

    tangent = np.array((1.0, 0.0))
    self.assertAlmostEqual(float(np.dot(aligned[:2] - drawer[:2], slide[:2])),
                           float(np.dot(native[:2] - drawer[:2], slide[:2])))
    self.assertAlmostEqual(float(np.dot(aligned[:2] - obj[:2], tangent)), 0.0)
    self.assertAlmostEqual(float(aligned[2]), float(native[2]))
    self.assertLessEqual(float(np.linalg.norm(aligned[:2] - obj[:2])), 0.39)

def test_drawer_source_anchor_rejects_zero_horizontal_slide_axis(self) -> None:
    with self.assertRaisesRegex(EnvironmentValidationError, "slide axis"):
        source_drawer_robot_anchor(
            drawer_center=(0.0, 0.0, 0.0),
            native_anchor=(0.0, -0.5, 0.0),
            object_center=(0.1, -0.2, 0.1),
            slide_axis_world=(0.0, 0.0, 1.0),
        )
```

- [ ] **Step 2: Run the tests and observe RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest `
  tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_drawer_source_anchor_preserves_slide_standoff_and_aligns_width `
  tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_drawer_source_anchor_rejects_zero_horizontal_slide_axis
```

Expected: import error because `source_drawer_robot_anchor` does not exist.

- [ ] **Step 3: Implement the minimal pure projection helper**

Add next to `source_cabinet_robot_anchor`:

```python
def source_drawer_robot_anchor(
    *, drawer_center: Sequence[float], native_anchor: Sequence[float],
    object_center: Sequence[float], slide_axis_world: Sequence[float],
) -> np.ndarray:
    drawer = np.asarray(drawer_center, dtype=float)
    native = np.asarray(native_anchor, dtype=float)
    obj = np.asarray(object_center, dtype=float)
    slide = np.asarray(slide_axis_world, dtype=float)[:2]
    norm = float(np.linalg.norm(slide))
    if not np.isfinite(norm) or norm <= 1e-9:
        raise EnvironmentValidationError("drawer slide axis has no horizontal direction")
    slide /= norm
    tangent = np.array((-slide[1], slide[0]), dtype=float)
    native_slide = float(np.dot(native[:2] - drawer[:2], slide))
    object_tangent = float(np.dot(obj[:2] - drawer[:2], tangent))
    result = native.copy()
    result[:2] = drawer[:2] + slide * native_slide + tangent * object_tangent
    return result
```

- [ ] **Step 4: Run the focused tests and observe GREEN**

Run the Step 2 command.

Expected: `Ran 2 tests ... OK`.

---

### Task 2: Install the Reset Hook From Source-Drawer Semantics

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Modify: `tools/vla82_full_sim/environment.py:638-744`
- Modify: `tools/vla82_full_sim/environment.py:1002-1010`

**Interfaces:**
- Consumes: `SceneRequest.source_fixture`, `raw.drawer`, `raw.object_placements["obj"]`, the drawer slide joint, and `source_drawer_robot_anchor`.
- Produces: `uses_drawer_source_alignment(request: SceneRequest | Any) -> bool` and `_configure_drawer_source_robot_spawn(raw: Any, request: SceneRequest) -> None`.
- Records: `_vla82_robot_spawn_diagnostics` with mode, native anchor, aligned anchor, object center, world slide axis, native slide projection, and aligned object distance.

- [ ] **Step 1: Write RED semantic and hook-selection tests**

Add:

```python
def test_vla82_037_uses_source_drawer_alignment(self) -> None:
    from tools.vla82_full_sim.environment import uses_drawer_source_alignment

    specs = {item.selection_id: item for item in _load_compiled_specs()}
    drawer_source = _scene_for_spec(specs["VLA82-037"])
    counter_to_drawer = _scene_for_spec(specs["VLA82-005"])
    cabinet_source = _scene_for_spec(specs["VLA82-036"])

    self.assertEqual(drawer_source.source_fixture, "drawer")
    self.assertTrue(uses_drawer_source_alignment(drawer_source))
    self.assertFalse(uses_drawer_source_alignment(counter_to_drawer))
    self.assertFalse(uses_drawer_source_alignment(cabinet_source))
```

- [ ] **Step 2: Run the semantic test and observe RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_vla82_037_uses_source_drawer_alignment
```

Expected: import error because `uses_drawer_source_alignment` does not exist.

- [ ] **Step 3: Implement semantic selection and live world-axis conversion**

Add:

```python
def uses_drawer_source_alignment(request: SceneRequest | Any) -> bool:
    return str(getattr(request, "source_fixture", "")) == "drawer"


def _configure_drawer_source_robot_spawn(raw: Any, request: SceneRequest) -> None:
    if not uses_drawer_source_alignment(request):
        return
    original = raw._setup_scene

    def configured(self: Any) -> None:
        original()
        self.sim.forward()
        try:
            object_center = np.asarray(self.object_placements["obj"][0], dtype=float)
            prefix = str(self.drawer.naming_prefix)
            joint_id = next(
                index for index in range(int(self.sim.model.njnt))
                if str(self.sim.model.joint_id2name(index) or "").startswith(prefix)
                and int(self.sim.model.jnt_type[index]) == 2
            )
            body_id = int(self.sim.model.jnt_bodyid[joint_id])
            body_rotation = np.asarray(self.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
            slide_axis_world = body_rotation @ np.asarray(self.sim.model.jnt_axis[joint_id], dtype=float)
            native_anchor = np.asarray(self.init_robot_base_pos_anchor, dtype=float).copy()
            anchor = source_drawer_robot_anchor(
                drawer_center=self.drawer.pos,
                native_anchor=native_anchor,
                object_center=object_center,
                slide_axis_world=slide_axis_world,
            )
        except (AttributeError, KeyError, IndexError, StopIteration, TypeError, ValueError) as error:
            raise EnvironmentValidationError(
                f"cannot configure source-drawer robot spawn: {request.selection_id}"
            ) from error
        self.init_robot_base_pos_anchor = anchor
        self.robot_spawn_deviation_pos_x = 0.0
        self.robot_spawn_deviation_pos_y = 0.0
        self.robot_spawn_deviation_rot = 0.0
        self._vla82_robot_spawn_diagnostics = {
            "selection_id": request.selection_id,
            "mode": "source_drawer_tangent_alignment",
            "native_anchor": native_anchor.tolist(),
            "anchor": anchor.tolist(),
            "object_center": object_center.tolist(),
            "slide_axis_world": slide_axis_world.tolist(),
            "aligned_object_distance": float(np.linalg.norm(anchor[:2] - object_center[:2])),
            "task_initially_solved": False,
        }

    raw._setup_scene = MethodType(configured, raw)
```

Call the hook in `_build_registered_or_custom_robocasa_task` after `_configure_cabinet_source_robot_spawn` and before `raw.reset()`:

```python
_configure_drawer_source_robot_spawn(raw, request)
```

- [ ] **Step 4: Run the semantic test and observe GREEN**

Run the Step 2 command.

Expected: `Ran 1 test ... OK`.

---

### Task 3: Begin Drawer-Source Tasks With Arm-Only Approach

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Modify: `tools/vla82_full_sim/expert.py:2372-2386`

**Interfaces:**
- Consumes: authoritative `SceneRequest.source_fixture`.
- Produces: `pick_place_initial_state_for_request(request)` returns `"approach"` for drawer-source tasks and preserves all existing cabinet/counter behavior.

- [ ] **Step 1: Write the RED controller-state test**

Add:

```python
def test_drawer_source_starts_arm_approach_without_base_chase(self) -> None:
    from tools.vla82_full_sim.expert import pick_place_initial_state_for_request

    specs = {item.selection_id: item for item in _load_compiled_specs()}
    self.assertEqual(
        pick_place_initial_state_for_request(_scene_for_spec(specs["VLA82-037"])),
        "approach",
    )
    self.assertEqual(
        pick_place_initial_state_for_request(_scene_for_spec(specs["VLA82-005"])),
        "align_yaw",
    )
    self.assertEqual(
        pick_place_initial_state_for_request(_scene_for_spec(specs["VLA82-036"])),
        "cabinet_front_stage",
    )
```

- [ ] **Step 2: Run the controller-state test and observe RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_drawer_source_starts_arm_approach_without_base_chase
```

Expected: FAIL because VLA82-037 currently returns `reach_base`.

- [ ] **Step 3: Add the minimal semantic transition**

Modify `pick_place_initial_state_for_request`:

```python
def pick_place_initial_state_for_request(request: SceneRequest | Any) -> str:
    if str(getattr(request, "source_fixture", "")) == "drawer":
        return "approach"
    if uses_cabinet_source_alignment(request) and str(getattr(request, "selection_id", "")) != "VLA82-021":
        return "cabinet_front_stage"
    return pick_place_initial_state(str(getattr(request, "selection_id", "")))
```

- [ ] **Step 4: Run the controller-state test and observe GREEN**

Run the Step 2 command.

Expected: `Ran 1 test ... OK`.

---

### Task 4: Prove Reset Reachability and Collision-Free Staging

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Verify: `tools/vla82_full_sim/environment.py`

**Interfaces:**
- Consumes: the reset hook and controller semantic transition.
- Produces: one environment smoke regression for VLA82-037 seed 2000.

- [ ] **Step 1: Write the environment smoke test**

Add:

```python
def test_vla82_037_reset_is_arm_reachable_without_base_drawer_contact(self) -> None:
    spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-037")
    environment = make_environment(_scene_for_spec(spec), seed=2000)
    try:
        raw = _raw_environment(environment)
        base, _ = raw.robots[0].composite_controller.part_controllers["base"].get_base_pose()
        object_body = raw.sim.model.body_name2id(raw.objects["obj"].root_body)
        source = np.asarray(raw.sim.data.body_xpos[object_body], dtype=float)
        self.assertLessEqual(float(np.linalg.norm(source[:2] - np.asarray(base)[:2])), 0.39)
        self.assertEqual(raw._vla82_robot_spawn_diagnostics["mode"], "source_drawer_tangent_alignment")
        self.assertFalse(raw._vla82_robot_spawn_diagnostics["task_initially_solved"])
        drawer_prefix = str(raw.drawer.naming_prefix)
        for index in range(int(raw.sim.data.ncon)):
            contact = raw.sim.data.contact[index]
            names = {
                str(raw.sim.model.geom_id2name(int(contact.geom1)) or ""),
                str(raw.sim.model.geom_id2name(int(contact.geom2)) or ""),
            }
            self.assertFalse(
                any(name.startswith("mobilebase0_") for name in names)
                and any(name.startswith(drawer_prefix) for name in names)
            )
    finally:
        environment.close()
```

- [ ] **Step 2: Run the smoke test**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_vla82_037_reset_is_arm_reachable_without_base_drawer_contact
```

Expected: `Ran 1 test ... OK` within the normal environment startup time; initial distance is approximately `0.359 m`.

- [ ] **Step 3: Stop on smoke failure**

If the distance exceeds `0.39 m`, any base-drawer contact exists, or the environment throws, do not run the full simulation. Read `_vla82_robot_spawn_diagnostics`, the world slide axis, and named contact pair; return to the failing helper/hook test rather than changing thresholds.

---

### Task 5: Run Regression Gates and One Strict Physical Retest

**Files:**
- Verify: `tools/tests/test_vla82_source_spawn.py`
- Verify: `tools/tests/test_vla82_pure_closed_loop.py`
- Verify: `tools/tests/test_vla82_predicate_stability.py`
- Generate: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-037/episode-2000.json`
- Generate: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-037/episode-2000.npz`
- Conditionally generate: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos_clear_view/VLA82-037/episode-2000.strict-pass-replay-wide.mp4`
- Conditionally update: strict-PASS workbook under `outputs/strict_pass_video_index_20260811/VLA82_严格PASS无遮挡结果视频路径汇总*`

**Interfaces:**
- Consumes: Tasks 1–4 and unchanged strict predicates.
- Produces: one fresh strict diagnostic episode; video and workbook changes only after strict and visual PASS.

- [ ] **Step 1: Run the full source-spawn suite**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
```

Expected: all tests pass.

- [ ] **Step 2: Run strict predicate regression**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest -q tools/tests/test_vla82_pure_closed_loop.py tools/tests/test_vla82_predicate_stability.py
```

Expected: all tests pass.

- [ ] **Step 3: Run exactly one VLA82-037 physical attempt**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py collect-demos --episodes-per-object 1 --max-attempts-per-object 1 --selection-id VLA82-037
```

Required trace gate before success assessment: the initial controller state is `approach`, no pre-grasp trace uses `reach_base`, and no base translation action is issued before the first object contact.

- [ ] **Step 4: Audit strict result**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -c "import json,pathlib; p=pathlib.Path(r'outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-037/episode-2000.json'); j=json.loads(p.read_text(encoding='utf-8')); pre=[t for t in j['controller_trace'] if not t.get('exact_gripper_object_contact')]; print({'status':j['status'],'steps':j['steps'],'predicate_success':j['predicate_success'],'errors':j['errors'],'first_state':j['controller_trace'][0].get('state'),'pregrasp_reach_base':sum(t.get('state')=='reach_base' for t in pre),'contact_verified':j.get('predicate',{}).get('contact_verified')})"
```

PASS gate: `status == "PASS"`, `predicate_success is True`, errors are empty, first state is `approach`, pre-grasp `reach_base` count is zero, contact is verified, and the object finishes stably supported by the counter.

- [ ] **Step 5: Stop cleanly on strict FAIL**

If any PASS gate fails, preserve the JSON/NPZ, identify the first failed arm stage, do not rerun, do not export a result video, and do not modify the workbook.

- [ ] **Step 6: Export and index only after strict PASS**

Run only after Step 4 passes:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\replay_vla82_strict_pass_video.py `
  outputs\midterm_testing_vla82\full_simulation_objectwise_8x60\expert_demos\VLA82-037\episode-2000.npz `
  outputs\midterm_testing_vla82\full_simulation_objectwise_8x60\expert_demos_clear_view\VLA82-037\episode-2000.strict-pass-replay-wide.mp4 `
  --fps 24 --width 1280 --height 720
```

Visually verify unobstructed grasp, extraction, transport, release, and final support. Then append the absolute MP4/JSON/NPZ paths to the established strict-PASS workbook using the artifact-tool pattern in `tools/update_vla82_strict_pass_index_11.mjs`, inspect formulas, render a preview, and reopen the saved workbook. Otherwise leave the 11/60 index unchanged.

