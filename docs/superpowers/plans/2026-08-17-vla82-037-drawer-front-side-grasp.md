# VLA82-037 Drawer-Front Side Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace VLA82-037's countertop-blocked top grasp with a collision-gated, two-stage drawer-front entry and strict physical side grasp.

**Architecture:** Add pure geometry helpers for the live drawer opening and tool-aligned waypoints, route authoritative drawer-source requests into a dedicated `drawer_front_side` strategy, then integrate three bounded arm-only states before the existing strict close/lift/place path. All world directions come from MuJoCo joint and geom state; no object, drawer, contact, phase, or success state is written directly.

**Tech Stack:** Python 3.11, NumPy, RoboCasa, RoboSuite, MuJoCo, `unittest`, `pytest`.

## Global Constraints

- Do not issue base translation before first object contact; no pre-grasp `reach_base` state.
- Do not modify drawer qpos, object pose, object friction, contact evidence, task phase, or success state.
- Require both RoboSuite native grasp and two native finger-pad contacts before `secure`.
- Abort drawer entry on real robot–fixture material contact.
- Preserve all strict transport, release, stable support, video, and workbook gates.
- Run exactly one full strict VLA82-037 attempt after all regressions pass.
- This workspace has no Git repository; record test outputs and file paths instead of commit steps.

---

### Task 1: Pure Drawer-Front Waypoint Geometry

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Modify: `tools/vla82_full_sim/expert.py`

**Interfaces:**
- Consumes: object center, robot base, drawer bbox low/high, slide-axis world vector, object geom rotation and half-size.
- Produces: `drawer_front_side_waypoints(...) -> tuple[np.ndarray, np.ndarray, np.ndarray]`, returning orient/stage-independent front stage, inserted side point, and centered grasp target.

- [ ] **Step 1: Add failing pure tests**

Add tests that call the wished-for helper with a drawer spanning `x=[1.7475,2.1025]`, `y=[-0.600,0.0]`, `z=[0.670,0.880]`, object center `(1.928,-0.584,0.743)`, robot base `(1.928,-0.993,0.700)`, slide axis `(0,1,0)`, and a horizontal 95×12×12 mm object. Assert:

```python
stage, inserted, grasp = expert.drawer_front_side_waypoints(
    object_center=(1.928, -.584, .743),
    robot_base=(1.928, -.993, .700),
    drawer_low=(1.7475, -.600, .670),
    drawer_high=(2.1025, 0., .880),
    slide_axis_world=(0., 1., 0.),
    geom_rotation=np.eye(3),
    geom_size=(.0475, .006, .006),
)
self.assertLess(stage[1], -.600)
self.assertAlmostEqual(inserted[1], -.584, places=6)
self.assertAlmostEqual(stage[0], inserted[0], places=6)
self.assertLessEqual(stage[2], .860)
self.assertGreaterEqual(stage[2], .764)
self.assertAlmostEqual(grasp[0], 1.928, places=6)
```

Add rotated-axis coverage and assert `EnvironmentValidationError` for a zero horizontal slide axis or an object whose valid side waypoint cannot fit inside the drawer width.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest `
  tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_drawer_front_side_waypoints_use_live_opening `
  tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_drawer_front_side_waypoints_rotate_with_slide_axis `
  tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_drawer_front_side_waypoints_reject_invalid_geometry
```

Expected: import or attribute failure because `drawer_front_side_waypoints` does not exist.

- [ ] **Step 3: Implement the minimum pure helper**

In `expert.py`, normalize the horizontal slide axis, choose its sign by the robot-base projection, compute the orthogonal drawer-width axis, derive the object's long horizontal world axis and projected half-extent, select the nearer long-axis end that remains within the drawer width, and return:

```python
front_stage = side_point + outward_slide * .050
front_stage[2] = np.clip(object_top + .035, drawer_low[2] + .035, drawer_high[2] - .020)
inserted = front_stage.copy()
inserted[:2] = side_point[:2]
inserted[2] = front_stage[2]
grasp = np.asarray(object_center, dtype=float).copy()
```

The side-point clearance is 35 mm beyond the selected object long-axis end. Validate finite inputs, nonzero axes, positive drawer extents, ceiling clearance, and drawer-width containment.

- [ ] **Step 4: Run the three tests and verify GREEN**

Run the Step 2 command. Expected: `Ran 3 tests ... OK`.

---

### Task 2: Authoritative Drawer-Front Strategy Routing

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Modify: `tools/vla82_full_sim/expert.py`

**Interfaces:**
- Consumes: `SceneRequest.source_fixture`, `selection_id`, and existing strategy helpers.
- Produces: `drawer_front_side` routing through `drawer_side_orient`, pad-center alignment, strict close locking, and same-strategy retry.

- [ ] **Step 1: Add failing routing tests**

Add assertions:

```python
self.assertTrue(expert.pick_place_uses_pad_center_alignment("VLA82-037"))
self.assertEqual(expert.pick_place_grasp_strategy("VLA82-037"), "drawer_front_side")
self.assertEqual(
    expert.pick_place_initial_state_for_request(_scene_for_spec(specs["VLA82-037"])),
    "drawer_side_orient",
)
self.assertEqual(expert.pick_place_retry_grasp_state("VLA82-037"), "drawer_side_orient")
np.testing.assert_allclose(
    expert.pick_place_close_target(
        "VLA82-037", eef=(1., 2., 3.), tracking_target=(4., 5., 6.),
    ),
    np.array((1., 2., 3.)),
)
```

Retain explicit controls showing VLA82-038 remains `top`, VLA82-021 remains `side`, and counter/cabinet source initial states are unchanged.

- [ ] **Step 2: Run routing tests and verify RED**

Run the named new test methods with `python -m unittest`. Expected: VLA82-037 currently returns no pad alignment, `top`, `approach`, and an unlocked close target.

- [ ] **Step 3: Implement minimal routing**

Update the existing helpers only:

```python
if selected == "VLA82-037":
    return "drawer_front_side"
```

Add VLA82-037 to pad-center alignment and close-target locking. For a request whose `source_fixture == "drawer"` and strategy is `drawer_front_side`, return `drawer_side_orient`; retry returns the same state. Do not alter non-drawer routing.

- [ ] **Step 4: Run routing tests and verify GREEN**

Expected: all new routing tests pass.

---

### Task 3: Integrate Collision-Gated Drawer Entry States

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Modify: `tools/vla82_full_sim/expert.py`

**Interfaces:**
- Consumes: `drawer_front_side_waypoints`, live source drawer slide joint/bbox, pad-center and opening-axis helpers, `action_for`, and material contact names.
- Produces: controller flow `drawer_side_orient -> drawer_front_stage -> drawer_front_insert -> side_center -> close`.

- [ ] **Step 1: Add failing transition and safety tests**

Add pure transition helpers and tests for the wished-for APIs:

```python
self.assertEqual(expert.drawer_front_orientation_transition(.20, .02), "drawer_side_orient")
self.assertEqual(expert.drawer_front_orientation_transition(.02, .20), "drawer_side_orient")
self.assertEqual(expert.drawer_front_orientation_transition(.02, .02), "drawer_front_stage")
self.assertEqual(expert.drawer_front_motion_transition("drawer_front_stage", .03, False), "drawer_front_insert")
self.assertEqual(expert.drawer_front_motion_transition("drawer_front_insert", .03, False), "side_center")
self.assertEqual(expert.drawer_front_motion_transition("drawer_front_insert", .10, True), "failed")
```

Add a source check that the drawer-entry states issue no base action, use `base_mode=-1`, and append `drawer_front_collision_abort` before returning on material robot–fixture contact.

- [ ] **Step 2: Run transition tests and verify RED**

Expected: missing helper failures.

- [ ] **Step 3: Implement transition helpers and state-machine branches**

Implement:

```python
def drawer_front_orientation_transition(tool_error, opening_error, tolerance=np.deg2rad(6.)):
    return "drawer_front_stage" if max(float(tool_error), float(opening_error)) <= tolerance else "drawer_side_orient"

def drawer_front_motion_transition(state, distance, fixture_collision, threshold=.04):
    if fixture_collision:
        return "failed"
    if float(distance) > threshold:
        return state
    return "drawer_front_insert" if state == "drawer_front_stage" else "side_center"
```

Before the loop, resolve the source drawer's slide joint and live bbox. During `drawer_side_orient`, align the tool axis to the selected object long axis and the gripper opening axis to the short horizontal object axis while holding position. During stage and insert, convert desired pad-center positions to wrist targets, command arm translation only, keep gripper open, set no torso/base motion, and fail immediately on material furniture contact. In `side_center`, move from the inserted long-axis end toward the object at limit `.20`; transition to `close` only within 25 mm or on exact contact.

Trace `drawer_front_stage_world`, `drawer_front_insert_world`, both alignment errors, fixture collision state, and all existing strict contact fields.

- [ ] **Step 4: Run transition tests and verify GREEN**

Expected: all new transition and source-safety tests pass.

---

### Task 4: Live Environment Smoke Gate

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Verify: `tools/vla82_full_sim/environment.py`
- Verify: `tools/vla82_full_sim/expert.py`

**Interfaces:**
- Consumes: seed-2000 VLA82-037 environment and live waypoint helper.
- Produces: proof that stage/insert points are valid for the actual generated drawer and tongs.

- [ ] **Step 1: Add the environment smoke test**

Load VLA82-037 seed 2000, derive the slide axis and object geom data, call `drawer_front_side_waypoints`, and assert all coordinates are finite, stage is outside the drawer front, inserted point is inside the width/depth opening, both heights are below `drawer_high_z - .020`, and reset has no material mobile-base/drawer contact.

- [ ] **Step 2: Run the smoke test**

Run the single named `unittest`. Expected: `Ran 1 test ... OK` within normal environment startup time.

- [ ] **Step 3: Stop on smoke failure**

If any waypoint violates the live bbox or reset contact gate, do not run the full physical simulation. Correct the pure geometry/helper discrepancy and repeat only the smoke test.

---

### Task 5: Regression Gates and One Strict Physical Attempt

**Files:**
- Verify: `tools/tests/test_vla82_source_spawn.py`
- Verify: `tools/tests/test_vla82_pure_closed_loop.py`
- Verify: `tools/tests/test_vla82_predicate_stability.py`
- Generate on attempt: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-037/episode-2000.json`
- Conditionally generate on strict PASS: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos_clear_view/VLA82-037/episode-2000.strict-pass-replay-wide.mp4`

**Interfaces:**
- Consumes: Tasks 1–4 and unchanged strict predicates.
- Produces: one auditable strict result; video/index changes only after strict and visual PASS.

- [ ] **Step 1: Run compile and full source-spawn regression**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m compileall -q tools\vla82_full_sim\expert.py tools\tests\test_vla82_source_spawn.py
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
```

Expected: compile succeeds and every source-spawn test passes.

- [ ] **Step 2: Run strict predicate regression**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest -q tools\tests\test_vla82_pure_closed_loop.py tools\tests\test_vla82_predicate_stability.py
```

Expected: all strict predicate tests pass.

- [ ] **Step 3: Run exactly one strict VLA82-037 attempt**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py collect-demos --episodes-per-object 1 --max-attempts-per-object 1 --selection-id VLA82-037
```

- [ ] **Step 4: Audit the trace**

Require `status == "PASS"`, empty errors, first state `drawer_side_orient`, ordered presence of `drawer_front_stage`, `drawer_front_insert`, `side_center`, `close`, zero pre-contact `reach_base`, zero pre-contact base translation, verified native grasp and two-pad contact, transport without grasp loss, and final stable counter support.

- [ ] **Step 5: Stop cleanly on strict FAIL**

Preserve JSON diagnostics, report the first failed state and measured geometry/contact evidence, do not rerun, do not export a result video, and do not update the strict-PASS workbook.

- [ ] **Step 6: Export and index only after strict PASS**

Replay at 24 fps and 1280×720 using `tools/replay_vla82_strict_pass_video.py`, visually verify unobstructed extraction/transport/release, then update and reopen the established strict-PASS workbook. If visual evidence is obstructed or the strict audit fails, keep the index at its prior count.

