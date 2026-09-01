# VLA82-029 Free-Edge Bypass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Panda arm physically route around the open microwave door's free edge, contact the closing side, and push the hinge from its open limit toward zero without direct state writes.

**Architecture:** Add one pure geometry helper that returns fixed bypass waypoints from the live door OBB, hinge anchor, end-effector position, and closing tangent. Integrate a microwave-only bypass state machine into `DrawerClosePrimitive`; after bypass completion, reuse the existing exact-contact-gated close controller and strict predicate.

**Tech Stack:** Python 3.11, NumPy, `unittest`, `pytest`, RoboCasa, robosuite, MuJoCo.

## Global Constraints

- Do not write fixture joint state, teleport bodies, disable collision, weaken the existing close threshold, or alter result files by hand.
- Enable free-edge bypass only when the selected close joint name contains `microjoint`; preserve existing drawer, fridge, oven, and cabinet behavior.
- Latch the bypass waypoints before executing the route; update the closing tangent from live joint geometry only after physical door motion begins.
- Require real selected-door/gripper contact before `push_close`.
- Mark VLA82-029 strict PASS only after a fresh physical run reports `status: PASS`, `predicate_success: true`, and consistent contact/joint traces.
- Do not update the strict-PASS workbook or publish a result video unless the fresh run passes every strict condition.
- This workspace has no Git repository, so replace commit steps with atomic `apply_patch` edits and explicit passing-test checkpoints.

---

### Task 1: Pure free-edge bypass geometry

**Files:**
- Modify: `tools/vla82_full_sim/expert.py:765-923`
- Test: `tools/tests/test_vla82_source_spawn.py:1170-1340`

**Interfaces:**
- Consumes: door center `Sequence[float]`, door rotation `Sequence[float]`, door half-size `Sequence[float]`, hinge anchor `Sequence[float]`, current EEF `Sequence[float]`, normalized close tangent `Sequence[float]`, and scalar clearances.
- Produces: `fixture_hinge_free_edge_waypoints(panel_center=, panel_rotation=, panel_half_size=, hinge_anchor=, eef_world=, push_direction=, edge_clearance=, side_clearance=) -> dict[str, np.ndarray]` with keys `free_edge`, `outside_current_side`, `outside_closing_side`, and `contact_prepoint`.

- [ ] **Step 1: Write failing synthetic-geometry tests**

Add tests equivalent to:

```python
def test_microwave_bypass_uses_edge_farthest_from_hinge(self):
    from tools.vla82_full_sim import expert
    points = expert.fixture_hinge_free_edge_waypoints(
        panel_center=(0., 0., 0.), panel_rotation=np.eye(3),
        panel_half_size=(.25, .01, .15), hinge_anchor=(.25, 0., 0.),
        eef_world=(0., -.20, 0.), push_direction=(0., -1., 0.),
        edge_clearance=.09, side_clearance=.06,
    )
    self.assertLess(points["free_edge"][0], -.249)
    self.assertLess(points["outside_current_side"][0], -.33)
    self.assertGreater(points["outside_closing_side"][1], .05)
    self.assertGreater(points["contact_prepoint"][1], .05)

def test_microwave_bypass_rotates_with_the_live_panel(self):
    from tools.vla82_full_sim import expert
    yaw = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
    points = expert.fixture_hinge_free_edge_waypoints(
        panel_center=(1., 2., .8), panel_rotation=yaw,
        panel_half_size=(.25, .01, .15), hinge_anchor=(1., 2.25, .8),
        eef_world=(1.2, 2., .8), push_direction=(1., 0., 0.),
        edge_clearance=.09, side_clearance=.06,
    )
    self.assertGreater(np.linalg.norm(points["free_edge"] - (1., 2., .8)), .24)
    self.assertLess(float(np.dot(
        points["contact_prepoint"] - np.array((1., 2., .8)),
        np.array((1., 0., 0.)),
    )), -.05)

def test_free_edge_bypass_is_microwave_only(self):
    from tools.vla82_full_sim import expert
    self.assertTrue(expert.fixture_close_requires_free_edge_bypass("microwave_microjoint"))
    self.assertFalse(expert.fixture_close_requires_free_edge_bypass("fridge_door_joint"))
    self.assertFalse(expert.fixture_close_requires_free_edge_bypass("drawer_slide_joint"))
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn.VLA82SourceSpawnTests.test_microwave_bypass_uses_edge_farthest_from_hinge tools.tests.test_vla82_source_spawn.VLA82SourceSpawnTests.test_microwave_bypass_rotates_with_the_live_panel tools.tests.test_vla82_source_spawn.VLA82SourceSpawnTests.test_free_edge_bypass_is_microwave_only
```

Expected: FAIL because `fixture_hinge_free_edge_waypoints` and `fixture_close_requires_free_edge_bypass` do not exist.

- [ ] **Step 3: Implement the minimal pure helpers**

Add helpers with this contract:

```python
def fixture_close_requires_free_edge_bypass(joint_id: str) -> bool:
    return "microjoint" in str(joint_id).lower()

def fixture_hinge_free_edge_waypoints(
    *, panel_center: Sequence[float], panel_rotation: Sequence[float],
    panel_half_size: Sequence[float], hinge_anchor: Sequence[float],
    eef_world: Sequence[float], push_direction: Sequence[float],
    edge_clearance: float = .09, side_clearance: float = .06,
) -> dict[str, np.ndarray]:
    center = np.asarray(panel_center, dtype=float)
    rotation = np.asarray(panel_rotation, dtype=float).reshape(3, 3)
    size = np.asarray(panel_half_size, dtype=float)
    hinge = np.asarray(hinge_anchor, dtype=float)
    eef = np.asarray(eef_world, dtype=float)
    push = np.asarray(push_direction, dtype=float)
    push_norm = float(np.linalg.norm(push))
    if push_norm <= 1e-9 or np.any(size <= 1e-9):
        raise ValueError("microwave bypass geometry is degenerate")
    push /= push_norm
    normal_index = int(np.argmin(size))
    in_plane = tuple(index for index in range(3) if index != normal_index)
    edge_index = max(
        in_plane,
        key=lambda index: float(size[index] * np.linalg.norm(rotation[:2, index])),
    )
    edge_axis = rotation[:, edge_index]
    candidates = (
        center + edge_axis * size[edge_index],
        center - edge_axis * size[edge_index],
    )
    free_edge = max(candidates, key=lambda point: float(np.linalg.norm(point - hinge)))
    outward_edge = free_edge - center
    outward_edge /= max(float(np.linalg.norm(outward_edge)), 1e-9)
    outside = free_edge + outward_edge * float(edge_clearance)
    current_sign = 1.0 if float(np.dot(eef - center, push)) >= 0.0 else -1.0
    return {
        "free_edge": free_edge.copy(),
        "outside_current_side": outside + current_sign * push * float(side_clearance),
        "outside_closing_side": outside - push * float(side_clearance),
        "contact_prepoint": center - push * float(side_clearance),
    }
```

The implementation must return fresh NumPy arrays, preserve the panel contact height, and raise `ValueError` for a degenerate push direction or unusable panel dimensions.

- [ ] **Step 4: Run targeted tests and verify GREEN**

Run the Step 2 command again.

Expected: the runner reports three tests and `OK`.

- [ ] **Step 5: Run the existing source-spawn regression**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
```

Expected: all tests pass; baseline was 114 tests, so the new total must be at least 117.

---

### Task 2: Microwave-only bypass state machine

**Files:**
- Modify: `tools/vla82_full_sim/expert.py:1103-1265`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Consumes: `fixture_close_requires_free_edge_bypass(joint_id)` and the latched waypoint dictionary from Task 1.
- Produces: `fixture_close_bypass_stage(joint_closed=, free_edge_distance=, cross_plane_distance=, precontact_distance=, exact_contact=, tolerance=) -> str` and microwave stages `bypass_to_free_edge`, `bypass_cross_plane`, `bypass_return_to_panel`, `seat_contact`, then the existing `push_close` / `settle_closed` stages.

- [ ] **Step 1: Write failing state-transition tests**

Add a table-driven test that asserts:

```python
cases = (
    ((False, .20, .20, .20, False), "bypass_to_free_edge"),
    ((False, .01, .20, .20, False), "bypass_cross_plane"),
    ((False, .01, .01, .20, False), "bypass_return_to_panel"),
    ((False, .01, .01, .01, False), "seat_contact"),
    ((False, .01, .01, .01, True), "push_close"),
    ((True, .01, .01, .01, True), "settle_closed"),
)
for arguments, expected in cases:
    self.assertEqual(expert.fixture_close_bypass_stage(
        joint_closed=arguments[0], free_edge_distance=arguments[1],
        cross_plane_distance=arguments[2], precontact_distance=arguments[3],
        exact_contact=arguments[4], tolerance=.025,
    ), expected)
```

Also assert that losing exact contact after `push_close` returns `seat_contact`, not another empty push.

- [ ] **Step 2: Run the new test and verify RED**

Run the exact new unittest method.

Expected: FAIL because `fixture_close_bypass_stage` does not exist.

- [ ] **Step 3: Implement the pure transition helper**

Implement the ordered, contact-gated transition logic from the table. It must test `joint_closed` first and `exact_contact` only after all three waypoint distances are within tolerance.

- [ ] **Step 4: Integrate the state machine into `DrawerClosePrimitive`**

For a microwave hinge only:

- Resolve `handle_id` as the chosen contactable panel and read its `geom_xmat` and `geom_size`.
- Compute the initial closing tangent and latch one waypoint dictionary before the action loop.
- Track three completion booleans so a later waypoint cannot cause regression to an earlier stage merely because the arm has moved away from it.
- Target the active waypoint with fixed base and a closed/stable gripper command.
- On `seat_contact`, target the panel contact point through `fixture_close_wrist_target_for_pad_contact`.
- On real contact, reuse the live `fixture_close_push_direction`, short hinge step, and existing strict close threshold.
- Disable `fixture_approach_base_local` during all bypass and microwave contact stages.
- Preserve the current code path unchanged for every non-microwave joint.

- [ ] **Step 5: Add diagnostic trace fields**

Each microwave trace row must add:

```python
{
    "bypass_required": True,
    "bypass_free_edge_world": np.asarray(bypass_points["free_edge"]).round(6).tolist(),
    "bypass_outside_current_side_world": np.asarray(bypass_points["outside_current_side"]).round(6).tolist(),
    "bypass_outside_closing_side_world": np.asarray(bypass_points["outside_closing_side"]).round(6).tolist(),
    "bypass_contact_prepoint_world": np.asarray(bypass_points["contact_prepoint"]).round(6).tolist(),
    "bypass_free_edge_complete": bool(bypass_free_edge_complete),
    "bypass_cross_plane_complete": bool(bypass_cross_plane_complete),
    "bypass_return_complete": bool(bypass_return_complete),
    "push_direction_world": np.asarray(push_direction).round(6).tolist(),
}
```

Non-microwave rows may use `bypass_required: false` and omit waypoint values.

- [ ] **Step 6: Run targeted and full unit regressions**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest -q tools/tests/test_vla82_pure_closed_loop.py tools/tests/test_vla82_predicate_stability.py
```

Expected: source-spawn suite all passes and pure suites report `5 passed` or more.

---

### Task 3: Fresh VLA82-029 physical simulation and evidence diagnosis

**Files:**
- Regenerate: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-029/episode-2000.json`
- Regenerate on successful collection: matching `episode-2000.npz`
- Inspect: the generated trace and contact details in the JSON/NPZ artifacts

**Interfaces:**
- Consumes: the Task 2 microwave bypass controller.
- Produces: fresh MuJoCo physical evidence for strict acceptance or one evidence-backed next repair.

- [ ] **Step 1: Run one fresh physical attempt**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py collect-demos --episodes-per-object 1 --max-attempts-per-object 1 --selection-id VLA82-029
```

Expected success: process exits 0 and the result JSON reports `status: PASS`, `predicate_success: true`, and no errors. A FAIL is diagnostic evidence, not completion.

- [ ] **Step 2: Audit geometry and physical causality**

Parse the fresh trace and verify:

- stages occur in the intended order;
- each bypass stage contains multiple frames and reaches its latched point;
- no door contact occurs before `seat_contact`;
- `push_close` contains a continuous exact-contact segment longer than one frame;
- the joint value moves monotonically away from approximately `-1.5708 rad` toward `0 rad` during useful contact;
- no direct-state mutation or predicate override appears in the run.

- [ ] **Step 3: Apply only evidence-backed repairs if needed**

If the run fails, identify the first failed invariant. Add a failing unit test for the smallest controllable cause, change one parameter or behavior, rerun regressions, then perform one new physical attempt. Continue while each round supplies a distinct controllable cause or measurable improvement; stop when further changes no longer add diagnostic information or fixed-base reachability is proven insufficient.

---

### Task 4: Strict-PASS video and index update, conditional on success

**Files:**
- Create only on strict success: `outputs/strict_pass_video_index_20260811/VLA82-029/episode-2000.strict-pass-microwave-close-720p.mp4`
- Update only on strict success: the existing strict-PASS workbook under `outputs/strict_pass_video_index_20260811/`
- Verify: generated replay metadata and source hashes

**Interfaces:**
- Consumes: a fresh Task 3 episode whose JSON and NPZ both satisfy strict PASS.
- Produces: one unobstructed 1280×720 evidence video and a verified absolute-path workbook entry.

- [ ] **Step 1: Reject export unless the source is strict PASS**

Run the existing exporter/replayer only after checking JSON `status`, `predicate_success`, errors, contact continuity, and hinge closure. The tool's own `require_strict_pass` guard must remain enabled.

- [ ] **Step 2: Generate the high-definition result video**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\replay_vla82_strict_pass_video.py --fps 24 --frame-stride 2 --width 1280 --height 720 outputs\midterm_testing_vla82\full_simulation_objectwise_8x60\expert_demos\VLA82-029\episode-2000.npz outputs\strict_pass_video_index_20260811\VLA82-029\episode-2000.strict-pass-microwave-close-720p.mp4
```

Expected: the video and metadata are created and source hashes match the strict-PASS episode.

- [ ] **Step 3: Visually verify the video**

Inspect start, bypass, contact, mid-close, and final frames. Confirm upright orientation, clear robot/door visibility, no obstruction, no discontinuity, and a stably closed final state.

- [ ] **Step 4: Update and validate the workbook path**

Write the exact existing absolute MP4 path to the VLA82-029 row without changing other rows. Reopen the workbook programmatically and assert that the path exists and points to the verified MP4.

- [ ] **Step 5: Run final regressions**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest -q tools/tests/test_vla82_pure_closed_loop.py tools/tests/test_vla82_predicate_stability.py tools/tests/test_audit_vla82_video_mapping.py tools/tests/test_replay_vla82_strict_pass_video.py
```

Expected: every suite passes. Report VLA82-029 as strict PASS only if these tests, the physical predicate, the trace audit, and the visual audit all pass.
