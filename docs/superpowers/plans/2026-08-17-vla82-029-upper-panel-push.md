# VLA82-029 Upper-Panel Push Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the Panda above the protruding microwave handle and establish continuous contact on the broad upper door panel before closing the hinge.

**Architecture:** Select the live thin, broad non-handle box from the microwave door joint, compute latched raise/precontact/face points from its OBB, and replace the post-cross free-edge return with raise, traverse, seat, push, and short reseat stages. Only broad panel contacts may trigger push.

**Tech Stack:** Python 3.11, NumPy, `unittest`, `pytest`, RoboCasa, robosuite, MuJoCo.

## Global Constraints

- Preserve the verified `bypass_to_free_edge` and `bypass_cross_plane` stages.
- Select panel geometry from live model type/size data; do not hardcode `microwave_main_group_1_g12`.
- Target 45% of panel half-height above center, at least `0.07 m` above the handle and `0.08 m` below the panel top.
- Use `0.06 m` precontact clearance and `0.006 m` short reseat distance.
- Handle and frame contacts must not trigger `push_close`.
- Keep the mobile base fixed and do not write joint/body state, disable collision, or weaken strict predicates.
- Enable only for the microwave free-edge bypass branch; preserve other fixture behavior.
- Export video and update the strict-PASS workbook only after fresh strict physical success.
- The workspace is not a Git repository; use atomic `apply_patch` changes and test checkpoints instead of commits.

---

### Task 1: Broad-panel selection and upper-path geometry

**Files:**
- Modify: `tools/vla82_full_sim/expert.py:790-980`
- Test: `tools/tests/test_vla82_source_spawn.py:1320-1520`

**Interfaces:**
- Produces `fixture_close_panel_geom_name(candidate_names, geom_types, geom_sizes) -> str`.
- Produces `fixture_close_upper_panel_waypoints(panel_center=, panel_rotation=, panel_half_size=, handle_world=, push_direction=, outside_closing_side=, vertical_fraction=.45, precontact_clearance=.06, min_handle_clearance=.07, top_clearance=.08) -> dict[str, np.ndarray]` with `raise`, `precontact`, `face`, and `vertical_axis`.

- [ ] **Step 1: Write failing tests**

Test that a thin broad non-handle box is selected over a handle and thick frame:

```python
selected = expert.fixture_close_panel_geom_name(
    candidate_names=("door_frame", "door_handle_main", "door_panel"),
    geom_types={"door_frame": 6, "door_handle_main": 6, "door_panel": 6},
    geom_sizes={
        "door_frame": (.25, .04, .18),
        "door_handle_main": (.18, .01, .015),
        "door_panel": (.234, .0028, .165),
    },
)
self.assertEqual(selected, "door_panel")
```

Test synthetic panel geometry with center `(0, 0, 1.3)`, identity rotation, half-size `(.234, .0028, .165)`, handle `(−.234, 0, 1.3)`, push `(0, −1, 0)`, and outside point `(−.33, .06, 1.3)`. Assert face height is at least `1.37`, top gap at least `.08`, precontact is `.06` behind the face, and raise/precontact have equal height. Add a rotated-panel case and an invalid-clearance case that raises `ValueError`.

- [ ] **Step 2: Verify RED**

Run the exact new unittest methods. Expected: missing-helper errors.

- [ ] **Step 3: Implement minimal geometry helpers**

For panel selection, filter MuJoCo box type `6`, exclude names containing `handle`, compute sorted half-sizes, and select `min((thickness, -area, name))`. For waypoints, identify the smallest-size normal axis, choose the remaining axis with greatest absolute world-Z component as vertical, orient it upward, calculate projected thickness along the normalized push direction, then calculate face and precontact exactly as the design specifies. Validate handle and top clearances before returning points.

- [ ] **Step 4: Run targeted and complete unit tests**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
```

Expected: all tests pass; baseline is 124 tests.

---

### Task 2: Upper-panel state machine and controller integration

**Files:**
- Modify: `tools/vla82_full_sim/expert.py:1260-1560`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Produces `fixture_close_upper_panel_stage(joint_closed=, raised=, traversed=, exact_panel_contact=, panel_contact_seen=) -> str`.
- Consumes Task 1 panel name/waypoints and existing contact-normal helpers.

- [ ] **Step 1: Write failing transition tests**

Assert the exact sequence:

```python
cases = (
    ((False, False, False, False, False), "raise_above_handle"),
    ((False, True, False, False, False), "traverse_to_upper_panel"),
    ((False, True, True, False, False), "seat_upper_panel"),
    ((False, True, True, True, True), "push_close"),
    ((False, True, True, False, True), "reseat_upper_panel"),
    ((True, True, True, True, True), "settle_closed"),
)
```

Add a test showing a handle-only contact leaves `exact_panel_contact=False` and cannot enter `push_close`.

- [ ] **Step 2: Verify RED, then implement the pure transition helper**

Run the new transition test, observe the missing-helper error, implement the ordered conditions, and rerun until green.

- [ ] **Step 3: Integrate live panel geometry**

During microwave setup, select `panel_name` from joint-bound box geoms, resolve `panel_id`, and define `panel_contact_geoms` as the selected panel plus non-handle semantic names ending in `door_main`. After the existing cross-plane waypoint is reached, latch Task 1 upper-panel waypoints using the live panel OBB, semantic handle position, push direction, and outside-closing-side point.

- [ ] **Step 4: Integrate upper stages and contact gates**

Track `upper_raise_complete`, `upper_traverse_complete`, and `upper_panel_contact_seen`. Use pad/contact-reference distance with `.025 m` tolerance. Target `raise`, `precontact`, `face`, the existing contact-preserving push direction, or the `0.006 m` last-normal reseat according to stage. Compute `exact_panel_contact` and control normals only from `panel_contact_geoms`; retain all door contacts separately for diagnostics.

- [ ] **Step 5: Add trace fields**

Record panel name/OBB, all upper waypoints, completion flags, exact panel contact, invalid handle/frame contacts, continuous panel-contact steps, hinge angle, original tangent, filtered normal, bias, and final direction.

- [ ] **Step 6: Run regressions**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m compileall -q tools/vla82_full_sim/expert.py tools/tests/test_vla82_source_spawn.py
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest -q tools/tests/test_vla82_pure_closed_loop.py tools/tests/test_vla82_predicate_stability.py
```

Expected: all source-spawn tests pass and pure suites report at least `5 passed`.

---

### Task 3: Fresh physical verification and conditional publication

**Files:**
- Regenerate: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-029/episode-2000.json`
- Regenerate on success: matching `episode-2000.npz`
- Create only on strict success: `outputs/strict_pass_video_index_20260811/VLA82-029/episode-2000.strict-pass-microwave-close-720p.mp4`
- Update only on strict success: existing strict-PASS workbook under `outputs/strict_pass_video_index_20260811/`

- [ ] **Step 1: Run one fresh physical attempt**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py collect-demos --episodes-per-object 1 --max-attempts-per-object 1 --selection-id VLA82-029
```

- [ ] **Step 2: Audit physical causality**

Require ordered free-edge/raise/traverse/seat stages, no handle contact accepted as push, broad-panel contact longer than the prior one-frame maximum, and hinge motion toward zero. If an early handle or top-edge collision occurs, use its target/actual height to make the one permitted evidence-backed height correction, rerun regressions, and perform one new physical attempt.

- [ ] **Step 3: Stop or continue from new evidence**

If the corrected upper path still contacts the handle/edge, stop this path and evaluate a hinge-side broad-panel target. If panel contact becomes continuous but closure stalls, investigate lever arm and reachability. Do not return to force-coefficient tuning.

- [ ] **Step 4: Publish only after strict PASS**

Require JSON `status: PASS`, `predicate_success: true`, no errors, stable strict closure, consistent trace/NPZ, and an unobstructed upright visual result before replaying the 1280×720 video and updating the workbook absolute path.
