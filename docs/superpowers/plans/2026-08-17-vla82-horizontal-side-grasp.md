# VLA82 Horizontal Side-Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a physically valid horizontal-wrist side-grasp controller, prove it first on VLA82-018, and preserve a strict-PASS trajectory and unobstructed result video before reusing it on VLA82-055.

**Architecture:** Extend the existing pick/place state machine with two explicit orientation gates: tool-axis alignment before approach and finger-opening-axis alignment at the side pre-grasp point. Reuse live MuJoCo geometry, named native gripper pads, and the existing strict predicate instead of modifying scene state or weakening acceptance criteria.

**Tech Stack:** Python 3.11, NumPy, RoboCasa/RoboSuite, MuJoCo, `unittest`, `pytest`, existing VLA82 collection/replay tools.

## Global Constraints

- Do not modify object spawn pose, target pose, mass, dimensions, or strict success predicate.
- Do not use friction changes to compensate for an incorrect grasp orientation.
- A strict PASS requires physical object contact, RoboSuite native grasp, opposing native-pad contact, stable transport, stable release, `predicate_success=true`, and an empty error list.
- A failed or visually plausible trajectory must not be exported as a result video.
- Stop retrying a difficult object after at most 5 complete simulation attempts; record the actual failure and defer it.
- Preserve the current 11 strict PASS records and their video mappings.
- This workspace has no `.git` directory, so the commit steps below become explicit diff-and-test checkpoints rather than Git commits.

---

### Task 1: Pure horizontal side-grasp geometry and state gates

**Files:**
- Modify: `tools/vla82_full_sim/expert.py:5674-5755`
- Test: `tools/tests/test_vla82_source_spawn.py:660-725`

**Interfaces:**
- Consumes: `fixture_handle_tool_alignment_command(...)`, MuJoCo world-space pad positions, object center/rotation/half-size.
- Produces: `gripper_opening_axis_world_from_named_pads(named_positions) -> np.ndarray`, `horizontal_side_grasp_tool_command(...) -> tuple[np.ndarray, float]`, `horizontal_side_grasp_roll_command(...) -> tuple[np.ndarray, float]`, and `horizontal_side_orientation_transition(...) -> str`.

- [ ] **Step 1: Add failing geometry and gate tests**

```python
def test_box_drink_selects_horizontal_side_grasp(self) -> None:
    from tools.vla82_full_sim import expert
    self.assertTrue(expert.pick_place_uses_pad_center_alignment("VLA82-018"))
    self.assertEqual(expert.pick_place_grasp_strategy("VLA82-018"), "horizontal_side")
    self.assertEqual(expert.pick_place_retry_grasp_state("VLA82-018"), "side_orient")

def test_horizontal_side_tool_command_rotates_downward_tool_toward_box_side(self) -> None:
    from tools.vla82_full_sim import expert
    command, error = expert.horizontal_side_grasp_tool_command(
        tool_axis_world=(0., 0., -1.),
        object_center=(0., 0., 0.),
        side_entry=(1., 0., 0.),
        base_rotation=np.eye(3),
    )
    self.assertGreater(error, 1.5)
    self.assertGreater(command[1], .49)

def test_horizontal_side_orientation_gate_requires_both_axes(self) -> None:
    from tools.vla82_full_sim import expert
    self.assertEqual(expert.horizontal_side_orientation_transition(.20, .02), "side_orient")
    self.assertEqual(expert.horizontal_side_orientation_transition(.02, .20), "side_axis_align")
    self.assertEqual(expert.horizontal_side_orientation_transition(.02, .02), "side_descend")
```

- [ ] **Step 2: Run the targeted tests and confirm RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_box_drink_selects_horizontal_side_grasp tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_horizontal_side_tool_command_rotates_downward_tool_toward_box_side tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_horizontal_side_orientation_gate_requires_both_axes
```

Expected: FAIL because the new helper functions and `horizontal_side` strategy do not exist.

- [ ] **Step 3: Add minimal pure geometry helpers**

```python
def gripper_opening_axis_world_from_named_pads(named_positions):
    positions = {str(k).lower(): np.asarray(v, dtype=float) for k, v in named_positions.items()}
    paired = [v for k, v in positions.items() if "f1_pad_collision" in k or "f2_pad_collision" in k]
    middle = [v for k, v in positions.items() if "finger_middle_pad_collision" in k]
    if paired and middle:
        axis = np.mean(middle, axis=0) - np.mean(paired, axis=0)
    else:
        values = list(positions.values())
        axis = values[1] - values[0]
    return axis / max(float(np.linalg.norm(axis)), 1e-9)

def horizontal_side_grasp_tool_command(*, tool_axis_world, object_center, side_entry, base_rotation):
    outward = np.asarray(side_entry, dtype=float) - np.asarray(object_center, dtype=float)
    return fixture_handle_tool_alignment_command(
        tool_axis_world=tool_axis_world,
        outward_normal_world=outward,
        base_rotation=base_rotation,
        tolerance=np.deg2rad(6.),
        max_command=.55,
    )

def horizontal_side_orientation_transition(tool_error, opening_error):
    if float(tool_error) > np.deg2rad(6.):
        return "side_orient"
    if float(opening_error) > np.deg2rad(6.):
        return "side_axis_align"
    return "side_descend"
```

Implement `horizontal_side_grasp_roll_command` with a signed rotation about the aligned tool axis so the native-pad opening axis reaches the object's narrow horizontal axis without changing the tool-axis direction.

- [ ] **Step 4: Run the targeted tests and confirm GREEN**

Expected: all three tests PASS.

- [ ] **Step 5: Run the full source-spawn suite as the checkpoint**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
```

Expected: all existing tests plus the new tests PASS.

---

### Task 2: Integrate orientation states into the closed-loop pick/place controller

**Files:**
- Modify: `tools/vla82_full_sim/expert.py:1778-1793`
- Modify: `tools/vla82_full_sim/expert.py:2913-3290`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Consumes: all helpers from Task 1, `selected_gripper_pad_center(raw)`, `action_for(...)`, `pick_place_side_entry_point(...)`.
- Produces: VLA82-018 state flow `cabinet_front_stage -> side_orient -> side_preapproach -> side_axis_align -> side_descend -> side_center -> close -> secure`.

- [ ] **Step 1: Add failing transition and lock tests**

```python
def test_box_drink_leaves_cabinet_stage_for_side_orientation(self) -> None:
    from tools.vla82_full_sim import expert
    self.assertEqual(expert.pick_place_post_cabinet_stage_state("VLA82-018"), "side_orient")
    self.assertEqual(expert.pick_place_post_cabinet_stage_state("VLA82-017"), "cabinet_front_insert")

def test_box_drink_close_locks_wrist_after_first_side_contact(self) -> None:
    from tools.vla82_full_sim import expert
    eef = np.array((.50, -4.00, .95))
    tracking = np.array((.40, -4.00, .95))
    np.testing.assert_allclose(
        expert.pick_place_close_target("VLA82-018", eef=eef, tracking_target=tracking), eef
    )
```

- [ ] **Step 2: Run both tests and confirm RED**

Expected: FAIL because the post-cabinet helper is absent and VLA82-018 does not yet lock the close target.

- [ ] **Step 3: Implement the state-machine integration**

Add VLA82-018 to native pad-center alignment, return `horizontal_side` from its grasp strategy, return `side_orient` on retry, and lock its wrist in `pick_place_close_target`.

In the controller loop, collect named pad positions once per step. In `side_orient`, compute the live side entry and tool axis (`pad_center - eef`), issue rotation only at the cabinet-front safety pose, and remain in that state until tool error is within 6 degrees. In `side_preapproach`, translate to the outside side point. In `side_axis_align`, keep position fixed and roll until the 3-D opening axis aligns with the object's narrow horizontal axis. Only then allow `side_descend` and `side_center`.

Trace these fields every step for diagnosis: `side_entry_world`, `tool_axis_world`, `target_tool_axis_world`, `tool_alignment_error_rad`, `opening_axis_world`, `target_opening_axis_world`, `opening_alignment_error_rad`, `raw_grasp`, and `two_pad_contact`.

- [ ] **Step 4: Run the two new tests and confirm GREEN**

Expected: both tests PASS.

- [ ] **Step 5: Run controller regression tests**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest -q tools/tests/test_vla82_pure_closed_loop.py tools/tests/test_vla82_predicate_stability.py
```

Expected: all tests PASS; no existing bottle, drawer, fixture, or side-entry behavior regresses.

---

### Task 3: Run VLA82-018 physical closed-loop attempts and diagnose bounded failures

**Files:**
- Regenerate: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-018/episode-2000.json`
- Regenerate on success: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-018/episode-2000.npz`

**Interfaces:**
- Consumes: controller from Task 2 and existing `collect-demos` command.
- Produces: either a verified strict-PASS episode or a bounded diagnostic explaining the first failed physical gate.

- [ ] **Step 1: Run one fresh VLA82-018 attempt**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py collect-demos --episodes-per-object 1 --max-attempts-per-object 1 --selection-id VLA82-018
```

Expected success: `episode-2000.json` reports `status: PASS`, `predicate_success: true`, and `errors: []`.

- [ ] **Step 2: Audit the physical gates from the JSON trace**

Verify that the trace contains a completed orientation gate, native-pad contacts, `raw_grasp=true`, `two_pad_contact=true`, lift, transit, release, and settle. Reject shell-only contacts or a trajectory that reaches place states without a valid grasp.

- [ ] **Step 3: If needed, make one root-cause change per rerun**

Use the first failed gate to choose exactly one correction:

- orientation timeout: correct local/world rotation conversion;
- no pad contact: adjust side clearance or the pad-center target, not friction;
- one-sided contact: correct narrow-axis roll sign or freeze the side-center target at first contact;
- post-lift slip: reduce lift/transit acceleration only after native grasp and two-pad contact were proven;
- release instability: reduce final lowering/release speed without weakening the target predicate.

Run at most 5 complete VLA82-018 attempts. After the fifth failure, stop and preserve the best diagnostic rather than continuing retries.

- [ ] **Step 4: Re-run all regressions after the successful correction**

Expected: source-spawn, pure-closed-loop, predicate-stability, and video-mapping tests all PASS.

---

### Task 4: Export and inspect strict-PASS evidence

**Files:**
- Create on success: `outputs/strict_pass_video_index_20260811/VLA82-018/episode-2000.strict-pass-horizontal-side-grasp-720p.mp4`
- Update on success: strict-PASS video index workbook under `outputs/strict_pass_video_index_20260811/`

**Interfaces:**
- Consumes: a strict-PASS VLA82-018 JSON/NPZ pair from Task 3.
- Produces: a clear 1280x720 replay video and a workbook row whose absolute path exists.

- [ ] **Step 1: Replay the strict episode with the audit camera**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\replay_vla82_strict_pass_video.py --fps 24 --frame-stride 2 --width 1280 --height 720 outputs\midterm_testing_vla82\full_simulation_objectwise_8x60\expert_demos\VLA82-018\episode-2000.npz outputs\strict_pass_video_index_20260811\VLA82-018\episode-2000.strict-pass-horizontal-side-grasp-720p.mp4
```

Expected: the replay tool accepts the source strict PASS and writes a non-empty MP4.

- [ ] **Step 2: Inspect beginning, grasp, lift, transit, and final frames**

Confirm the robot, object, source fixture, grasp, full transfer, and target placement remain visible; reject inverted, obstructed, truncated, or duplicate-looking footage.

- [ ] **Step 3: Update and verify the workbook**

Add one VLA82-018 row only after the video passes visual inspection. Store the existing absolute MP4 path, strict JSON path, predicate result, resolution, and generation time. Open the workbook with the spreadsheet tooling and verify formulas, formatting, and every listed path.

- [ ] **Step 4: Run evidence audits**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest -q tools/tests/test_audit_vla82_video_mapping.py tools/tests/test_replay_vla82_strict_pass_video.py
```

Expected: all evidence audits PASS.

---

### Task 5: Reuse the proven controller on VLA82-055

**Files:**
- Modify only if geometry classification requires it: `tools/vla82_full_sim/expert.py`
- Test: `tools/tests/test_vla82_source_spawn.py`
- Create on success: VLA82-055 strict JSON/NPZ and clear video under the same evidence roots.

**Interfaces:**
- Consumes: the verified horizontal-side controller and strict evidence workflow.
- Produces: a VLA82-055 result without copying object-specific coordinates from VLA82-018.

- [ ] **Step 1: Add a failing strategy-selection test for VLA82-055**

Assert VLA82-055 uses the horizontal-side strategy only if its live half-extents fit the same grasp affordance: horizontal narrow width below gripper aperture and vertical extent too large for a top-side pinch.

- [ ] **Step 2: Implement geometry-based reuse without hard-coded trajectory points**

Reuse the live object rotation, half-extents, nearest-side selection, orientation gates, and native-pad contact gates. Keep only the selection policy object-specific.

- [ ] **Step 3: Run the full unit/regression suite**

Expected: all tests PASS.

- [ ] **Step 4: Run at most 5 complete VLA82-055 attempts**

Use the same one-root-cause-per-rerun rule and the same strict PASS conditions.

- [ ] **Step 5: On success, export, visually inspect, and index the video**

Do not add VLA82-055 to the strict workbook if any physical gate or visibility check fails.
