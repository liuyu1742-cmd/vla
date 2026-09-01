# VLA82-029 Raise Base Assist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use at most 50 mm of collision-gated mobile-base motion to free the arm from its raise-stage workspace plateau, then lock the base for the remainder of VLA82-029.

**Architecture:** Add one pure trigger/direction helper driven by stage, distance history, target delta, live base rotation, cumulative displacement, and collision state. Integrate it only into `raise_above_handle`; retain normal arm motion and all upper-panel stages.

**Tech Stack:** Python 3.11, NumPy, `unittest`, `pytest`, RoboCasa, robosuite, MuJoCo.

## Global Constraints

- Require at least 60 raise frames and a 40-frame improvement below `0.003 m` before assisting.
- Assist only while target distance exceeds `0.025 m`.
- Command magnitude is `0.08`; measured base displacement must remain below `0.05 m`.
- Material mobile-base/fixed-fixture contact at distance `<= -0.001 m` permanently locks assistance for that episode.
- Base is fixed outside `raise_above_handle`, including traverse, seat, push, and reseat.
- Do not lower the safe target, relax the stage tolerance, write state, disable collisions, or alter strict predicates.
- Enable only in the VLA82-029 microwave free-edge branch.
- This workspace is not a Git repository; use atomic patches and passing-test checkpoints.

---

### Task 1: Pure assist trigger and local direction

**Files:**
- Modify: `tools/vla82_full_sim/expert.py:700-1000`
- Test: `tools/tests/test_vla82_source_spawn.py:1500-1620`

**Interface:**
- Produces `fixture_raise_base_assist_local(stage=, stage_frames=, distance_history=, target_delta_world=, base_rotation=, cumulative_displacement=, collision=, locked=, min_stage_frames=60, history_size=40, plateau_improvement=.003, target_tolerance=.025, max_displacement=.05, command_magnitude=.08) -> np.ndarray | None`.

- [ ] **Step 1: Write failing tests**

Assert no command before 60 frames, while 40-frame improvement is at least `.003`, at target tolerance, at displacement cap, on collision/lock, outside raise stage, and for zero planar error. Assert that a plateau history from `.0290` to `.0278` returns a norm-`.08` command and that a 90-degree base rotation converts world +X to local −Y.

- [ ] **Step 2: Verify RED**

Run the exact new unittest methods. Expected: missing-helper error.

- [ ] **Step 3: Implement the pure helper**

Validate the stage and gates in the order above, require at least 40 history values, compute improvement as `history[-40] - history[-1]`, normalize the planar world delta, transform through `base_rotation[:2, :2].T`, normalize again, and multiply by `.08`.

- [ ] **Step 4: Verify GREEN and full unit regression**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
```

Expected: all tests pass; baseline is 129 tests.

---

### Task 2: Controller integration, collision lock, and trace

**Files:**
- Modify: `tools/vla82_full_sim/expert.py:1300-1650`
- Test: `tools/tests/test_vla82_source_spawn.py`

- [ ] **Step 1: Add raise-stage state**

Maintain `raise_stage_frames`, a list capped to the latest 40 distances, `base_assist_start_world`, `base_assist_displacement`, `base_assist_locked`, and the current material collision flag. Reset frame/history counters before the first raise frame; never reset the displacement or collision lock during the episode.

- [ ] **Step 2: Reuse live base pose and collision semantics**

Read both position and rotation from `base_controller.get_base_pose()`. Scan current MuJoCo contacts and call existing `is_material_mobile_base_fixture_contact(names, distance=contact.dist)`. When true, set the permanent lock before selecting a base command.

- [ ] **Step 3: Apply the assist only during raise**

Call Task 1 with `upper_panel_points["raise"] - contact_reference`. When a command first becomes available, latch the current base world position as the assist start. Recompute actual displacement each frame. Write the two local components and `base_mode=1` only for a valid assist command; otherwise preserve fixed base mode. Once stage advances, no further base command is permitted.

- [ ] **Step 4: Add diagnostics**

Record stage frames, distance-window endpoints/improvement, assist eligibility, local command, assist start/current base world positions, actual displacement, cap reached, collision flag, and permanent lock.

- [ ] **Step 5: Run all regressions**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m compileall -q tools/vla82_full_sim/expert.py tools/tests/test_vla82_source_spawn.py
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest -q tools/tests/test_vla82_pure_closed_loop.py tools/tests/test_vla82_predicate_stability.py
```

Expected: source-spawn suite all green and at least `5 passed` in pure suites.

---

### Task 3: Fresh physical verification

**Files:**
- Regenerate: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-029/episode-2000.json`
- Regenerate on strict success: matching NPZ, strict-PASS video, and workbook entry.

- [ ] **Step 1: Run one physical attempt**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py collect-demos --episodes-per-object 1 --max-attempts-per-object 1 --selection-id VLA82-029
```

- [ ] **Step 2: Audit the assist**

Require measured displacement `<= .05`, no material base collision, raise distance `<= .025`, transition to `traverse_to_upper_panel`, and fixed base thereafter. Then audit broad-panel contact continuity and hinge motion.

- [ ] **Step 3: Follow evidence stop conditions**

If cap or collision prevents raise completion, stop base-assist changes and redesign a fixed-base reachable panel path. If raise succeeds, continue diagnosing only the first later failed invariant. Export or index video only after complete strict PASS.
