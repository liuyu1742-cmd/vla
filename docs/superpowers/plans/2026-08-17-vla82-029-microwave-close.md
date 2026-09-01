# VLA82-029 Microwave Door Close Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct hinge close tangents so VLA82-029 moves from the open joint limit toward zero under real robot contact, then export and index a strict-PASS video if the physical predicate succeeds.

**Architecture:** Keep the existing `DrawerClosePrimitive` contact, orientation, approach, push, and settle pipeline. Change only the pure hinge tangent helper first; subsequent changes are allowed only when a fresh physical trace identifies a different first failing gate.

**Tech Stack:** Python 3.11, NumPy, RoboCasa/RoboSuite, MuJoCo, `unittest`, `pytest`, strict replay tooling.

## Global Constraints

- Never write fixture joint state directly or weaken the closed threshold/predicate.
- Require exact moving-door contact before each push.
- Change one root-cause variable per simulation round.
- Continue while a new evidence-backed controllable cause exists; stop only when reruns repeat without new controllable information or require invalid shortcuts.
- Existing strict-PASS videos and workbook rows must not be overwritten.
- This workspace has no `.git`; use test and artifact checkpoints instead of commits.

---

### Task 1: Correct the pure hinge close tangent

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py:1168-1205`
- Modify: `tools/vla82_full_sim/expert.py:760-795`

- [ ] Update the existing positive-angle hinge test to expect the model-space point velocity toward zero.
- [ ] Add a negative-angle microwave test: Z axis, X lever, negative joint value must produce positive Y direction.
- [ ] Run both tests and verify RED against the existing extra-negation formula.
- [ ] Replace `-sign_to_zero * cross(axis, lever)` with `sign_to_zero * cross(axis, lever)` for hinge joints only.
- [ ] Run both tests and verify GREEN.
- [ ] Run `python -m unittest tools.tests.test_vla82_source_spawn` and expect all tests to pass.

### Task 2: Run a fresh VLA82-029 physical round

- [ ] Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py collect-demos --episodes-per-object 1 --max-attempts-per-object 1 --selection-id VLA82-029
```

- [ ] Verify exact contact occurs before `push_close`.
- [ ] Compare first/last/min/max joint values during push. The value must increase from approximately `-1.57` toward `0`.
- [ ] If strict PASS, proceed to Task 4. If not, identify only the first new failed physical gate.

### Task 3: Evidence-driven continuation

- [ ] Write a failing unit test for the newly identified gate before changing production code.
- [ ] Apply one minimal correction and run all source-spawn, pure-closed-loop, and predicate-stability tests.
- [ ] Rerun VLA82-029 and compare the new trace to the previous round.
- [ ] Repeat only if the trace exposes new controllable information; otherwise stop and report the preserved diagnostic.

### Task 4: Strict evidence export

- [ ] Verify JSON has `status=PASS`, `predicate_success=true`, `errors=[]`, and the NPZ exists.
- [ ] Replay at 1280×720 to `outputs/strict_pass_video_index_20260811/VLA82-029/episode-2000.strict-pass-microwave-close-720p.mp4`.
- [ ] Inspect opening position, physical contact, continuous closing motion, final closed state, visibility, orientation, and resolution.
- [ ] Update the strict-PASS workbook with the existing absolute video path.
- [ ] Run video mapping and replay audits.
