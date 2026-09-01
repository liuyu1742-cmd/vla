# VLA82 Clean Motion Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate VLA82-007, drawer-placement cases, and VLA82-036 with no collateral collision, no post-release recontact, and materially shorter idle-looking correction phases while preserving physical strict-PASS criteria.

**Architecture:** Add small pure decision helpers to the expert controller and cover them with unit tests before integrating them into the existing knob and pick-place state machines. Validate first at the unit level, then by fresh single-object simulations whose JSON/NPZ/video evidence must pass both the existing strict predicate and new trajectory-quality checks.

**Tech Stack:** Python 3.11, NumPy, RoboCasa/MuJoCo, unittest/pytest, FFmpeg.

## Global Constraints

- Do not directly mutate object or fixture joint state; all motion must pass through the public simulator action API.
- A new result is retained only if JSON reports `status=PASS` and `predicate_success=true`, NPZ exists, and the video passes full-timeline quality review.
- Reject episodes with non-target cookware displacement, post-release target recontact, or controller-cap exhaustion.
- Preserve the user’s existing files and unrelated workspace changes.

---

### Task 1: Stop knob control immediately after the physical goal is reached

**Files:**
- Modify: `tools/vla82_full_sim/expert.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Produces: `knob_turn_should_stop(*, initial_joint: float, current_joint: float, required_delta: float, contact_observed: bool) -> bool`
- Consumes: live selected-knob joint value and exact selected-body contact history.

- [ ] Add a failing test proving the controller must stop after contact and a 0.30-radian selected-joint change, but not before contact or below the threshold.
- [ ] Run the focused test and confirm it fails because the helper is missing.
- [ ] Implement the pure helper and use it in `KnobPrimitive.actions()` to return before additional wrist motion.
- [ ] Run the focused and nearby knob tests.
- [ ] Run a fresh VLA82-007 episode; require no robot–`cookware_*` contact and no meaningful cookware displacement after the knob predicate becomes true.

### Task 2: Terminate pick-place after a stable release instead of reacquiring the object

**Files:**
- Modify: `tools/vla82_full_sim/expert.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Produces: `pick_place_release_transition(*, raw_grasp: bool, target_contact: bool, phase_completed: bool) -> str`
- Consumes: live grasp/contact state and recorded phase-completion state.

- [ ] Add a failing test proving that a released object already stably supported by its target transitions to `settle`, never `close/grasp`.
- [ ] Run the test and confirm the missing behavior fails.
- [ ] Implement the helper and wire it into the loss-of-hold transitions in `lower`, `insert_drawer`, and release handling.
- [ ] Run focused pick-place tests.
- [ ] Regenerate VLA82-036 and require a terminal settle/retreat segment with zero target recontact after successful release.

### Task 3: Remove excessive drawer-placement hesitation

**Files:**
- Modify: `tools/vla82_full_sim/expert.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Produces: bounded drawer pose-convergence decisions based on position/yaw error trends rather than long fixed-step loops.
- Consumes: live object pose, drawer bounds, wrist joint state, and the existing action-only control path.

- [ ] Add failing tests for immediate transition when position/yaw tolerances are already satisfied and for bounded correction when progress stalls.
- [ ] Run tests and confirm the current fixed-step behavior fails them.
- [ ] Implement the smallest transition changes without weakening final containment or stable-release criteria.
- [ ] Run focused and full VLA82 unit tests.
- [ ] Regenerate VLA82-005, VLA82-038, and VLA82-045; compare phase durations and reject any regrasp after release.

### Task 4: Quality gate, evidence, and spreadsheet refresh

**Files:**
- Modify: `outputs/strict_pass_video_index_20260811/VLA82_严格PASS无遮挡结果视频路径汇总.xlsx`
- Create: `outputs/midterm_testing_vla82/clean_motion_regeneration_20260811/`

**Interfaces:**
- Consumes: fresh PASS JSON, NPZ, MP4 and quality-audit summaries.
- Produces: revised recommended-video index containing only clean-motion episodes.

- [ ] Run strict predicates and trajectory-quality checks on every regenerated candidate.
- [ ] Review each full video timeline plus start/contact/release/end frames.
- [ ] Replace workbook paths only for candidates that satisfy all checks; otherwise remove their “recommended clean video” designation.
- [ ] Inspect formulas and render the workbook for visual verification.
- [ ] Report exact retained PASS count and evidence paths without promoting failed or visually poor episodes.

