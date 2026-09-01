# VLA82-018 Continuous Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent retry-and-drop trajectories from receiving strict PASS and produce a clean one-grasp VLA82-018 simulation result.

**Architecture:** Add a pure controller-trace audit at the episode acceptance boundary and reuse the existing debounced transport-hold primitives inside cabinet extraction. Keep physical predicates independent, then require both physics and controller-quality evidence.

**Tech Stack:** Python 3.11, NumPy, pytest, RoboCasa, robosuite, MuJoCo, ffmpeg.

## Global Constraints

- No direct simulator state writes.
- A valid result has one continuous grasp and an observed `release → settle` sequence.
- Final media must be upright, unobstructed, at least 1280×720, and fully decodable.

---

### Task 1: Strict trace quality gate

**Files:**
- Modify: `tools/vla82_full_sim/expert.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Produces: `pick_place_strict_trace_errors(phases, controller_trace) -> tuple[str, ...]`

- [ ] Add a failing test whose trace contains three separated sustained grasp runs and no release/settle; expect reacquisition and missing controlled-release errors.
- [ ] Add a passing reference test with one continuous grasp followed by release and settle.
- [ ] Run the focused test and confirm RED because the audit function is absent.
- [ ] Implement the minimal pure trace audit and call it from `run_episode` after predicate evaluation.
- [ ] Run focused and related strict-audit tests and confirm GREEN.

### Task 2: Debounced cabinet extraction hold

**Files:**
- Modify: `tools/vla82_full_sim/expert.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Produces: `cabinet_extract_transition(hold_valid, lost_frames, extracted, threshold, next_state) -> str`

- [ ] Add a failing test proving one lost frame keeps `cabinet_extract`, three lost frames request retry, and sufficient extraction with a valid hold advances.
- [ ] Run the focused test and confirm RED because the transition function is absent.
- [ ] Implement the transition and use it in the cabinet extraction state while maintaining the closed gripper command.
- [ ] Run focused controller tests and confirm GREEN.

### Task 3: Physical rerun and evidence gate

**Files:**
- Generate: `outputs/midterm_testing_vla82/vla018_continuous_grasp_*/VLA82-018/episode-*.npz`
- Generate: matching JSON and result video
- Move after verification: `datasets/VLA82 video/VLA82-018_盒装饮料_严格PASS_结果视频.mp4`
- Move after verification: `datasets/VLA82 video/VLA82-018_盒装饮料_严格PASS_判定.json`

- [ ] Run VLA82-018 with a fresh seed and the corrected controller.
- [ ] Reject any run with more than one sustained grasp acquisition, missing release/settle, or nonzero transport loss.
- [ ] Replay the accepted episode with the unobstructed world-locked camera.
- [ ] Inspect a dense contact sheet and the full video for one grasp, continuous carry, and stable placement.
- [ ] Verify H.264 decode, resolution, JSON PASS fields, hashes, and archive pairing before moving files.

