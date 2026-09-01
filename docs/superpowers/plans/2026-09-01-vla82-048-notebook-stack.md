# VLA82-048 Notebook Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make VLA82-048 physically stack a recognizable notebook on a recognizable folder and produce one new strict PASS.

**Architecture:** Extend the existing open-support environment with a fixed folder target and route the authoritative `stack` phase through the proven pick-place controller. Keep appearance geometry separate from simple collision bodies and add only selection-specific grasp/release calibration.

**Tech Stack:** Python 3.11, RoboCasa, robosuite, MuJoCo, pytest, ffmpeg.

## Global Constraints

- Preserve `VLA82-048 / 笔记本 / 叠放于文件夹上` as the authoritative operation.
- Require real notebook-folder collision contact and controlled stable release.
- Generate 1280×720 upright unobstructed video and archive video plus JSON in `datasets/VLA82 video`.
- Use test-first changes and bounded targeted retries.

---

### Task 1: Bind the notebook and folder scene

**Files:**
- Modify: `tools/vla82_full_sim/environment.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Consumes: `build_scene_request`, `_WorkStudyOpenSupportEnvironment.build`.
- Produces: VLA82-048 request with `table -> folder / on` and physical `obj`, `folder` objects.

- [ ] Add a failing test asserting authoritative semantics and physical folder target.
- [ ] Run the focused test and confirm the missing mapping/object failure.
- [ ] Add VLA82-048 to open support and bind a fixed folder target.
- [ ] Run the focused test to green.

### Task 2: Add recognizable assets and reachable grasp affordance

**Files:**
- Modify: `tools/vla82_full_sim/assets.py`
- Modify: `tools/vla82_full_sim/expert.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Consumes: `_xml_for_asset`, `pick_place_grasp_target`.
- Produces: notebook/folder visual geoms and raised notebook binding grasp target.

- [ ] Add failing XML and grasp-target assertions.
- [ ] Run focused tests and confirm expected failures.
- [ ] Add notebook and folder geometry plus selection-specific grasp target.
- [ ] Run focused tests to green.

### Task 3: Route stack through physical pick-place and validate

**Files:**
- Modify: `tools/vla82_full_sim/expert.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Consumes: `pick_place_execution_phases`.
- Produces: `(("grasp", "place"), "stack")` for one authoritative stack phase.

- [ ] Add a failing controller-routing test.
- [ ] Run it and confirm `None` is returned before implementation.
- [ ] Add the minimal stack mapping.
- [ ] Run focused tests and then one physical episode.

### Task 4: Strict replay and archive

**Files:**
- Create: `datasets/VLA82 video/VLA82-048_笔记本_严格PASS_结果视频.mp4`
- Create: `datasets/VLA82 video/VLA82-048_笔记本_严格PASS_判定.json`

**Interfaces:**
- Consumes: PASS NPZ/JSON episode.
- Produces: 720p video, strict JSON, updated index.

- [ ] Verify predicate success, real target contact, zero target-fixture collision and final stability.
- [ ] Render and visually inspect the entire action sequence.
- [ ] Archive video and JSON.
- [ ] Update the strict PASS workbook and verify paths/formulas/rendering.
