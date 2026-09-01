# VLA82-016 Bedside Book Strict PASS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make VLA82-016 physically pick up a recognizable book and place it stably on a visible bedside table, then archive strict-PASS evidence.

**Architecture:** Route VLA82-016 to a dedicated bedroom scene with a physical floor-standing bed and fixed bedside-table target. Reuse the proven short-edge top-grasp controller profile for rectangular books, while keeping asset geometry honest and target contact explicit.

**Tech Stack:** Python 3.11, MuJoCo, RoboCasa/robosuite, pytest, FFmpeg, artifact-tool.

## Global Constraints

- The operation is exactly “从床边取用并放至床头柜”.
- No raised grasp block, hidden handle, direct state write, or scripted terminal state.
- Final video and JSON go to `C:\OpenVLA-Simulator\datasets\VLA82 video`.
- Video is upright, unobstructed, and 1280×720.
- Only a genuine strict predicate PASS may increase the count.

---

### Task 1: Define scene and asset behavior with a failing test

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Consumes: `_load_compiled_specs()`, `_scene_for_spec()`, `_xml_for_asset()`
- Produces: regression expectations for `VLA82BedsideBookPlace`, book/nightstand XML geometry, and top short-edge grasp policy

- [ ] Add `test_vla82_016_places_a_recognizable_book_on_a_physical_nightstand` asserting the authoritative operation, task class, `bedside→nightstand/on` semantics, recognizable geometry names, no grasp block, and yaw-aligned top grasp.
- [ ] Run `python -m pytest tools/tests/test_vla82_source_spawn.py -k "016_places" -q` and confirm failure because the old task class is `PickPlaceCounterToCabinet`.

### Task 2: Implement the dedicated physical scene and appearance

**Files:**
- Modify: `tools/vla82_full_sim/environment.py`
- Modify: `tools/vla82_full_sim/assets.py`
- Modify: `tools/vla82_full_sim/expert.py`
- Modify: `configs/vla82_full_simulation/object_assets.json`
- Modify: `configs/vla82_full_simulation/authoritative_asset_manifest.json`

**Interfaces:**
- Consumes: `SceneRequest`, `_WorkStudyOpenSupportEnvironment`, `AssetSpec`
- Produces: `VLA82BedsideBookPlace`, fixed `nightstand`, honest book XML, and VLA82-016 controller profile

- [x] Route VLA82-016 to `VLA82BedsideBookPlace` with `bed`, `nightstand`, `on` semantics.
- [x] Add fixed bed and nightstand assets, spawn the book on the mattress, and bind the cabinet top as the target support.
- [x] Render book covers/pages/spine/label, a recognizable bed, and a wooden nightstand without nonphysical grasp geometry.
- [x] Set the compact book dimensions and reuse yaw alignment, pad-centred descent, exact contact-height release, and short secure-hold settings.
- [x] Re-run the focused test and require PASS.

### Task 3: Run and diagnose the physical closed loop

**Files:**
- Output: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-016/episode-2000.{npz,json}`

**Interfaces:**
- Consumes: implemented scene and controller
- Produces: a strict predicate result and complete physics trace

- [ ] Run `tools/run_vla82_full_simulation.py collect-demos --selection-id VLA82-016 --episodes-per-object 1 --max-attempts-per-object 1`.
- [ ] If it fails, inspect state counts, transition events, exact contacts, object/target poses, and final stability before making one tested correction.
- [ ] Stop candidate work only if continued repair is structurally unjustified; otherwise require `status=PASS` and `predicate_success=true`.

### Task 4: Produce and archive evidence

**Files:**
- Create: `datasets/VLA82 video/VLA82-016_书籍_严格PASS_结果视频.mp4`
- Create: `datasets/VLA82 video/VLA82-016_书籍_严格PASS_判定.json`
- Modify: latest workbook under `outputs/strict_pass_video_index_20260811/`

**Interfaces:**
- Consumes: strict episode NPZ/JSON
- Produces: 720p video, formal JSON, and incremented workbook row

- [ ] Replay the exact NPZ at 1280×720 and generate a one-second contact sheet.
- [ ] Visually confirm upright framing, recognizable book/nightstand, full action visibility, no drop, and stable final placement.
- [ ] Archive MP4 and byte-identical JSON in the requested directory.
- [ ] Update the strict-PASS workbook count and append the VLA82-016 row using artifact-tool; render and inspect the result.

### Task 5: Independent completion verification

**Files:**
- Verify all outputs above without modifying them.

**Interfaces:**
- Consumes: formal assets and workbook
- Produces: evidence-backed completion report

- [ ] Run focused pytest tests and require zero failures.
- [ ] Assert strict JSON status, predicate success, contact verification, controlled release, target support contact, and final stability.
- [ ] Verify MP4 resolution/frame count, file existence, JSON hash, no forbidden grasp geometry, workbook row/count, and zero formula errors.

## Self-Review

All design requirements map to a task; no placeholders remain; interface names match the current simulator modules; the plan adds only one new strict PASS candidate and its evidence.
