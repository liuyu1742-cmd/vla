# VLA82 Ten Strict-PASS Assets Visual Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the recognizable appearance of ten already strict-PASS VLA82 simulation assets, regenerate clear 720p strict-PASS evidence, update the evidence workbook, and delete each superseded old video only after its replacement is verified.

**Architecture:** Keep each asset's collision geometry, mass, friction, joints, controller, action primitive, and success predicate unchanged; add only named visual geoms to the generated MJCF. Process every selection as a guarded transaction: test and materialize the visual model, run a fresh strict episode, render and inspect the replacement video, update the workbook, then delete only that selection's former MP4.

**Tech Stack:** Python 3.11, MuJoCo, RoboCasa/robosuite, NumPy, unittest/pytest, FFmpeg/ffprobe, artifact-tool spreadsheet runtime, PowerShell.

## Global Constraints

- Target selections are exactly `VLA82-004`, `005`, `014`, `017`, `019`, `036`, `038`, `042`, `045`, and `060`.
- Prefer the source video's recognizable colors and structure; when unclear, use a common identifiable household appearance.
- Do not change collision geometry, friction, mass, joints, controller parameters, action primitive, or strict-PASS thresholds.
- New evidence must be a fresh strict-PASS run, upright, unobstructed, and 1280x720 or better.
- Delete an old video only after the same selection has a verified replacement video and the replacement path is written into the new workbook.
- Preserve every old video for a selection whose replacement fails any gate.
- This directory is not a Git repository, so the plan's normal commit steps are replaced by file/status snapshots.

---

### Task 1: Add Visual Contracts for the Ten Assets

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Consumes: `tools.vla82_full_sim.assets._xml_for_asset(asset: AssetSpec, texture_name: str) -> str`
- Produces: regression contracts for named visual parts, unchanged collision count, and absence of `fromto` geoms.

- [ ] **Step 1: Write the failing visual-contract test**

```python
def test_ten_strict_pass_assets_have_recognizable_visual_parts(self):
    expected = {
        "VLA82-004": ("spray_trigger", "spray_nozzle", "product_label"),
        "VLA82-005": ("whisk_handle", "whisk_wire_0", "whisk_wire_5"),
        "VLA82-014": ("measuring_handle", "measure_mark_25", "measure_mark_75"),
        "VLA82-017": ("bottle_cap", "bottle_shoulder", "grip_ring_2"),
        "VLA82-019": ("can_top_rim", "can_label", "pull_tab"),
        "VLA82-036": ("mug_rim", "mug_inner", "mug_handle_2"),
        "VLA82-038": ("pizza_blade", "blade_axle", "pizza_handle"),
        "VLA82-042": ("brush_grip", "brush_head", "bristle_group_3"),
        "VLA82-045": ("foil_roll", "foil_sheet", "serrated_cutter"),
        "VLA82-060": ("pump_head", "pump_nozzle", "liquid_layer"),
    }
    for selection_id, names in expected.items():
        xml = self._asset_xml(selection_id)
        self.assertNotIn("fromto=", xml)
        self.assertEqual(xml.count('class="collision"'), 2)
        for name in names:
            self.assertIn(f'name="{name}"', xml)
```

- [ ] **Step 2: Run the test and verify it fails on missing names**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn.SourceSpawnTests.test_ten_strict_pass_assets_have_recognizable_visual_parts -v`

Expected: `FAIL` because the ten complete profiles do not yet exist.

- [ ] **Step 3: Record the pre-change generated XML collision counts**

Run: `rg -n 'class="collision"|name="(spray_|whisk_|measuring_|bottle_|can_|mug_|pizza_|brush_|foil_|pump_)' assets/vla82_source_textured/VLA82-{004,005,014,017,019,036,038,042,045,060}/model.xml`

Expected: one physical collision geom per asset plus the default collision declaration.

### Task 2: Implement Recognizable Visual-Only MJCF Profiles

**Files:**
- Modify: `tools/vla82_full_sim/assets.py:466-668`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Consumes: `AssetSpec.selection_id`, `AssetSpec.semantic_class`, and the existing generated-asset template.
- Produces: `_visual_geoms_for_vla82_asset(asset: AssetSpec) -> str` used by `_xml_for_asset` without changing the collision geom.

- [ ] **Step 1: Add a focused visual profile helper**

```python
def _visual_geoms_for_vla82_asset(asset: AssetSpec) -> str | None:
    """Return visual-only MJCF geoms for recognized VLA82 selections."""
    profiles = {
        "VLA82-004": _spray_bottle_visuals,
        "VLA82-005": _whisk_visuals,
        "VLA82-014": _measuring_cup_visuals,
        "VLA82-017": _water_bottle_visuals,
        "VLA82-019": _food_can_visuals,
        "VLA82-036": _coffee_mug_visuals,
        "VLA82-038": _pizza_cutter_visuals,
        "VLA82-042": _dish_brush_visuals,
        "VLA82-045": _aluminum_foil_visuals,
        "VLA82-060": _soap_dispenser_visuals,
    }
    builder = profiles.get(asset.selection_id)
    return None if builder is None else builder(asset)
```

- [ ] **Step 2: Add the ten named visual builders**

Use only `pos`, `size`, `quat`/`euler`, `rgba`, and visual materials. Use capsules, cylinders, boxes, ellipsoids, and torus-like segmented capsules; never use `fromto`. Keep the existing single `class="collision"` physical geom untouched.

- [ ] **Step 3: Integrate the helper ahead of generic visual fallbacks**

```python
profile = _visual_geoms_for_vla82_asset(asset)
if profile is not None:
    visual_geoms = profile
elif asset.selection_id == "VLA82-058" and asset.semantic_class == "牙刷":
    ...
```

- [ ] **Step 4: Run the focused visual contract**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn.SourceSpawnTests.test_ten_strict_pass_assets_have_recognizable_visual_parts -v`

Expected: `OK`.

- [ ] **Step 5: Run all asset/source-spawn tests**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn -v`

Expected: all tests pass.

### Task 3: Materialize and Load-Test Every Updated Asset

**Files:**
- Modify (generated): `assets/vla82_source_textured/VLA82-004/model.xml`
- Modify (generated): `assets/vla82_source_textured/VLA82-005/model.xml`
- Modify (generated): `assets/vla82_source_textured/VLA82-014/model.xml`
- Modify (generated): `assets/vla82_source_textured/VLA82-017/model.xml`
- Modify (generated): `assets/vla82_source_textured/VLA82-019/model.xml`
- Modify (generated): `assets/vla82_source_textured/VLA82-036/model.xml`
- Modify (generated): `assets/vla82_source_textured/VLA82-038/model.xml`
- Modify (generated): `assets/vla82_source_textured/VLA82-042/model.xml`
- Modify (generated): `assets/vla82_source_textured/VLA82-045/model.xml`
- Modify (generated): `assets/vla82_source_textured/VLA82-060/model.xml`

**Interfaces:**
- Consumes: `materialize_custom_asset(asset: AssetSpec) -> AssetSpec` and `_scene_for_spec`.
- Produces: loadable model XML files whose visual names satisfy Task 1.

- [ ] **Step 1: Materialize all ten assets through the project API**

Run a Python one-liner that loads `_load_compiled_specs()`, resolves both object and receptacle `AssetSpec` values for the ten IDs, and calls `materialize_custom_asset` for each.

Expected: ten generated model directories exist with current timestamps.

- [ ] **Step 2: Parse every XML and verify collision invariance**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -c "...ElementTree.parse(...); assert collision_count == 2..."`

Expected: all ten parse successfully and retain one physical collision geom.

- [ ] **Step 3: Reset each RoboCasa scene for ten simulation steps**

Run a Python smoke script using `make_environment(_scene_for_spec(spec), seed, has_renderer=False)` for each selection and ten zero-action steps.

Expected: all ten environments reset and step without MuJoCo warnings, scale errors, or Python crashes.

### Task 4: Regenerate Fresh Strict-PASS Episodes and 720p Evidence

**Files:**
- Create: `outputs/midterm_testing_vla82/visual_refresh_10_assets_20260828/expert_demos/<selection>/episode-<seed>.json`
- Create: `outputs/midterm_testing_vla82/visual_refresh_10_assets_20260828/expert_demos/<selection>/episode-<seed>.npz`
- Create: `outputs/strict_pass_videos_visual_refresh_20260828/<selection>_真实外观_严格PASS_720p正向无遮挡.mp4`

**Interfaces:**
- Consumes: existing expert primitive and original passing seed (`2005` for 004; recorded passing seed for every other selection).
- Produces: strict JSON/NPZ and a 1280x720 MP4 per successfully refreshed selection.

- [ ] **Step 1: Run a fresh expert episode for one selection at a time**

Use the existing collection entry point with `--selection-id`, the known passing seed, and the visual-refresh output root. Do not copy or relabel an old JSON.

Expected: JSON contains a successful terminal predicate, no fallback-only success, and a new timestamp.

- [ ] **Step 2: Reproduce each failed run with one targeted adjustment only**

Because collision and control logic are unchanged, first compare scene initial state and seed with the previous strict run. If needed, adjust only the initial visual-safe pose or use the last known passing action profile; do not weaken predicates.

- [ ] **Step 3: Replay every successful NPZ to 1280x720 MP4**

Run `tools/replay_vla82_strict_pass_video.py` with width `1280`, height `720`, `30` fps, upright orientation, and the established clear camera.

Expected: ffprobe reports H.264 MP4, 1280x720, positive duration, and 30 fps.

- [ ] **Step 4: Generate first/middle/last contact sheets**

Use FFmpeg to extract first, midpoint, and terminal frames for each MP4 into `outputs/strict_pass_videos_visual_refresh_20260828/qa/`.

Expected: the target asset, gripper, source, and destination are visible; no door or robot body blocks the decisive operation.

- [ ] **Step 5: Reject any selection with motion or appearance defects**

Reject a replacement if grasp is missed, object drops, the robot pauses abnormally, placement is incomplete, the image is inverted, or the object is not recognizable. Keep that selection's old video and do not update its workbook row.

### Task 5: Update the Strict-PASS Evidence Workbook

**Files:**
- Read: `outputs/strict_pass_video_index_20260811/VLA82严格PASS_23项_058牙刷放入漱口杯_真实外观更新版.xlsx`
- Create: `outputs/strict_pass_video_index_20260811/VLA82严格PASS_23项_10物体真实外观更新版.xlsx`

**Interfaces:**
- Consumes: verified replacement MP4 absolute paths and their strict JSON evidence paths.
- Produces: a visually checked workbook with only verified replacements substituted.

- [ ] **Step 1: Load the spreadsheet runtime and inspect the source workbook**

Use artifact-tool to inspect sheet names, used ranges, formulas, styles, and the rows for the ten selection IDs.

- [ ] **Step 2: Replace only verified video and evidence paths**

For each verified selection, update the absolute MP4 path, strict JSON/NPZ path, resolution, and note `真实外观增强，正向无遮挡，严格PASS复验通过`. Leave unsuccessful selections unchanged.

- [ ] **Step 3: Export to the new workbook path**

Export first to an ASCII temporary path if the runtime cannot write the Chinese filename, then copy it with `Copy-Item -LiteralPath`.

- [ ] **Step 4: Inspect and render the result**

Check all ten rows, formulas, broken links, and spreadsheet error tokens. Render the evidence sheet to PNG and visually inspect column widths, wrapping, and row readability.

Expected: no `#REF!`, `#DIV/0!`, `#VALUE!`, or `#NAME?`; every updated MP4 path exists.

### Task 6: Delete Superseded Videos Safely and Run Final Verification

**Files:**
- Delete: exactly the former MP4 recorded in the source workbook for each successfully replaced selection.
- Preserve: all JSON, NPZ, screenshots, workbook versions, and any old MP4 whose replacement did not pass.

**Interfaces:**
- Consumes: source workbook old paths and new workbook verified paths.
- Produces: one current strict-PASS video per refreshed selection with no dangling workbook path.

- [ ] **Step 1: Resolve and validate exact deletion targets**

For every old path, call `Resolve-Path -LiteralPath` and assert the resolved path starts with `C:\OpenVLA-Simulator\outputs\`; reject any duplicate where old and new resolve to the same file.

- [ ] **Step 2: Recheck every replacement gate immediately before deletion**

Assert: new MP4 exists, ffprobe is 1280x720 or better, strict JSON is PASS, new workbook contains the new absolute path, and contact-sheet QA is accepted.

- [ ] **Step 3: Delete explicit old MP4 files only**

Use `Remove-Item -LiteralPath <exact-old-path>` one file at a time, with no wildcard and no recursive option.

- [ ] **Step 4: Verify final evidence integrity**

Run the focused and full source-spawn tests, parse all refreshed strict JSON files, ffprobe all new MP4s, inspect all workbook paths, and verify each deleted path is absent.

Expected: tests pass; every refreshed row points to an existing strict-PASS 720p MP4; only superseded old MP4s are absent.

## Self-Review

- Spec coverage: all ten requested assets, source-informed appearance, visual-only changes, fresh strict replay, 720p unobstructed QA, workbook update, and guarded deletion are assigned to Tasks 1-6.
- Placeholder scan: no `TBD`, `TODO`, deferred implementation, or unspecified acceptance gate remains.
- Type consistency: `_visual_geoms_for_vla82_asset(asset: AssetSpec) -> str | None` is produced and consumed consistently; episode and workbook paths are passed as absolute Windows paths.

