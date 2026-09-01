# VLA82-058 Realistic Toothbrush and Cup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the primitive VLA82-058 box visuals with a recognizable blue-white toothbrush and open cylindrical mouthwash cup while retaining the verified collision and strict-PASS behavior.

**Architecture:** Add two VLA82-058 semantic-class-specific visual branches to the existing MJCF generator. Visual geoms are non-colliding; the current toothbrush collision box and five-wall cup collision contract remain unchanged. Re-materialize the assets, rerun seed 2005, and publish a new 720p evidence video only after strict verification.

**Tech Stack:** Python 3.11, MuJoCo MJCF, RoboCasa/RoboSuite, pytest, FFmpeg/ffprobe.

## Global Constraints

- Preserve the current VLA82-058 collision geometry names, sizes, friction, `reg_bbox`, scene placement, controller, and predicate thresholds.
- Toothbrush appearance: blue-white curved-looking handle, narrowed neck, oval head, visible white/light-blue bristles.
- Cup appearance: light-blue cylindrical body, circular rim, visible open cavity, no handle.
- Final evidence must be 1280×720, 30 FPS, upright, unobstructed, and backed by a strict `PASS` JSON.
- The workspace is not a Git repository; record test outputs instead of commit checkpoints.

---

### Task 1: Specify Recognizable MJCF Visual Contracts

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Modify: `tools/vla82_full_sim/assets.py`

**Interfaces:**
- Consumes: `_xml_for_asset(asset: AssetSpec, texture_name: str) -> str`
- Produces: VLA82-058 XML containing named visual parts while retaining the existing collision contracts.

- [ ] **Step 1: Write failing toothbrush and cup visual tests**

```python
def test_vla82_058_assets_have_recognizable_visual_parts(self) -> None:
    from tools.vla82_full_sim.assets import _xml_for_asset
    spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-058")
    request = _scene_for_spec(spec)
    toothbrush_xml = _xml_for_asset(request.object_assets[0], "source_texture.png")
    cup_xml = _xml_for_asset(request.object_assets[1], "source_texture.png")
    for name in ("toothbrush_handle_visual", "toothbrush_neck_visual", "toothbrush_head_visual", "toothbrush_bristle_white_0"):
        self.assertIn(f'name="{name}"', toothbrush_xml)
    for name in ("cup_outer_visual", "cup_inner_visual", "cup_rim_visual"):
        self.assertIn(f'name="{name}"', cup_xml)
    self.assertEqual(toothbrush_xml.count('class="collision"'), 2)
    self.assertGreaterEqual(cup_xml.count('class="collision"'), 6)
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tools\tests\test_vla82_source_spawn.py -q -k vla82_058_assets_have_recognizable_visual_parts`

Expected: FAIL because the named visual geoms do not exist.

- [ ] **Step 3: Add semantic-specific visual geometry**

In `_xml_for_asset`, add the toothbrush branch before generic geometry handling. It must retain the original single collision box:

```python
if asset.selection_id == "VLA82-058" and asset.semantic_class == "牙刷":
    bbox = f"{sx:.6f} {sy:.6f} {sz:.6f}"
    bristles = "\n".join(
        f'    <geom name="toothbrush_bristle_white_{index}" class="visual" type="capsule" '
        f'fromto="{x:.4f} -0.010 0.060 {x:.4f} -0.022 0.060" size="0.0012" rgba="0.96 0.98 1 1"/>'
        for index, x in enumerate((-0.006, -0.002, 0.002, 0.006))
    )
    visual_and_collision = f'''    <geom name="toothbrush_handle_visual" class="visual" type="capsule" fromto="0 0 -0.090 0 0 0.030" size="0.0075" rgba="0.12 0.48 0.88 1"/>
    <geom name="toothbrush_neck_visual" class="visual" type="capsule" fromto="0 0 0.025 0 0 0.055" size="0.0045" rgba="0.92 0.96 1 1"/>
    <geom name="toothbrush_head_visual" class="visual" type="ellipsoid" pos="0 0 0.067" size="0.009 0.005 0.018" rgba="0.15 0.52 0.90 1"/>
{bristles}
    <geom name="collision" class="collision" type="box" size="0.010000 0.010000 0.090000"/>'''
```

For `asset.selection_id == "VLA82-058" and asset.semantic_class == "漱口杯"`, generate 20 closely spaced vertical capsule wall segments around a circle, 20 short capsule rim segments joining adjacent circle points, and one thin bottom cylinder. This leaves the top physically visible as open while reading as a round light-blue cup; retain the existing five box collision geoms and `reg_bbox`.

- [ ] **Step 4: Run the test and verify GREEN**

Run the Step 2 command.

Expected: PASS.

### Task 2: Materialize and Validate the Physical Scene

**Files:**
- Regenerate: `assets/vla82_source_textured/VLA82-058/model.xml`
- Regenerate: `assets/vla82_source_textured/VLA82-058/mouthwash_cup/model.xml`
- Verify: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Consumes: `materialize_custom_asset(asset: AssetSpec) -> AssetSpec`
- Produces: loadable MJCF assets and unchanged capture-contract collision names.

- [ ] **Step 1: Materialize both request assets**

```python
spec = next(item for item in _load_compiled_specs() if item.selection_id == "VLA82-058")
request = _scene_for_spec(spec)
for asset in request.object_assets:
    materialize_custom_asset(asset)
```

- [ ] **Step 2: Run VLA82-058 asset and live-scene tests**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest tools\tests\test_vla82_source_spawn.py -q -k "vla82_058 or mouthwash_cup"`

Expected: all selected tests PASS; the live contract still exposes five cup collision walls and an `inside` target.

### Task 3: Re-establish Strict PASS and Publish Video

**Files:**
- Create: `outputs/midterm_testing_vla82/vla058_realistic_assets_256p_20260828/expert_demos/VLA82-058/episode-2005.{json,npz}`
- Create: `outputs/strict_pass_videos_720p_20260828/VLA82-058_牙刷放入真实漱口杯_严格PASS_720p正向无遮挡.mp4`
- Modify: `outputs/strict_pass_video_index_20260811/VLA82严格PASS_23项_058牙刷放入漱口杯.xlsx`

**Interfaces:**
- Consumes: `collect_expert_episode`, `replay_vla82_strict_pass_video.py`
- Produces: strict JSON/NPZ, 720p video, and updated workbook path.

- [ ] **Step 1: Run a fresh seed-2005 episode in a new output directory**

Expected JSON fields: `status="PASS"`, `predicate_success=true`, empty `errors`, `contact_verified=true`, `containment=true`, `stable_release=true`.

- [ ] **Step 2: Replay the new NPZ at 1280×720 and 30 FPS**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\replay_vla82_strict_pass_video.py outputs\midterm_testing_vla82\vla058_realistic_assets_256p_20260828\expert_demos\VLA82-058\episode-2005.npz outputs\strict_pass_videos_720p_20260828\VLA82-058_牙刷放入真实漱口杯_严格PASS_720p正向无遮挡.mp4 --width 1280 --height 720 --fps 30`

- [ ] **Step 3: Inspect a six-frame contact sheet**

Expected: upright frames clearly show the blue-white toothbrush, brush head/bristles, circular open cup, grasp, transport, insertion, release, and final in-cup pose with no visual obstruction.

- [ ] **Step 4: Update the VLA82-058 workbook row with the new absolute paths**

Expected: strict count remains 23; formula error scan returns zero matches; workbook preview remains legible.

- [ ] **Step 5: Run final verification**

Run focused pytest, JSON assertions, `ffprobe`, NPZ non-pickle load, file-existence checks, and workbook inspection.

Expected: all VLA82-058 focused tests PASS; video is 1280×720/30 FPS; all evidence files are non-empty.
