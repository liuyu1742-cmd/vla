# VLA82-037 Horizontal Tongs Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make VLA82-037 materialize as a horizontally resting pair of tongs, reject invalid vertical drawer assets in seconds, and run one evidence-gated physical closed-loop retest.

**Architecture:** Keep the authoritative dimensions and source texture unchanged, but give VLA82-037 a source-specific horizontal axis permutation inside the existing MJCF generator. Add a pure XML preflight in the asset layer and run it before `create_env`, so invalid drawer geometry fails before RoboCasa enters its 50-attempt placement loop. Preserve all existing strict contact, phase, transport, release, and support predicates.

**Tech Stack:** Python 3.11, `xml.etree.ElementTree`, NumPy, `unittest`, MuJoCo/RoboCasa, existing VLA82 evidence exporter and Node workbook updater.

## Global Constraints

- Change only VLA82-037 asset orientation and its initialization preflight.
- Preserve the multiset of half-extents `{0.012, 0.012, 0.095}`, density `300`, and friction `0.45 0.004 0.0001`.
- Do not modify RoboCasa third-party placement or success logic.
- Do not disable placement-boundary or collision checks.
- Do not write object pose or joint state after reset.
- Do not weaken strict PASS predicates or export a failed episode as a result video.
- Run at most one complete VLA82-037 physical retest after preflight and regression tests pass.
- The workspace is not a Git repository; record verification outputs, but omit commit commands.

---

### Task 1: Generate VLA82-037 With Its Long Axis Horizontal

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Modify: `tools/vla82_full_sim/assets.py:428-486`

**Interfaces:**
- Consumes: `_scene_for_spec(spec).primary_asset` and `_xml_for_asset(asset, texture_name)`.
- Produces: `_vla82_037_oriented_half_extents(asset: AssetSpec) -> tuple[float, float, float]` and a VLA82-037 MJCF whose visual, collision, and `reg_bbox` half-extents are `(0.095, 0.012, 0.012)`.

- [ ] **Step 1: Write the failing orientation test**

Add imports and this test to `SourceSpawnReachabilityTest`:

```python
from tools.vla82_full_sim.assets import _xml_for_asset

def test_vla82_037_tongs_materialize_with_long_axis_horizontal(self) -> None:
    specs = {item.selection_id: item for item in _load_compiled_specs()}
    asset = _scene_for_spec(specs["VLA82-037"]).primary_asset
    root = ET.fromstring(_xml_for_asset(asset, "source_texture.png"))
    bbox = np.fromstring(root.find(".//geom[@name='reg_bbox']").attrib["size"], sep=" ")
    visual = np.fromstring(root.find(".//geom[@name='visual']").attrib["size"], sep=" ")
    collision = np.fromstring(root.find(".//geom[@name='collision']").attrib["size"], sep=" ")

    self.assertIn(int(np.argmax(bbox)), (0, 1))
    self.assertEqual(sorted(bbox.tolist()), sorted(asset.dimensions))
    np.testing.assert_allclose(visual, bbox)
    np.testing.assert_allclose(collision, bbox)
```

- [ ] **Step 2: Run the test and observe RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_vla82_037_tongs_materialize_with_long_axis_horizontal
```

Expected: FAIL because `np.argmax(bbox) == 2` for `(0.012, 0.012, 0.095)`.

- [ ] **Step 3: Add the minimal source-specific axis permutation**

Add before `_xml_for_asset`:

```python
def _vla82_037_oriented_half_extents(asset: AssetSpec) -> tuple[float, float, float]:
    sx, sy, sz = (float(value) for value in asset.dimensions)
    if asset.selection_id != "VLA82-037":
        return sx, sy, sz
    return sz, sy, sx
```

In `_xml_for_asset`, use the helper for VLA82-037 without changing other geometries:

```python
elif asset.selection_id == "VLA82-037":
    ox, oy, oz = _vla82_037_oriented_half_extents(asset)
    geom = f'type="box" size="{ox:.6f} {oy:.6f} {oz:.6f}"'
    bbox = f"{ox:.6f} {oy:.6f} {oz:.6f}"
    visual_and_collision = f'''    <geom name="visual" class="visual" {geom} material="source_material"/>
    <geom name="collision" class="collision" {geom}/>'''
```

Place this branch before the generic geometry branches so future materialization cannot overwrite it with the old vertical box.

- [ ] **Step 4: Run the focused test and observe GREEN**

Run the Step 2 command.

Expected: `Ran 1 test ... OK`.

---

### Task 2: Fail Fast on Invalid Drawer Asset Geometry

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Modify: `tools/vla82_full_sim/assets.py:508-549`

**Interfaces:**
- Consumes: generated MJCF text and the authoritative `AssetSpec`.
- Produces: `validate_vla82_037_drawer_asset_xml(asset: AssetSpec, model_text: str) -> tuple[float, float, float]`.
- Raises: `AssetResolutionError` before `create_env` if the longest axis is vertical, vertical full extent exceeds `0.05 m`, the axis values differ from the authoritative dimensions, or visible/collision box geometry exceeds `reg_bbox`.

- [ ] **Step 1: Write failing preflight tests**

Add:

```python
from tools.vla82_full_sim.assets import (
    AssetResolutionError,
    _xml_for_asset,
    validate_vla82_037_drawer_asset_xml,
)

def test_vla82_037_drawer_preflight_rejects_vertical_tongs(self) -> None:
    specs = {item.selection_id: item for item in _load_compiled_specs()}
    asset = _scene_for_spec(specs["VLA82-037"]).primary_asset
    vertical = _xml_for_asset(asset, "source_texture.png").replace(
        'size="0.095000 0.012000 0.012000"',
        'size="0.012000 0.012000 0.095000"',
    )
    with self.assertRaisesRegex(AssetResolutionError, "horizontal drawer orientation"):
        validate_vla82_037_drawer_asset_xml(asset, vertical)

def test_vla82_037_drawer_preflight_accepts_generated_asset(self) -> None:
    specs = {item.selection_id: item for item in _load_compiled_specs()}
    asset = _scene_for_spec(specs["VLA82-037"]).primary_asset
    half_extents = validate_vla82_037_drawer_asset_xml(
        asset, _xml_for_asset(asset, "source_texture.png")
    )
    self.assertEqual(half_extents, (0.095, 0.012, 0.012))
```

- [ ] **Step 2: Run the new tests and observe RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest `
  tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_vla82_037_drawer_preflight_rejects_vertical_tongs `
  tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_vla82_037_drawer_preflight_accepts_generated_asset
```

Expected: import failure because `validate_vla82_037_drawer_asset_xml` does not exist.

- [ ] **Step 3: Implement the pure XML preflight**

Add this focused validator to `assets.py`:

```python
def validate_vla82_037_drawer_asset_xml(
    asset: AssetSpec, model_text: str,
) -> tuple[float, float, float]:
    if asset.selection_id != "VLA82-037":
        return tuple(float(value) for value in asset.dimensions)
    try:
        root = ET.fromstring(model_text)
        bbox_geom = next(
            geom for geom in root.iter("geom") if geom.get("name") == "reg_bbox"
        )
        bbox = np.fromstring(bbox_geom.attrib["size"], sep=" ", dtype=float)
    except (ET.ParseError, KeyError, StopIteration, ValueError) as error:
        raise AssetResolutionError("VLA82-037 drawer preflight cannot read reg_bbox") from error
    if bbox.shape != (3,) or int(np.argmax(bbox)) == 2 or 2.0 * float(bbox[2]) > 0.05:
        raise AssetResolutionError("VLA82-037 requires horizontal drawer orientation")
    if not np.allclose(np.sort(bbox), np.sort(np.asarray(asset.dimensions, dtype=float))):
        raise AssetResolutionError("VLA82-037 oriented dimensions differ from authoritative dimensions")
    for geom in root.iter("geom"):
        if geom.get("name") not in {"visual", "collision"}:
            continue
        size = np.fromstring(geom.attrib.get("size", ""), sep=" ", dtype=float)
        pos = np.fromstring(geom.attrib.get("pos", "0 0 0"), sep=" ", dtype=float)
        if size.shape != (3,) or pos.shape != (3,) or np.any(np.abs(pos) + size > bbox + 1e-9):
            raise AssetResolutionError("VLA82-037 reg_bbox does not cover physical geometry")
    return tuple(float(value) for value in bbox)
```

`assets.py` already imports `xml.etree.ElementTree as ET` and NumPy; reuse those imports rather than adding a second parser.

- [ ] **Step 4: Call preflight before writing the model**

In `materialize_custom_asset`, replace the direct write/read pair with:

```python
model_text = _xml_for_asset(asset, texture_path.name)
validate_vla82_037_drawer_asset_xml(asset, model_text)
model_path.write_text(model_text, encoding="utf-8")
```

This executes before RoboCasa `create_env()` and before the model is persisted.

- [ ] **Step 5: Run focused tests and observe GREEN**

Run the Step 2 command.

Expected: `Ran 2 tests ... OK`.

---

### Task 3: Materialize Evidence and Run Regression Gates

**Files:**
- Regenerate: `assets/vla82_source_textured/VLA82-037/model.xml`
- Regenerate: `assets/vla82_source_textured/VLA82-037/asset_evidence.json`
- Verify: `tools/tests/test_vla82_source_spawn.py`
- Verify: `tools/tests/test_vla82_pure_closed_loop.py`
- Verify: `tools/tests/test_vla82_predicate_stability.py`

**Interfaces:**
- Consumes: corrected generator and preflight from Tasks 1–2.
- Produces: a regenerated VLA82-037 asset and evidence JSON whose `model_sha256` matches the corrected MJCF.

- [ ] **Step 1: Materialize only VLA82-037 and print audited geometry**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -c "from tools.run_vla82_full_simulation import _load_compiled_specs,_scene_for_spec; from tools.vla82_full_sim.assets import materialize_custom_asset,validate_vla82_037_drawer_asset_xml; s=next(x for x in _load_compiled_specs() if x.selection_id=='VLA82-037'); a=materialize_custom_asset(_scene_for_spec(s).primary_asset); t=open(a.asset_path_or_group,encoding='utf-8').read(); print(validate_vla82_037_drawer_asset_xml(a,t)); print(a.evidence_path)"
```

Expected: `(0.095, 0.012, 0.012)` followed by the absolute `asset_evidence.json` path; runtime should be seconds, not minutes.

- [ ] **Step 2: Verify evidence hash and unchanged physics profile**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -c "import hashlib,json,pathlib; p=pathlib.Path(r'assets/vla82_source_textured/VLA82-037/model.xml'); e=json.loads(pathlib.Path(r'assets/vla82_source_textured/VLA82-037/asset_evidence.json').read_text(encoding='utf-8')); print(e['model_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()); print(e['dimensions']); print(e['physics_profile']['friction']); print(e['physics_profile']['density'])"
```

Expected:

```text
True
[0.012, 0.012, 0.095]
[0.45, 0.004, 0.0001]
300
```

- [ ] **Step 3: Run full source-spawn regression**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
```

Expected: all tests pass.

- [ ] **Step 4: Run strict closed-loop and predicate regression**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest -q tools/tests/test_vla82_pure_closed_loop.py tools/tests/test_vla82_predicate_stability.py
```

Expected: all tests pass with no predicate weakening.

---

### Task 4: Run One Physical Retest and Gate All Evidence Outputs

**Files:**
- Generate: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-037/episode-2000.json`
- Generate: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-037/episode-2000.npz`
- Conditionally generate: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos_clear_view/VLA82-037/episode-2000.strict-pass-replay-wide.mp4`
- Conditionally modify/create: strict-PASS video index workbook under `outputs/strict_pass_video_index_20260811/VLA82_严格PASS无遮挡结果视频路径汇总*`

**Interfaces:**
- Consumes: a GREEN preflight and regression suite.
- Produces: one fresh diagnostic episode; only a strict PASS may produce a result video and workbook row.

- [ ] **Step 1: Run exactly one full VLA82-037 attempt**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py collect-demos --episodes-per-object 1 --max-attempts-per-object 1 --selection-id VLA82-037
```

Expected minimum gate: result reports `steps > 0`; `collector_exception` and `PlacementError` are absent.

- [ ] **Step 2: Audit strict physical evidence before any export**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -c "import json,pathlib; p=pathlib.Path(r'outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-037/episode-2000.json'); j=json.loads(p.read_text(encoding='utf-8')); print({'status':j['status'],'steps':j['steps'],'predicate_success':j['predicate_success'],'errors':j['errors'],'phase_results':j.get('predicate',{}).get('phase_results',[]),'contact_verified':j.get('predicate',{}).get('contact_verified')})"
```

PASS gate: `status == "PASS"`, `predicate_success is True`, required phase result is successful, contact is verified, and `errors` is empty. If any condition fails, stop this task, retain diagnostics, do not export video, and do not update the workbook.

- [ ] **Step 3: Export an unobstructed replay only after strict PASS**

Run only after Step 2 passes:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\replay_vla82_strict_pass_video.py `
  outputs\midterm_testing_vla82\full_simulation_objectwise_8x60\expert_demos\VLA82-037\episode-2000.npz `
  outputs\midterm_testing_vla82\full_simulation_objectwise_8x60\expert_demos_clear_view\VLA82-037\episode-2000.strict-pass-replay-wide.mp4 `
  --fps 24 --width 1280 --height 720
```

Expected: a 1280×720 MP4 generated through public replay; the exporter refuses non-PASS evidence.

- [ ] **Step 4: Visually inspect start, grasp/transport, and final support frames**

Extract or inspect at least three frames and confirm: robot and drawer are visible, the gripper actually contacts the tongs, the tongs do not disappear or pass through furniture, transport is controlled, and the final object rests on the counter without camera obstruction. Reject the video if any condition is false.

- [ ] **Step 5: Update the strict-PASS workbook only after visual acceptance**

Use the existing artifact-tool workbook pattern from `tools/update_vla82_strict_pass_index_11.mjs`: append one VLA82-037 row with the absolute MP4, JSON, and NPZ paths; increment the PASS formulas; inspect the written table for formula errors; render a preview; and reopen the exported workbook to verify the saved values. Do not overwrite the established 11-row workbook unless all checks pass.

- [ ] **Step 6: Report the round**

Report one of two evidence-backed outcomes:

- Strict PASS: new total `12/60`, absolute JSON/NPZ/video/workbook paths, and the verified physical operation summary.
- Strict FAIL: total remains `11/60`, exact first failed physical stage and diagnostic JSON path; no result video and no workbook change.

