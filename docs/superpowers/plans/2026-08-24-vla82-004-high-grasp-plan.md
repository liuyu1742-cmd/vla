# VLA82-004 High-Grasp Placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace VLA82-004's mid-body grasp with a physically verified high grasp that permits upright counter placement.

**Architecture:** Add a pure high-grasp target helper in the expert module, route only VLA82-004's cabinet approach through it, and retain the existing MuJoCo grasp, contact, upright, release, and stability predicates. Collect one new physical episode only after unit validation succeeds.

**Tech Stack:** Python 3.11, NumPy, RoboCasa, robosuite, MuJoCo, unittest.

## Global Constraints

- VLA82-004 is the only selection whose grasp target changes.
- Use public robot actions only; never write object pose or contact state.
- Do not update the strict PASS workbook unless the complete physical predicate passes.

---

### Task 1: Define and test the high bottle grasp target

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Modify: `tools/vla82_full_sim/expert.py`

**Interfaces:**
- Produces: `pick_place_high_grasp_target(selection_id: str, object_center: Sequence[float], object_rotation: Sequence[Sequence[float]], object_half_length: float) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
target = expert.pick_place_high_grasp_target(
    "VLA82-004", object_center=(1., 2., 3.), object_rotation=np.eye(3), object_half_length=.085,
)
np.testing.assert_allclose(target, (1., 2., 3.045))
np.testing.assert_allclose(
    expert.pick_place_high_grasp_target("VLA82-019", (1., 2., 3.), np.eye(3), .085),
    (1., 2., 3.),
)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_cabinet_spray_uses_high_grasp_target`

Expected: FAIL because `pick_place_high_grasp_target` is undefined.

- [ ] **Step 3: Write minimal implementation**

```python
def pick_place_high_grasp_target(selection_id, object_center, object_rotation, object_half_length):
    center = np.asarray(object_center, dtype=float)
    if str(selection_id) != "VLA82-004":
        return center.copy()
    axis = np.asarray(object_rotation, dtype=float).reshape(3, 3)[:, 2]
    return center + axis / np.linalg.norm(axis) * min(.045, float(object_half_length) - .015)
```

- [ ] **Step 4: Run test to verify it passes**

Run the command from Step 2. Expected: PASS.

### Task 2: Route only VLA82-004 through the high grasp

**Files:**
- Modify: `tools/vla82_full_sim/expert.py`
- Modify: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Consumes: `pick_place_high_grasp_target`
- Produces: cabinet approach targets centred on the high grasp point for VLA82-004 only.

- [ ] **Step 1: Write the failing selection-isolation test**

```python
self.assertTrue(expert.pick_place_uses_high_grasp_target("VLA82-004"))
self.assertFalse(expert.pick_place_uses_high_grasp_target("VLA82-017"))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_cabinet_spray_high_grasp_is_selection_scoped`

Expected: FAIL because the selection helper is undefined.

- [ ] **Step 3: Implement the minimum route**

Use the live collision geom rotation and half-length at the cabinet pad-centre approach. Replace the VLA82-004 approach `aligned_obj` with `pick_place_high_grasp_target(...)`; leave all other selections and all grasp/contact gates unchanged.

- [ ] **Step 4: Verify unit tests and compilation**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn tools.tests.test_vla82_predicate_stability`

Expected: targeted tests PASS.

### Task 3: Collect and verify one physical episode

**Files:**
- Use: `tmp/run_vla82_004_seed.py`
- Output: `outputs/midterm_testing_vla82/vla004_upright_repair_20260824/expert_demos/VLA82-004/`

- [ ] **Step 1: Run the VLA82-004 physical episode**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tmp\run_vla82_004_seed.py`

- [ ] **Step 2: Inspect predicate output**

Accept only `status=PASS`, a true predicate result, target contact, controlled release, stable frames, and upright cosine at or above `cos(12°)`.

- [ ] **Step 3: Generate clear video only after PASS**

Run `tools/replay_vla82_strict_pass_video.py` using the PASS NPZ and inspect the resulting clear, unobstructed video.

- [ ] **Step 4: Update the strict workbook only after video review**

Create a new 16-item workbook that cites the new valid video. Keep the existing 15-item workbook unchanged if Task 3 does not pass.
