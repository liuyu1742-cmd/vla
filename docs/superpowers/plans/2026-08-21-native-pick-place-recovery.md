# 原生抓取—放置恢复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以真实 MuJoCo 接触与物理运动改善原生抓取—放置任务的可达性。

**Architecture:** 对非抽屉、非家具关节的原生物体使用重置时源侧底盘锚定与双指中心接近门槛。控制器、判定器和录像器保持原状，仍由真实物理状态决定 PASS。

**Tech Stack:** Python 3.11、RoboCasa、robosuite、MuJoCo、unittest。

## Global Constraints

- 禁止 `qpos`、物体位姿或家具关节的直接状态写入。
- PASS 必须保留抓取接触、目标接触/关系、稳定帧与无遮挡视频。
- 不修改现有 15 项严格 PASS 的证据路径。

---

### Task 1: 确定恢复配置选择条件

**Files:**
- Modify: `tools/vla82_full_sim/environment.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Produces: `uses_native_pick_place_source_anchor(request) -> bool`

- [ ] **Step 1: Write the failing test**

```python
assert uses_native_pick_place_source_anchor(native_counter_to_cabinet)
assert not uses_native_pick_place_source_anchor(drawer_source)
assert not uses_native_pick_place_source_anchor(fixture_request)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_native_pick_place_anchor_scope`

- [ ] **Step 3: Implement the minimal predicate**

```python
return request.primary_asset.kind == "robocasa_native" and request.source_fixture != "drawer"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_native_pick_place_anchor_scope`

### Task 2: 使用双指中心确认接近状态

**Files:**
- Modify: `tools/vla82_full_sim/expert.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Produces: `pick_place_pregrasp_reached(pad_center, target, tolerance) -> bool`

- [ ] **Step 1: Write the failing test**

```python
assert pick_place_pregrasp_reached((.0, .0, 1.1), (.0, .0, 1.12), .03)
assert not pick_place_pregrasp_reached((.0, .0, 1.1), (.0, .0, 1.16), .03)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tools.tests.test_vla82_source_spawn.SourceSpawnReachabilityTest.test_pick_place_pregrasp_uses_pad_center`

- [ ] **Step 3: Implement the minimal helper and use it in approach transition**

```python
return float(np.linalg.norm(np.asarray(pad_center) - np.asarray(target))) <= tolerance
```

- [ ] **Step 4: Run focused tests and a VLA82-012 physical episode**

Run: `python -m unittest tools.tests.test_vla82_source_spawn`

### Task 3: 严格验证和证据更新

**Files:**
- Output: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-012/episode-2000.json`
- Output: `outputs/strict_pass_video_index_20260811/`

- [ ] **Step 1: Run a single real simulation**

Run: `python tools/run_vla82_full_simulation.py collect-demos --selection-id VLA82-012 --episodes-per-object 1 --max-attempts-per-object 1`

- [ ] **Step 2: Verify gates**

Require status `PASS`, predicate success, non-empty contact evidence, stable target relation and a clear result video.

- [ ] **Step 3: Update strict-PASS workbook only if all gates pass**

Preserve all existing rows and add the final MP4/JSON/NPZ paths for the new selection.
