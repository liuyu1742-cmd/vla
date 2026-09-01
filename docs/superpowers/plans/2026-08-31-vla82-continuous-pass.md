# VLA82 Continuous Strict PASS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 VLA82-055 长迟滞并新增一个具有正确操作语义和可辨外观的严格 PASS。

**Architecture:** 保留现有 RoboCasa/MuJoCo 专家控制与严格证据链，只增加选择项专用门限和候选专用资产/控制参数。所有生产改动先由失败测试约束，再通过新物理回放和视频画面验收。

**Tech Stack:** Python、RoboCasa、robosuite、MuJoCo、pytest、FFmpeg。

## Global Constraints

- 不使用真实机械臂，不使用瞬移或直接写 MuJoCo 状态。
- 输出视频为 1280×720、正向、无遮挡。
- 新增视频和 JSON 放入 `C:\OpenVLA-Simulator\datasets\VLA82 video`。
- 候选操作必须先从正式操作表确认，目标设施不得用错误物体替代。

---

### Task 1: VLA82-055 连续轨迹

**Files:**
- Modify: `tools/vla82_full_sim/expert.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Consumes: `pick_place_transit_ready_for_selection(selection_id, eef, desired, object_center, release)`
- Produces: VLA82-055 在 15 mm 内进入放置姿态校正，其他选择项行为不变。

- [ ] 写入 14 mm 位置误差应视为 VLA82-055 搬运到位的失败测试。
- [ ] 运行该测试并确认旧实现按 10 mm 门限失败。
- [ ] 将 VLA82-055 专用门限改为 15 mm。
- [ ] 运行聚焦测试并重新采集 seed 2000 轨迹。
- [ ] 验证步骤数、状态持续时间、姿态、碰撞和严格 PASS。
- [ ] 生成 1280×720 视频并逐帧抽查，替换正式视频与 JSON。

### Task 2: 新增候选筛选与语义确认

**Files:**
- Read: `configs/vla82_full_simulation/operation_specs.json`
- Read: `configs/vla82_full_simulation/authoritative_asset_manifest.json`
- Read: `outputs/midterm_testing_vla82/**/episode-*.json`

**Interfaces:**
- Consumes: 当前严格 PASS ID 集合与未通过轨迹。
- Produces: 一个操作、源物体、目标设施均明确的最高优先级候选。

- [ ] 排除已 PASS 的 25 项。
- [ ] 读取剩余候选的真实操作文本与源路径。
- [ ] 比较最近失败阶段、可达性、抓取几何和目标复杂度。
- [ ] 选择无需错误目标替代且最有可能一次闭环完成的候选。

### Task 3: 候选外观与控制闭环

**Files:**
- Modify: `tools/vla82_full_sim/assets.py` 或 `assets/vla82_source_textured/<ID>/model.xml`
- Modify: `tools/vla82_full_sim/expert.py`（仅在诊断证明需要时）
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Consumes: Task 2 的选择项、操作文本与目标设施。
- Produces: 可辨视觉资产、匹配碰撞几何和完成操作的专家轨迹。

- [ ] 为资产类别和关键外观部件写失败测试。
- [ ] 实现最小外观与碰撞几何改动并使测试通过。
- [ ] 对当前失败轨迹按选择项专用原因编写失败测试。
- [ ] 实现最小控制修复并运行聚焦测试。
- [ ] 运行物理仿真；低价值方向及时切换，不无限重试。

### Task 4: 严格验证与归档

**Files:**
- Create: `datasets/VLA82 video/<ID>_*_严格PASS_结果视频.mp4`
- Create: `datasets/VLA82 video/<ID>_*_严格PASS_判定.json`
- Modify: `outputs/strict_pass_video_index_20260811/*.xlsx`

**Interfaces:**
- Consumes: PASS 轨迹 NPZ/JSON。
- Produces: 新增严格 PASS 的正式视频、JSON 与更新汇总表。

- [ ] 验证 JSON PASS、predicate、物理阶段、接触、稳定和无碰撞条件。
- [ ] 生成 1280×720 正向无遮挡视频并检查首帧、中帧、末帧。
- [ ] 将视频和判定 JSON 放入正式目录。
- [ ] 更新汇总表并核对路径存在。
- [ ] 运行完成前验证并只报告真实新增结果。
