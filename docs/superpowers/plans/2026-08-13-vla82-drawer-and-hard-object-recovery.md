# VLA82 Drawer and Hard Object Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按既定顺序生成物理真实、无遮挡、无碰撞和无明显停顿的严格 PASS 操作证据。

**Architecture:** 先把抽屉放置拆为预对齐、锁定插入和垂直撤离三个可独立验证的阶段，再处理资产真实性和困难抓取候选。严格判定同时读取状态机、接触记录、物体轨迹和视频，不接受谓词误报。

**Tech Stack:** Python 3.11、MuJoCo、RoboCasa/robosuite、NumPy、FFmpeg、unittest。

## Global Constraints

- 执行顺序固定为 045 → 005 → 042 → 004/017/020 → 007 → 038。
- 每项最多五次有明确假设的完整修复运行。
- 视频必须 1280×720、无遮挡，且与本次通过轨迹一一对应。
- 进入抽屉开口后底盘不得移动，夹爪不得接触抽屉结构。

---

### Task 1: VLA82-045 固定路点插入

**Files:**
- Modify: `tools/vla82_full_sim/expert.py`
- Modify: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Consumes: `drawer_transit_control_stage(...)` 和抽屉几何边界。
- Produces: 045 的底盘锁定判断、直接插入速度及阶段预算函数。

- [ ] 写失败测试：045 到达抽屉开口后必须锁定底盘，插入阶段不得回到姿态修正。
- [ ] 运行定向 unittest，确认因缺少行为而失败。
- [ ] 实现最小控制器改动：固定上方路点、锁定插入、设置运动速度下限。
- [ ] 运行定向测试及 72 项控制器回归测试。
- [ ] 完整运行一次 045，检查接触、状态变化与视频关键帧；不合格时仅按单一新假设迭代。

### Task 2: VLA82-005 夹爪避碰放置

**Files:**
- Modify: `tools/vla82_full_sim/expert.py`
- Modify: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Consumes: Task 1 的锁定插入阶段。
- Produces: 手柄端抓取偏置和夹爪—抽屉组合余量检查。

- [ ] 写失败测试：005 夹爪中心不得下降到抽屉边沿以下，物体中心仍能到达内部支撑面。
- [ ] 验证测试按预期失败。
- [ ] 实现手柄端偏置、上方预对齐和先垂直撤离。
- [ ] 运行回归测试和完整仿真。
- [ ] 检查视频无边沿碰撞、无长时间微动后生成 720p 结果。

### Task 3: VLA82-042 可辨识洗碗刷

**Files:**
- Modify: `assets/vla82_source_textured/VLA82-042/model.xml`
- Modify: `configs/vla82_full_simulation/object_assets.json`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Produces: 含手柄、刷头、刷毛区的简化物理资产。

- [ ] 写资产结构测试，要求三个可辨识视觉组成部分。
- [ ] 验证测试失败。
- [ ] 修改 XML 并保持碰撞体稳定、质量和摩擦符合清洁工具。
- [ ] 运行资产审计、完整仿真和视频检查。

### Task 4: 004/017/020 离线抓取候选

**Files:**
- Modify: `tools/vla82_full_sim/expert.py`
- Create: `tools/vla82_grasp_candidate_probe.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Produces: 每个物体至多一组经短时物理探针验证的抓取位姿。

- [ ] 测试候选筛选会拒绝推碰、非双指接触及 50 mm 抬升滑脱。
- [ ] 实现有限候选生成和确定性评分。
- [ ] 每项用最佳候选运行完整轨迹，最多五次。
- [ ] 仅为真实严格 PASS 生成视频。

### Task 5: 007 与 038 专项轨迹

**Files:**
- Modify: `tools/vla82_full_sim/expert.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Produces: 007 的按钮后撤离阶段，以及 038 的锁底盘三维安全通道。

- [ ] 测试 007 按钮完成后必须先撤离锅体区域。
- [ ] 测试 038 插入阶段底盘动作恒为零且抽屉关节不反向回弹。
- [ ] 分别实现并运行回归测试。
- [ ] 每项最多五次完整仿真并检查高清视频。

### Task 6: 证据与 Excel 收敛

**Files:**
- Modify: `outputs/strict_pass_video_index_20260811/VLA82_严格PASS无遮挡结果视频路径汇总_最终核验版.xlsx`

**Interfaces:**
- Consumes: 所有新严格 PASS 视频、JSON 与 NPZ。
- Produces: 仅含已复核结果的最终路径表。

- [ ] 校验所有视频、JSON、NPZ 路径存在。
- [ ] 对视频首段、中段、释放段和末段抽帧检查。
- [ ] 更新 Excel，扫描公式错误和失效路径。
- [ ] 输出通过、放弃及原因汇总。

