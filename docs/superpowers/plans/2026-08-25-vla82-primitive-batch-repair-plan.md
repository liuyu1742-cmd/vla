# VLA82 操作原语批量修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以重置期正常布局配置提高 VLA82 真实物理操作的严格通过率。

**Architecture:** 在环境配置层定义选择项到操作原语的布局策略；专家控制器继续仅通过公开 `env.step` 执行动作。每个原语的结果由既有物理谓词和视频复核。

**Tech Stack:** Python、RoboCasa、MuJoCo、unittest。

## Global Constraints

- 不在运行期直接写入物体、门或任务完成状态。
- 仅更新通过完整物理谓词验证的严格 PASS 记录。

---

### Task 1: 建立重置期布局策略

**Files:**
- Modify: `tools/vla82_full_sim/environment.py`
- Test: `tools/tests/test_vla82_source_spawn.py`

- [ ] 为每个操作原语编写失败测试，验证配置只作用于预重置采样器。
- [ ] 实现最小的原语布局选择器，保留源物体、目标夹具和真实操作语义。
- [ ] 用固定种子创建代表场景，验证物体位于可达支撑区域且门具有合理初始开度。

### Task 2: 按原语复核

**Files:**
- Modify: `tmp/run_vla82_034_candidate.py`
- Output: `outputs/midterm_testing_vla82/`

- [ ] 依次运行开放台面、抽屉、门体和清洁代表物体。
- [ ] 对每个通过项生成无阻挡高清视频。
- [ ] 仅将谓词、接触和视频均通过的项写入严格 PASS 索引。
