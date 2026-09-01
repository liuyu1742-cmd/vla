# OpenVLA 家政技能学习项目阶段路线图

**基准日期：** 2026-07-24  
**最终目标：** 与 Task-2 的 15 个任务类别、123 个候选物体和 138 条正式
relation 完全对齐；至少 120 个不同物体各有一条可访问的成功证据；15 类任务均有
成功关系；建立“人类视频动作捕获与解析 → Skill IR → OpenVLA / 机器人闭环”的
可审计链路。

## 当前实际状态

| 层级 | 当前结果 |
|---|---|
| OpenVLA 本地推理、CUDA、RoboCasa、IPC | 已打通 |
| 正式 relation | `organizing::toy` |
| L3 仿真闭环 | held-out seeds 101/102/103 全部成功 |
| 执行性质 | hybrid：OpenVLA + 状态感知定位恢复 + 抓取时序校准 + final-contact servo |
| pure autonomous VLA | 尚未通过，不能用 hybrid 结果代替 |
| 权威覆盖结构 | 15 个 Task-2 类别、123 个物体、138 条关系、1845 条显式矩阵行 |
| 已验证物体 | 1 / 120 最低目标 |
| 已验证 service task | 1 / 15 |
| 持久证据账本 | 已建立，重建注册表后证据不会丢失 |
| 人类视频原始关键点 readiness | 旧 EPIC 映射为 0；已有 keypoint smoke 仍不足 |

## 阶段 2：证据账本与覆盖闭环

**状态：已完成。**

交付物：

- `data/skill_coverage/evidence_ledger.json`
- `tools/skill_coverage/evidence_merge.py`
- `tools/merge_skill_coverage_evidence.py`
- evidence-preserving authoritative registry builder
- `organizing::toy -> organization_storage::toy` L3 hybrid 证据

验收结果：

```text
STRUCTURAL_VALIDATION: PASS
validated_objects=1
validated_tasks=1/15
remaining_validation_gap=119
```

## 阶段 3：首条真实人类视频技能迁移关系

**推荐 relation：** `organizing::storage_box`

选择理由：

- Task-2 本地目录已有 231 个 MP4、653 张图像和 1557 条标注；
- storage box 是可移动刚体，能复用 `locate → grasp → move → place` 技能族；
- 比 bookshelf、wardrobe、desk、cabinet 等“不可整体抓取”对象更符合当前 Panda
  机械臂和 RoboCasa 场景的物理约束；
- 完成后同时验证人类视频前端和第二个正式物体，而不是只做离线姿态演示。

### 3A：视频动作捕获与 Skill IR

1. 从只读目录选择连续、可解码、有完整手—物交互的 storage-box MP4；
2. 记录视频路径、大小、SHA-256、帧率和时间范围；
3. 用 GPU 运行人体/手部关键点与 storage-box 检测；
4. 保存非零原始关键点轨迹、物体轨迹、接触状态和置信度；
5. 自动解析 `locate/reach/grasp/lift/transport/place/release`；
6. 生成与 Task-2 完全一致的 `organizing::storage_box` Skill IR；
7. 输出 L2 视觉证据；缺关键点、接触或释放事件时必须失败，不能用 narration 补造。

**验收门槛：**

- 至少一段真实 MP4；
- 至少 30 个有效时序采样；
- 有连续手部关键点、物体轨迹、一次接触建立和一次接触释放；
- canonical actions 与冻结契约逐项相等；
- L2 证据可由独立工具重新验证。

### 3B：storage-box RoboCasa 环境与专家示教

1. 建立可移动 storage-box 资产和 cabinet/storage 目标；
2. 定义原生成功谓词：抓取、离台、进入收纳区、释放、仿真器成功；
3. 在非 held-out seeds 上运行专家并收集成功示教；
4. 增加尺寸、材质、初始位置和相机扰动；
5. 保留 held-out seeds 201/202/203，禁止进入训练或 DAgger。

**验收门槛：** 至少 30 个成功训练 seed，专家成功率不低于 95%，held-out 零重叠。

### 3C：共享 pick-place adapter 与 held-out 闭环

1. 从当前 DAgger-r3 adapter 继续训练，不重新训练整个 7B 基座；
2. toy 与 storage-box 混合采样，防止灾难性遗忘；
3. 先做离线动作探针，再做 300 步 held-out 三 seed；
4. 若需要 recovery/servo，继续透明记录为 hybrid；
5. 三 seed 全部原生谓词成功后导入 evidence ledger。

**完成标志：** validated objects 从 1 增至 2；`organizing` 技能族具有真实视频
L2 和仿真闭环 L3 的同 relation 追踪链。

## 阶段 4：建立六个可复用技能族

不能为 123 个物体各训练一个独立模型。应按动作原语和后端复用：

1. **刚体抓取运输：** locate、grasp、move、place、release；
2. **表面处理：** wipe、clean、scrub、rinse、dry；
3. **可动部件与按钮：** open、close、press、turn、lock；
4. **柔性物体：** fold、spread、smooth、straighten；
5. **液体与容器：** fill、pour、water；
6. **设备状态与监控：** turn_on/off、inspect、confirm_state、control。

每个技能族先选择一个物理可实现的 Task-2 relation 做 L2+L3 样板，然后再扩对象。
不可移动家具、整屋设施和虚拟设备不能硬套 Panda 抓取；应使用部件交互、
custom MuJoCo device 或 symbolic/device-state backend，并明确证据后端。

## 阶段 5：覆盖全部 15 类任务

按“每类先完成一个成功 relation”推进，优先顺序：

1. organizing / organization storage；
2. object fetching 与 food serving；
3. cleaning、window care、waste disposal；
4. appliance management 与 smart cooking；
5. laundry 与 clothing care；
6. maintenance、security monitoring；
7. bedroom service、elderly assistance、entertainment service。

每次通过后都使用 evidence ledger 导入，验收统计同时保留：

- Task-2 dataset task coverage：目标 15/15；
- Word/service-task coverage：目标 15/15；
- distinct validated objects：目标至少 120；
- L2、L3、L4 分级，不能混为一个成功数字。

## 阶段 6：扩展到至少 120 个物体

1. 为每个技能族建立共享训练集和对象条件；
2. 对物体形状、纹理、尺度、初始姿态、背景、光照和相机做域随机化；
3. 每个新物体至少三个未训练 seed；
4. 失败轨迹进入 DAgger recovery，但 held-out 永不进入训练；
5. 证据账本达到至少 120 个 distinct object；
6. 独立 validator 输出 `PROJECT_ACCEPTANCE: READY`。

目录存在、语义映射、检测准确率和训练 loss 均不能代替对象闭环成功。

## 阶段 7：减少 hybrid 依赖并评测纯 autonomous VLA

当前 state-aware recovery 与 final-contact servo 保证任务完成，但不属于纯
autonomous VLA。后续单独建立 autonomous 协议：

- OpenVLA 输入加入受控的机器人状态/短历史，而不是仅靠单帧猜阶段；
- 训练接近接触、遮挡和恢复状态；
- 禁用 expert recovery 和 final-contact servo 后重新评测；
- hybrid 成功率与 autonomous 成功率分表报告；
- autonomous 失败不撤销已有 L3 hybrid 证据，但不能冒充 autonomous 成功。

## 阶段 8：真机、安全与 L4

1. 接入真实相机、机器人本体和末端执行器；
2. 完成动作尺度、坐标系、延迟、急停、碰撞和力矩限制校准；
3. 仿真成功 relation 按风险分批迁移；
4. 人在回路验证后再进行无人值守测试；
5. L4 与 L3 证据分别保存。

## 推荐执行顺序

```text
证据账本（已完成）
  → organizing::storage_box 真实视频 L2
  → storage-box 仿真环境与专家数据
  → 共享 pick-place adapter + held-out L3
  → 六个技能族样板
  → 15 类任务至少各一条成功关系
  → 扩展到至少 120 个 distinct object
  → autonomous VLA 单独评测
  → 真机 L4
```
