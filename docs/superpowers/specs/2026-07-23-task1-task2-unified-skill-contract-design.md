# 任务一与任务二统一技能契约设计

**状态：** 已批准  
**日期：** 2026-07-23  
**范围：** `C:\OpenVLA-Simulator`（任务一）只读接入
`C:\RobotProject\RobotProject`（任务二）。

## 决策

两个项目共用一套任务、物体和操作语义：任务二负责人类视频/指令的动作解析，任务一
负责把同一动作序列转换为 OpenVLA 连续机械臂动作并在 RoboCasa/MuJoCo 或后续真机
闭环执行。共同主键为：

```text
relation_key = task_id::object_id
```

Excel 定义哪些任务—物体关系有效，任务二冻结动作基准定义每条关系的指令、目标和
标准动作序列。覆盖模型是 15 个任务、123 个唯一物体、138 条有效关系，不是 15×123
全组合。Word 第五章分类只保留为需求参考，不作为最终验收主键。

## 权威来源与已核验范围

任务二 Excel：

```text
C:\RobotProject\RobotProject\docs\家庭服务任务物体数据表.xlsx
sheet: 任务物体数据表
rows: 138
sha256: 23e39dd2803f3d90725169a6095f354b227feb4c1f54f92310679e84732e9ae0
```

任务二冻结动作基准：

```text
C:\RobotProject\RobotProject\datasets\acceptance_instruction_action_benchmark_15tasks.json
relations: 138
unique objects: 123
sha256: 97198b51f7b0ebf3e160bd6397fbaed70ae73cd966b431180a8431ad6b75e212
```

Excel 的 138 个 `(task_id, object)` 与基准的 138 个
`(acceptance_task, object)` 逐条相等，无缺失和额外关系。

| task_id | 中文名称 | 关系数 |
|---|---|---:|
| cleaning | 清洁任务 | 23 |
| organizing | 整理任务 | 8 |
| smart_cooking | 智能烹饪 | 8 |
| appliance_management | 家电综合管理 | 9 |
| security_monitoring | 智慧安防 | 8 |
| laundry | 衣物清洗 | 8 |
| waste_disposal | 垃圾处理 | 8 |
| clothing_care | 服装护理 | 8 |
| window_care | 门窗护理 | 8 |
| bedroom_service | 卧室服务 | 8 |
| food_serving | 送餐饮水 | 8 |
| object_fetching | 物品取送 | 8 |
| elderly_assistance | 健康辅助 | 8 |
| maintenance_management | 维护管理 | 10 |
| entertainment_service | 娱乐服务 | 8 |

14 个物体跨多个任务。`remote_control` 对应三个任务；其余重复物体对应两个任务。
跨任务物体包括 `air_conditioner`、`book`、`electric_curtain`、`fan`、
`fire_alarm`、`laptop`、`mobile_phone`、`refrigerator`、`remote_control`、
`robot_vacuum`、`smart_lock`、`television`、`toy`、`washing_machine`。

权威层级为：Excel 关系范围 → 冻结动作基准 → `household_task_catalog.json` →
`operation_plans.py` → 历史报告。旧 120/135/124 统计和兼容目录不参与当前验收。

## 漂移规则

任务一只读导入任务二文件并保存源路径、大小、修改时间和 SHA-256。同步必须断言：

```text
task_count == 15
unique_object_count == 123
relation_count == 138
excel_relation_keys == benchmark_relation_keys
relation_key == task_id + "::" + object_id
```

任一断言失败即为 `CONTRACT_DRIFT`，禁止继续训练、评测或写入覆盖通过状态。任务一不
写入或静默修正 `C:\RobotProject`。

## 统一 Skill IR

任务二输出、任务一消费 `household_skill_ir_v2`：

```json
{
  "schema_version": "household_skill_ir_v2",
  "contract_version": "task2_snapshot_<sha256-prefix>",
  "relation_key": "organizing::toy",
  "task_id": "organizing",
  "task_name": "整理任务",
  "object_id": "toy",
  "object_name": "玩具",
  "instruction": "整理任务：整理并归位玩具（toy）。",
  "target": "none",
  "canonical_actions": [
    "locate(toy)",
    "grasp(toy)",
    "move(storage)",
    "place(toy)"
  ],
  "source_video": null,
  "perception_evidence": null,
  "parser_evidence": null,
  "execution_request": null,
  "execution_evidence": null
}
```

`canonical_actions` 必须与冻结基准完全相等。任务一可以把一个规范动作展开为多个低层
阶段，但必须记录展开关系。恢复动作和安全干预单独保存，不能修改任务二标签。缺失
证据使用 `null` 和原因，不能用零数组或空记录伪装为已检测。

## 同物体多任务

```text
appliance_management::fan
  locate(fan) → turn_on(fan) → inspect(fan)

maintenance_management::fan
  locate(fan) → inspect(fan) → clean_blades(fan)
  → maintain(fan) → confirm_state(fan)
```

两条关系共享物体检测器和别名，但动作计划、执行器、成功条件和证据独立。系统不能仅
凭检测到 `fan` 决定操作，必须使用任务上下文或完整 `relation_key`。

## 两个项目的接口

任务二目标数据流：

```text
人类任务视频
  → 人体/手部关键点与物体检测
  → 手—物接触、相对运动和时序阶段
  → task/object/target 候选
  → relation_key
  → canonical_actions
  → household_skill_ir_v2
```

人体动作不直接复制为 Panda 关节角。视频提供语义、接触关系、阶段顺序、目标状态和
运动约束，机器人在自身运动学和碰撞约束下重新执行。

任务一数据流：

```text
Skill IR + 当前相机 + 机器人状态
  → 技能编排器选择当前规范动作
  → 阶段提示
  → OpenVLA 7D 动作
  → 尺度校准和安全约束
  → RoboCasa/MuJoCo step
  → 状态反馈、成功谓词和失败恢复
```

原始 OpenVLA 动作、尺度调整动作、最终执行动作和监督器干预必须分别记录。专家或
恢复器可以保护执行，但不能把专家动作冒充为 OpenVLA 独立成功。

## 动作原语与共享技能族

138 条关系当前包含 48 种原语，任务一按技能族复用策略和数据：

1. 感知确认：`locate`、`inspect`、`confirm_state`；
2. 抓取运输：`grasp`、`move`、`place`、`release`；
3. 表面处理：`wipe`、`clean`、`scrub`、`wash`、`rinse`、`dry`；
4. 可动部件：`open`、`lock`、`press`、`turn_on`、`turn_off`；
5. 柔性物体：`fold`、`spread`、`smooth`、`straighten`、`fluff`；
6. 液体供餐：`fill`、`pour`、`water`；
7. 设备交互：`aim`、`connect`、`control`、`play`、`adjust_volume`；
8. 专项维护：`clean_filter`、`clean_blades`、`check_seal`、`test_alarm` 等。

这样保持 138 条语义完全对应，同时在 123 个物体间迁移底层技能。

## 执行后端与证据等级

| 等级 | 含义 |
|---|---|
| L0 | 关系在 Excel/基准中有效 |
| L1 | 任务二能输出关系和规范动作 |
| L2 | 具有对象、手部、接触或时序视觉证据 |
| L3 | 在 RoboCasa/MuJoCo/设备模拟器闭环成功 |
| L4 | 在真实机器人或真实设备闭环成功 |

后端分为 `robocasa_openvla`、`mujoco_custom_device`、
`symbolic_interaction_sim` 和后续 `real_robot_adapter`。报告必须分别显示各等级覆盖，
不能用 L0 目录覆盖替代 L3/L4 成功。

## 旧水杯链路

`PickPlaceCounterToCabinet / glass_cup`“杯子入柜”不在 138 条正式关系中，只保留为
OpenVLA 服务、IPC、RoboCasa 相机/执行、尺度和恢复的冒烟回归测试。它不能计入任何
正式 `relation_key` 或 15/123/138 覆盖，也不能映射成非 Excel 主键。

旧 `service_tasks.json` 和 15×123 全组合矩阵降级为历史诊断或 Word 交叉参考；新的
构建、训练和验收入口不得依赖它们判断正式覆盖。

## 第一条正式闭环

```text
relation_key: organizing::toy
instruction: 整理任务：整理并归位玩具（toy）。
actions: locate(toy) → grasp(toy) → move(storage) → place(toy)
```

它属于 Excel 和动作基准，只有四个高复用原语，可复用现有抓取运输放置基础。
`storage` 映射为 RoboCasa 中明确的收纳区域、容器或抽屉。成功谓词为：

```text
ever_grasped(toy) == true
toy_lift > configured_minimum
inside(toy, storage_region) == true
gripper_released == true
stable_for_n_frames == true
canonical_action_progress == 4/4
```

后续优先实现 `object_fetching::remote_control`、
`entertainment_service::remote_control`、`appliance_management::fan` 和
`maintenance_management::fan`，验证同物体多任务不会混淆。

## 验收与真实边界

任务二现有证据必须分开表述：固定标准指令 552 项中 540 项原始整序列正确，97.83%；
独立人工同义盲测原始整序列准确率 65.94%；EPIC 视频严格迁移中姿态可用与严格通过均
为 0。当前不能宣称任意视频已可直接驱动机器人。

契约验收要求 15/123/138 完整一致。动作解析以原始预测动作严格整序列准确率计分，
不能用约束后结果代替。仿真回合必须携带有效 `relation_key`/Skill IR，保存原始动作、
干预和恢复；成功由关系级状态谓词决定，并区分训练种子与未训练种子的 300 步长回合。

最终总表至少报告任务覆盖 15/15、唯一物体覆盖 123/123、关系契约 138/138、固定指令
准确率、人工盲测准确率、视频视觉证据覆盖率、L3 仿真成功数和 L4 真机成功数。

## 失败与测试

失败状态包括 `CONTRACT_DRIFT`、`MISSING_CANONICAL_PLAN`、
`OUT_OF_SCOPE_RELATION`、`INSUFFICIENT_VISUAL_EVIDENCE` 和
`UNSUPPORTED_PRIMITIVE`。不支持的关系保留 L0/L1，不能伪造 L3。

自动测试覆盖：15/123/138 和哈希、跨任务物体不丢失、同物体动作不混淆、Skill IR
Schema、旧水杯不能计入覆盖、`organizing::toy` 集成、原语到阶段映射、RoboCasa
成功/恢复/超时/IPC 中断，以及证据汇总不得用目录存在或训练损失替代真实指标。

## 实施顺序

1. 导入并冻结任务二 15/123/138 契约；
2. 建立 `household_skill_ir_v2` 和验证器；
3. 将旧水杯链路降级为非正式冒烟测试；
4. 建立 `organizing::toy` 环境映射、成功谓词和正式证据；
5. 接通任务二动作输入、OpenVLA 执行与 300 步评测；
6. 实现 `remote_control` 和 `fan` 的同物体多任务对照；
7. 按共享技能族扩展到全部 138 条关系；
8. 补齐视频姿态、手—物接触和时序解析；
9. 最后接入真实机器人与真实家电，升级 L4 证据。

后续计划和代码必须以本设计为准。138 条关系外的演示可保留为研究或回归测试，但不能
替代统一验收范围。
