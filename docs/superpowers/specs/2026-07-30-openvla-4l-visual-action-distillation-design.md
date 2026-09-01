# OpenVLA-4L 视觉—动作蒸馏设计

## 目标

在单张 NVIDIA GeForce RTX 3090 上，针对 RoboCasa
`PickPlaceCounterToCabinet/glass_cup` 任务，对项目适配的 OpenVLA-4L
进行视觉—动作蒸馏。优先获得无专家恢复、无规则控制器介入的纯学生闭环
非零成功率；若多轮蒸馏后纯策略仍为零，则启用明确披露的专家恢复兜底，
保证混合闭环成功率非零。

## 验收协议

- 训练起点：现有 OpenVLA-4L 基座。
- 名义演示：seed 0、1、2、3、5 的五条成功专家演示。
- DAgger 训练种子：0、1、3、5；seed 2 保持为未参与 DAgger 的诊断种子。
- 纯策略评测：学生模型独立输出七维动作，无专家恢复、无规则控制器。
- 第一成功门：LoRA（rank=32）在五个评测种子中至少成功 1 回合。
- 公平对比：确定蒸馏数据后，用同一数据训练 Full Fine-tuning、
  LoRA（rank=32）、Last-Layer-Only 和 Frozen-Vision，并为每种方法
  输出数值成功率。
- 兜底门：若某方法纯策略仍为零，使用相同状态专家恢复协议重新评测，
  单列“混合闭环成功率”，不得写成纯 OpenVLA 成功率。

## 推荐架构

### 1. 学生状态采集

复用 `tools.collect_water_cup_dagger_v2`。每个时刻由 OpenVLA-4L 预测
动作，状态教师根据仿真中的末端位置、物体位置、抓取状态和柜体几何
生成动作标签。动作执行采用课程式混合：

- round 0：beta=0.7，优先得到完整成功轨迹；
- round 1：beta=0.3，扩大策略访问状态覆盖；
- round 2：beta=0.0，仅保留强制安全阶段和夹爪门控，使数据集中于
  学生真实偏离状态。

每轮都保存 RGB 帧、教师动作、学生动作、实际执行动作、阶段、抓取状态、
物体抬升和末端—物体距离。

### 2. 蒸馏数据构建

将五条 stride=1 名义演示与 DAgger 教师标签合并。每个种子输出一个
学生训练文件，使既有四方法训练器仍使用完全相同的数据接口。数据构建
需满足：

- 教师动作是唯一监督标签；
- 按文件内容哈希去重，不重复计入中断后重跑的回合；
- 动作阶段分为 open_motion、closed_motion、settle；
- settle 样本最多占 15%，避免再次出现静止动作主导；
- 接近和抓取纠偏样本至少占 50%；
- 生成可审计 manifest，记录每个来源文件、样本数和阶段计数。

### 3. 学生训练与迭代

先训练 LoRA（rank=32）作为快速探针。每轮使用 BF16、Adafactor、
梯度检查点、effective batch size=16，并保存训练过程图。若五种子纯策略
成功率仍为零，读取几何诊断：

- 最小末端—物体距离大于 0.06m：继续增加 locate/approach 纠偏；
- 已接触但从未抓取：增加 gripper-close 接触窗口样本；
- 已抓取但未放置：增加 transport/place 样本；
- 动作长时间为零：降低 settle 配额并提高运动样本权重。

一旦 LoRA 达到至少 1/5，冻结蒸馏数据版本，并用同一版本训练另外三种
微调方法。

### 4. 混合恢复兜底

若在最多三轮 DAgger 后某方法仍为 0%，启用项目已验证的状态专家恢复。
恢复只在以下情况下介入：

- 连续 20 步末端—物体距离无改善；
- 接近阶段错误闭合夹爪；
- 抓取后物体脱落；
- 放置阶段超过时限。

报告必须同时给出 `pure_success_rate`、`hybrid_success_rate`、
`expert_recovery_steps` 和 `expert_recovery_ratio`。过程图需标出每个
关键帧由学生还是专家控制。

## 文件边界

- `tools/openvla_4l_distillation_dataset.py`：去重、阶段限额和合并。
- `tools/openvla_4l_distillation_pipeline.py`：轮次选择、停止条件和报告。
- `tools/openvla_4l_hybrid_eval.py`：纯策略失败后的恢复评测。
- `tools/run_openvla_4l_distillation_round.ps1`：模型服务与采集编排。
- `tests/test_openvla_4l_distillation_dataset.py`：数据合同。
- `tests/test_openvla_4l_distillation_pipeline.py`：停止条件。

## 错误处理

- 推理服务必须等待 READY 日志，不能只检查端口监听。
- 单个种子采集失败时保留日志并重试；同一错误连续三次后全面排查，
  修复后继续任务。
- 数据集构建发现非有限动作、七维动作不完整、无运动阶段或来源哈希重复
  时立即拒绝训练。
- 纯策略与混合策略使用不同输出目录和报告 schema，防止结果混淆。

## 证据与最终交付

- 每轮采集报告和关键帧；
- 每种方法训练损失、GPU显存和检查点；
- 每种方法纯策略闭环过程图；
- 必要时增加混合恢复过程图；
- 重新生成完整 4.6.1，明确数据版本、纯策略成功率和混合成功率。
