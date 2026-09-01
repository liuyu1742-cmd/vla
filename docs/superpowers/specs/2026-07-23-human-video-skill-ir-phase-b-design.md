# 人类视频到 Skill IR 与 OpenVLA 闭环：阶段 B 绑定设计

## 1. 目标

本阶段交付第一条可审计的人类视频技能迁移链：真实 EPIC MP4 自动产生视觉证据和
`household_skill_ir_v1`，同一个 Skill IR 再驱动现有
`PickPlaceCounterToCabinet / glass_cup` OpenVLA + RoboCasa 闭环，在 `seed 0`
和 held-out `seed 2` 上完成拿杯入柜。

本阶段不把人类二维关节点当作机器人 7D 动作标签，不训练“视频直接输出电机参数”
模型，也不宣称真实机械臂已经验证。它为后续多对象、多原语训练建立真实输入、统一
中间表示和端到端证据格式。

## 2. 已核验事实

- 本机有 737 个 EPIC 480p 动作片段和 2 个 720p 合并片段。
- 本机现有 `P29_03_clip_0014.mp4` 对应海绵和壶盖清洗，不是杯子任务。
- 现有 `epic_pose_evidence.json` 中 `pose_available=0`，原因是缺少本地姿态权重。
- 原动作解析项目 `.venv` 已有 CUDA PyTorch 2.7.1、Ultralytics 8.4.75、OpenCV，
  RTX 3090 可用。
- 已训练的 16 类家庭物体 YOLO 权重包含 `cup` 类别。
- 现有 OpenVLA 水杯 adapter 和受监督执行器已经保存 `seed 0/2` 成功报告，但尚未
  接受 Skill IR 作为运行输入。

这些事实意味着旧的“P29_03 已完成姿态解析”说法不能沿用。本阶段必须下载正确的
两个官方小片段、补齐手部关键点能力，并生成新的视觉与执行证据。

## 3. 方案比较与选择

### 方案 A：人体骨架直接重定向为机械臂动作

优点是流程直观。缺点是 EPIC 为第一视角，完整人体经常不可见；单目尺度、人体与
PandaOmron 运动学和夹爪状态也不一致。该方案不能可靠产生机器人动作真值，不采用。

### 方案 B：视频直接条件化端到端 VLA

长期最接近“输入视频直接产生动作”。但当前没有成规模的人类视频—机器人动作配对
数据，也没有足够多任务失败恢复样本。立即采用会把数据不足误当成模型问题，不作为
当前阻塞项。

### 方案 C：视频语义与接触解析 + Skill IR + 机器人闭环策略

视频提供对象、目标、手—物接触、阶段顺序和相对运动；机器人专家/监督器和 OpenVLA
在机器人自身视觉与动作空间中执行。该方案既满足视觉动作捕获与技能迁移要求，又能
复用现有成功水杯后端，并为以后训练视频条件化 VLA 沉淀配对数据，因此采用方案 C。

## 4. 输入示范与只读策略

输入固定为同一 EPIC 原视频 `P04_113` 中的连续语义对：

- `epic_P04_113_63`：`take cup`，`pick up glass mug`；
- `epic_P04_113_64`：`insert cup`，`put glass mug in cupboard`。

两个文件从官方 EPIC 预切片镜像下载到：

```text
C:\OpenVLA-Simulator\datasets\human_video_skill_ir\source_clips
```

项目记录 URL、EPIC clip ID、大小和 SHA-256，再在工作区合成为一个连续 MP4，并保存
源片段到合成时间轴的映射。不得写入、移动或删除
`C:\RobotProject\RobotProject\datasets` 下的任何文件。

## 5. 组件与接口

### 5.1 数据契约

`tools/human_video_skill_ir/contracts.py` 定义并校验：

- `VideoFrameEvidence`：帧号、时间、手部/身体关键点、物体检测、跟踪与接触；
- `SkillPhase`：阶段、时间区间、置信度、对象、目标和完成谓词；
- `SkillIR`：源视频、任务、阶段、前置条件、成功条件、失败原因和版本；
- `EndToEndEvidence`：视觉产物、映射、模型、种子、动作报告和真实性声明。

缺失视觉信息使用 `null` 和明确原因。零数组不能表示未检测到的关键点。

### 5.2 视频物化与时间轴

物化器只下载两个锁定 clip，进行 MP4 解码检查和 SHA-256 校验。合成器输出一个 MP4
和时间映射 JSON。重复运行时若哈希一致则复用；哈希不一致时失败，不静默覆盖。

### 5.3 视频视觉分析

视觉分析在原动作解析 `.venv` 中运行：

- 家庭 YOLO 在 CUDA 上检测 `cup`；
- 手部模型输出可见手的二维关键点与置信度；
- OpenCV 光流和 IoU/中心距离维持杯子轨迹；
- 手部关键点到杯框的距离、杯子与手的共运动和滞回窗口产生
  `approaching/touching/held/released` 接触状态；
- 第一视角中不可见的完整人体骨架保留为 `null`，不阻止手—物证据充分的片段。

EPIC narration 只作为对象和目标的语义先验，不直接填写视觉接触结果或阶段边界。

### 5.4 阶段解析与 Skill IR

阶段解析器从视觉时序自动产生：

```text
locate → reach → grasp → lift → transport → place → release
```

边界采用连续帧窗口和滞回，避免单帧抖动。至少需要：杯子在足够帧中可见、至少一只手
存在有效轨迹、一次接触建立、接触期间的杯手共运动、目标段中的接触释放。强制阶段缺失
时输出分析失败报告，不生成“通过”的 Skill IR。

### 5.5 RoboCasa 映射和 OpenVLA 执行

`cup + cupboard` 映射到：

```text
task: PickPlaceCounterToCabinet
object_group: glass_cup
service_task_id: item_delivery
coverage_object_id: water_cup
```

运行器读取 Skill IR，而不是使用硬编码的完整任务阶段列表。Skill IR 决定允许的阶段、
对象、目标和阶段提示；RoboCasa 相机与 OpenVLA 输出动作。现有状态监督器仍负责安全、
恢复和阶段完成判断，报告必须写入 `uses_simulator_state_supervisor=true`，并分别记录
OpenVLA 原始动作和最终执行动作。

## 6. 数据流

```text
官方 EPIC clips
  → 哈希校验与合成视频
  → 杯子检测 + 手部关键点 + 跟踪/光流
  → 手—杯接触事件
  → 自动阶段解析
  → Skill IR v1 校验
  → RoboCasa 任务映射
  → OpenVLA 服务 + Skill IR 受监督运行器
  → seed 0/2 报告
  → 端到端证据汇总
```

## 7. 失败与恢复

- 下载失败：保留 `.part`，支持续传；不生成分析产物。
- MP4 无法解码或哈希变化：失败并报告确切文件。
- 无杯子检测：降低到配置允许的最低阈值重试一次；仍无检测则失败。
- 无手部关键点或无接触证据：失败，不用 narration 伪造。
- 阶段顺序非法或置信度过低：Skill IR 校验失败，禁止启动仿真。
- OpenVLA 服务不可达、动作不是 7D 或 NaN：停止回合并保存部分轨迹。
- 抓取失败：由监督器按 Skill IR 的恢复边回到 `reach/grasp`；超过上限则失败。
- 任一种子失败：端到端状态为 `NOT_READY`，保留诊断证据，不能写入覆盖通过状态。

## 8. 测试与验收

### 8.1 单元和集成测试

- 契约拒绝缺失视频、非法时间线、伪造零关键点和缺少强制阶段；
- 下载器拒绝未知 clip 和哈希不一致；
- 合成时间轴可逆映射到两个源 clip；
- 合成视觉夹具验证接触滞回、抓取、搬运和释放阶段；
- 映射器只接受受支持的 `cup + cupboard`；
- 运行器验证 Skill IR 阶段提示、原始/执行动作和监督器记录均存在；
- 端到端汇总拒绝缺失视觉证据或失败种子。

### 8.2 首条闭环验收门槛

1. 输入是实际 MP4，不是手写 JSON；
2. 杯子检测、手部关键点、对象轨迹和接触事件均有非空证据；
3. 自动生成的 Skill IR 通过 Schema 和阶段顺序校验；
4. 同一个 Skill IR 驱动 `seed 0` 和 `seed 2`；
5. 两个报告均满足 `ever_grasped=true`、`object_lift > 0.08`、`success=true`；
6. 报告保存 OpenVLA 直接执行比例和监督器干预统计；
7. 最终报告明确区分视频语义迁移成功、仿真执行成功和真实机器人未验证。

满足以上七项后，阶段 B 才标记完成。训练损失下降、语义映射成功或已有旧水杯报告均不能
单独替代该阶段的端到端验收。

## 9. 环境与 PyCharm

- 视频分析解释器：`C:\RobotProject\RobotProject\.venv\Scripts\python.exe`；
- OpenVLA 服务解释器：`C:\Users\sjtu101\miniconda3\envs\openvla\python.exe`；
- RoboCasa 执行解释器：`C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe`；
- 三个配置的工作目录统一为 `C:\OpenVLA-Simulator`。

视频分析、模型服务和仿真执行通过文件契约与 TCP 连接，不在同一 Python 进程混装依赖。
