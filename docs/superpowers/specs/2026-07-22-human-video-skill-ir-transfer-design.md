# 人类视频到机器人技能迁移设计

## 1. 目标

建立项目第一条可审计的“人类视频技能迁移”闭环：输入一段人拿起杯子并放入柜子的家庭视频，自动提取人体/手部与物体交互信息，解析成统一技能表示 `Skill IR`，再驱动现有 RoboCasa + OpenVLA 执行链完成等价机器人任务。

用户侧体验是“输入视频，得到机器人执行结果”；系统内部保留可验证的动作捕获、技能解析、机器人重定向、闭环执行和安全恢复模块。

## 2. 当前基线

- OpenVLA 7B、本地 CUDA 推理、双环境 TCP IPC 已可运行。
- RoboCasa 虚拟相机、PandaOmron 动作执行和反馈已可运行。
- `glass_cup → cabinet` 在 `seed 0` 和 held-out `seed 2` 上通过 `OpenVLA + 状态监督器` 闭环验收。
- 当前 EPIC 映射仅支持 `cup → PickPlaceCounterToCabinet / glass_cup`。
- 当前 `datasets/robocasa_vla/manifest.json` 的 episode 数量为 0；现有水杯轨迹位于独立训练目录，尚未纳入通用 VLA 数据契约。
- 当前状态监督器使用模拟器物体位姿、末端位姿和抓取状态，只能作为仿真里程碑和数据生成工具。

## 3. 范围

本里程碑包含：

1. 从一个本地 MP4 视频读取按时间排序的帧。
2. 读取或生成身体、手部关键点和置信度。
3. 检测并跟踪目标杯子，记录可见性、框/掩码和轨迹。
4. 识别手—物接触、抓取、搬运和释放事件。
5. 自动解析 `locate → reach → grasp → lift → transport → place → release` 阶段。
6. 生成并校验 `Skill IR v1`。
7. 把 `Skill IR` 映射到 RoboCasa 水杯放柜任务及阶段指令。
8. 使用当前 OpenVLA + 状态监督器执行，并保存源视频到机器人结果的完整证据链。

本里程碑不包含：

- 将人的三维骨骼坐标直接作为机器人 7D 动作标签。
- 真实机械臂控制或真实家庭环境成功率。
- 15 类任务和 120 类物体的全部实现。
- 仅凭单目 RGB 恢复精确抓取力、绝对尺度或可靠六自由度物体位姿。
- 宣称当前状态监督器已经摆脱模拟器真值。

## 4. 总体架构

系统分为两条通路，并通过 `Skill IR` 连接。

### 4.1 人类视频理解通路

`视频 → 人体/手部关键点 → 物体检测/跟踪 → 手—物交互事件 → 时序阶段 → Skill IR`

该通路回答：人在操作什么、以什么顺序操作、物体发生了什么状态变化。

### 4.2 机器人闭环执行通路

`Skill IR → 任务/对象映射 → 阶段规划 → OpenVLA 7D 动作 → 状态监督 → RoboCasa → 反馈/恢复`

该通路回答：机器人在当前画面和当前阶段下应该输出什么动作。

### 4.3 边界原则

- 人类视频提供任务语义、接触事件、阶段顺序和对象相对运动，不直接提供机器人本体动作真值。
- RoboCasa/机器人轨迹提供机器人视觉—动作监督。
- OpenVLA负责连续动作候选；阶段规划和安全监督不伪装成 OpenVLA 输出。
- 每次监督器干预必须记录原因、OpenVLA 原始动作和最终执行动作。

## 5. 数据接口

### 5.1 视频分析记录

每个采样时刻包含：

```json
{
  "frame_index": 120,
  "timestamp_s": 4.0,
  "body_keypoints": [[0.0, 0.0, 0.9]],
  "left_hand_keypoints": [[0.0, 0.0, 0.8]],
  "right_hand_keypoints": [[0.0, 0.0, 0.95]],
  "objects": [
    {
      "track_id": "cup_0",
      "category": "cup",
      "bbox_xyxy": [120, 80, 170, 160],
      "confidence": 0.93,
      "visible": true
    }
  ],
  "contacts": [
    {
      "hand": "right",
      "object_track_id": "cup_0",
      "state": "touching",
      "confidence": 0.86
    }
  ]
}
```

缺失关键点使用 `null` 和独立置信度表示，不用全零坐标伪装成有效观测。

### 5.2 Skill IR v1

```json
{
  "schema": "household_skill_ir_v1",
  "source": {
    "video_path": "absolute/path/to/clip.mp4",
    "clip_id": "P29_03",
    "fps": 30.0
  },
  "task": {
    "verb": "store_object",
    "object_category": "cup",
    "destination_category": "cabinet"
  },
  "phases": [
    {
      "index": 0,
      "skill": "locate",
      "object_track_id": "cup_0",
      "start_s": 0.0,
      "end_s": 1.2,
      "confidence": 0.91,
      "completion_predicate": "object_visible"
    },
    {
      "index": 1,
      "skill": "reach",
      "object_track_id": "cup_0",
      "start_s": 1.2,
      "end_s": 2.3,
      "confidence": 0.88,
      "completion_predicate": "hand_near_object"
    },
    {
      "index": 2,
      "skill": "grasp",
      "object_track_id": "cup_0",
      "start_s": 2.3,
      "end_s": 3.0,
      "confidence": 0.86,
      "completion_predicate": "object_grasped"
    },
    {
      "index": 3,
      "skill": "lift",
      "object_track_id": "cup_0",
      "start_s": 3.0,
      "end_s": 3.8,
      "confidence": 0.84,
      "completion_predicate": "object_lifted"
    },
    {
      "index": 4,
      "skill": "transport",
      "object_track_id": "cup_0",
      "target": "cabinet",
      "start_s": 3.8,
      "end_s": 5.4,
      "confidence": 0.82,
      "completion_predicate": "object_at_destination"
    },
    {
      "index": 5,
      "skill": "place",
      "object_track_id": "cup_0",
      "target": "cabinet",
      "start_s": 5.4,
      "end_s": 6.2,
      "confidence": 0.85,
      "completion_predicate": "object_inside_destination"
    },
    {
      "index": 6,
      "skill": "release",
      "object_track_id": "cup_0",
      "target": "cabinet",
      "start_s": 6.2,
      "end_s": 6.8,
      "confidence": 0.89,
      "completion_predicate": "gripper_open_and_object_stable"
    }
  ],
  "preconditions": ["cup_visible", "cabinet_available"],
  "success_conditions": ["cup_inside_cabinet", "cup_released"],
  "overall_confidence": 0.85
}
```

校验规则：

- `schema` 必须等于 `household_skill_ir_v1`。
- `source.video_path`、任务动词、对象和目标不能为空。
- phase index 从 0 连续递增，时间区间非负且不倒序。
- 当前水杯任务必须包含一次 `grasp`、一次 `place` 和一次 `release`。
- `place` 和 `release` 必须引用相同目标。
- phase 置信度和整体置信度位于 `[0, 1]`。
- 缺少强制阶段、对象跟踪丢失或置信度低于阈值时，不启动机器人执行。

## 6. 视频动作解析

### 6.1 输入

- 第一条闭环使用一个本地家庭视频或已选 EPIC clip。
- 适配器接受原始 MP4，以及当前动作解析项目已生成的视觉/时序特征目录。
- 原始帧和已有特征必须通过 frame index 或 timestamp 对齐。

### 6.2 事件推断

- `reach`：手—物距离持续下降且没有稳定接触。
- `grasp`：接触置信度越过阈值，随后物体与手同步运动。
- `lift`：抓取后物体出现持续上移或离开支撑面。
- `transport`：抓取保持期间物体朝目标区域移动。
- `place`：物体进入目标区域并停止被搬运。
- `release`：手—物接触结束，物体保持稳定。

事件使用时间窗口和滞回阈值，避免单帧抖动导致阶段反复切换。若单目视频无法恢复可靠深度，第一版使用二维相对运动和语义目标，不生成伪精确三维轨迹。

## 7. 人到机器人重定向

重定向分为语义和几何两级：

1. 语义级：`cup + cabinet + ordered phases` 映射到 RoboCasa `PickPlaceCounterToCabinet / glass_cup`。
2. 几何级：机器人从自身相机和状态提供器获取杯子、柜子和末端位姿；人的关节坐标不直接转换成机械臂关节或 7D 增量。

第一版沿用当前状态监督器保证成功，但执行报告必须写入 `uses_simulator_state_supervisor=true`。后续将同一状态接口替换为视觉物体位姿和机器人本体状态。

## 8. OpenVLA执行契约

每一步 OpenVLA 请求包含：

- 当前机器人相机图像。
- 当前 `Skill IR` phase 生成的阶段指令。
- 当前对象和目标语义。
- 后续版本加入末端/夹爪状态和短历史；第一版在报告中标记尚未输入模型。

每一步记录：

- OpenVLA 原始 7D 动作。
- 最终执行 7D 动作。
- 当前 phase。
- 状态监督器是否干预及原因。
- 抓取、抬升、放置和任务成功谓词。

## 9. 失败处理

- 视频文件不可读：终止，不生成 Skill IR。
- 关键点或目标物体连续丢失：生成分析失败报告，不启动机器人。
- 阶段顺序不合法：Skill IR 校验失败。
- 映射对象或目标不受 RoboCasa 支持：明确报告 unsupported mapping。
- OpenVLA 服务断开或返回非 7D 动作：停止仿真并保留已完成轨迹。
- 接触前闭爪、搬运中松开或动作严重偏离：状态监督器干预并记录。
- 抓取失败：回到 `reach/grasp`，重试次数达到上限后结束回合。

## 10. 首条闭环验收

### 10.1 视频侧

- 输入真实 MP4，而不是手写 JSON。
- 输出带时间戳和置信度的关键点、物体轨迹和接触事件。
- 自动生成合法 `Skill IR v1`。
- 至少正确包含 `reach、grasp、transport、place、release`。
- 报告能够定位每个 phase 对应的视频时间范围。

### 10.2 机器人侧

- 同一个 Skill IR 映射到现有水杯放柜任务。
- RoboCasa `seed 0` 成功。
- held-out `seed 2` 成功。
- 报告中 `ever_grasped=true`、`object_lift > 0.08 m`、`success=true`。
- 保存 OpenVLA 直接执行比例和监督器干预统计。

### 10.3 端到端证据

最终报告包含：

- 原始视频路径和 clip id。
- 视频分析产物路径。
- Skill IR 路径及校验结果。
- RoboCasa 任务、对象、目标和随机种子。
- OpenVLA adapter 和动作统计版本。
- seed 0/2 机器人执行报告路径。
- 明确区分视频语义迁移成功、仿真机器人执行成功和真实机器人尚未验证。

## 11. 向15类任务和120物体扩展

- 15类任务不实现为15套互不相干模型，而由原子技能图组合。
- 第一批原子技能为 `locate、reach、grasp、release、transport、place、open、close、push、pull、press、pour、wipe、fold、navigate`。
- 120物体通过对象注册表和操作属性分族，包括刚性容器、碗盘、餐具、带把手物体、盒类、柔性衣物、清洁工具、家具、家电和按钮/旋钮。
- 每个对象条目声明检测类别、抓取类型、允许技能、目标区域和执行后端。
- RoboCasa覆盖刚性厨房/家庭操作；柔性衣物、移动清洁和智能家电分别使用适合的执行后端，但共享 Skill IR 和评测契约。
- 人类视频通路保持后端无关，机器人执行适配器按任务选择机械臂、移动机器人、柔性物体或设备控制后端。

## 12. 后续演进

1. 用视觉检测/6D位姿和机器人本体状态替换模拟器真值监督器。
2. 扩展 `Skill IR` 对象和目标注册表，完成三对象×三目标的通用抓取放置。
3. 收集人类视频—Skill IR—机器人轨迹对，训练阶段条件化和短历史 OpenVLA adapter。
4. 将模块化系统产生的数据用于视频条件化 VLA，使用户接口逐步接近端到端，同时保留安全控制层。
5. 按任务×对象×场景×种子矩阵统计成功率和监督器干预率。
