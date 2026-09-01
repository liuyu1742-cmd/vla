# `organizing::toy` 正式技能运行手册

## 目标与验收口径

首个正式任务不是旧的水杯基础烟测，而是任务二契约中的
`organizing::toy`：机器人从自身相机画面出发，将玩具抓起并放入柜子。

正式成功必须同时满足：

- RoboCasa 原生成功谓词为真；
- 曾经形成真实双指抓取；
- 玩具进入柜体内部；
- 最终松开夹爪；
- 300 次策略决策内完成；
- 评测 seed 没有进入训练集；
- 报告中的数据清单、Skill IR 和适配器都有 SHA256 指纹。

## 已固化的数据边界

- 训练 seed：`0..11`，12 条成功示教，4042 帧；
- 未见 seed：`101, 102, 103`，3 条成功专家参考，969 帧；
- 训练清单：`datasets/formal_skills/organizing_toy/training_manifest.json`；
- Skill IR：`data/skill_coverage/generated/organizing_toy_skill_ir.json`；
- 任务二总契约：15 个任务、123 个唯一物体、138 条任务—物体关系；
- 首轮稳定门槛：三个未见 seed 中至少两个通过正式证据验收。

## PyCharm 运行配置

项目已提供四个共享配置：

1. `Formal 01 - Resume OpenVLA Training`
2. `Formal 02 - OpenVLA Skill Server`
3. `Formal 03 - RoboCasa 300-step Seed 101`
4. `Formal 04 - Auto Posttrain 3-seed Eval`

训练和推理服务都使用：
`C:\Users\sjtu101\miniconda3\envs\openvla\python.exe`。

RoboCasa 闭环使用：
`C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe`。

RTX 3090 上不要同时手工启动训练和推理服务。训练结束后再启动服务；自动流水线会按这个顺序处理。

## 正式训练

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -u -m tools.finetune_formal_skill_openvla_resumable `
  --epochs 2 --batch-size 1 --learning-rate 0.0001 `
  --checkpoint-every 500 --log-every 1 `
  --output C:\OpenVLA-Simulator\models\openvla-organizing-toy-lora
```

训练计划是 8084 次更新。每 500 步保存一次完整的 LoRA 和优化器状态；重复运行同一命令会自动选择与当前 manifest 哈希一致的最新完整检查点。

实时查看：

```powershell
Get-Content C:\OpenVLA-Simulator\outputs\organizing_toy_formal_train_resumable.log -Wait
```

训练错误：

```powershell
Get-Content C:\OpenVLA-Simulator\outputs\organizing_toy_formal_train_resumable.err.log -Wait
```

`CHECKPOINT_SAVED step=500 ...` 表示首个可恢复检查点已经落盘。训练真正结束的标志是：

`models/openvla-organizing-toy-lora/training_report.json` 中
`completed_updates == planned_updates == 8084`。

## 训练后的自动闭环

自动流水线会等待上面的原子完成报告，然后依次：

1. 启动 `tools.openvla_formal_skill_tcp_server`；
2. 检查适配器和动作统计；
3. 对 `101/102/103` 分别执行 300 次策略决策；
4. 保存逐步 raw action、安全约束后的 action、阶段、抓取和柜内谓词；
5. 运行独立证据验收器；
6. 汇总 2/3 泛化门槛。

实时查看总流水线：

```powershell
Get-Content C:\OpenVLA-Simulator\outputs\formal_skill_posttrain_pipeline.log -Wait
```

进入某个 seed 后查看该回合：

```powershell
Get-Content C:\OpenVLA-Simulator\outputs\formal_skill_posttrain_pipeline\eval_seed_101.log -Wait
```

最终汇总：

`outputs/formal_skill_posttrain_pipeline/report.json`

单 seed 报告和视频：

- `outputs/formal_skill_eval/seed_101/report.json`
- `outputs/formal_skill_eval/seed_101/rollout.mp4`

## 阶段条件和安全边界

OpenVLA 仍然根据机器人相机图像输出 7D 动作。高层 Skill IR 向模型提供当前阶段：
`locate -> grasp -> move -> place`。阶段切换只接受模拟器的实际接近、夹持、柜内和成功谓词，不把模型自报状态当真。

执行层固定未训练的旋转通道为零，并按阶段约束平移上限和夹爪开合。报告同时保存模型原始动作与最终执行动作，便于区分模型贡献和安全监督器干预。

## 已处理的 Windows 编排问题

Windows 下不能用 Unix 习惯的 `os.kill(pid, 0)` 探测训练进程；它可能终止目标进程。当前自动流水线不再接触训练 PID，只等待训练器原子写出的完成报告。训练器同时增加了检查点与自动续训，避免外部中断造成整轮重训。
