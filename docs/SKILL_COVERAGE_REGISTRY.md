# 15 类任务 / 123 个候选物体覆盖注册表

## 1. 作用与当前结论

该注册表把两个只读权威来源转成项目可执行、可检查的覆盖基线：

- Word 第五章规定最终 15 类家庭服务任务；
- Excel 规定 15 个数据集任务组、138 条任务—物体源关系、123 个去重候选物体；
- 15 类任务与 123 个物体不是全笛卡尔积意义上的有效组合；
- 系统仍为全部 `15 × 123 = 1845` 个组合生成显式记录，用 `not_applicable` 表示不应执行的组合；
- “语义上已映射”不等于“机器人闭环已通过”。只有适用组合附带真实证据且状态为 `passed`，物体才计入不少于 120 种物体的验收覆盖。

当前生成结果是：15 类服务任务、123 个候选物体、138 条源关系、1845 个显式矩阵项、123 个语义映射物体、0 个闭环验证物体。因此注册表结构已经完成，但 120 物体验收仍是 `NOT_READY`，不能把候选数写成成功数。

## 2. 权威来源与只读约束

源文件：

- `C:\Users\sjtu101\Desktop\2. 基于视觉的人类动作捕获、解析和技能学习迁移技术研究.docx`
- `C:\RobotProject\RobotProject\docs\家庭服务任务物体数据表.xlsx`

构建器只读取 Excel，不修改 Word 或 Excel。构建前后会计算 Excel SHA-256；读文件期间哈希变化会使构建失败。当前锁定的 Excel SHA-256 为：

```text
23e39dd2803f3d90725169a6095f354b227feb4c1f54f92310679e84732e9ae0
```

项目内的任务目录、动作原语和适用性策略位于：

- `data/skill_coverage/service_tasks.json`
- `data/skill_coverage/action_primitives.json`
- `data/skill_coverage/applicability_policy.json`

## 3. PyCharm 运行配置

两个配置都使用：

```text
Python interpreter: C:\Users\sjtu101\miniconda3\envs\openvla\python.exe
Working directory:  C:\OpenVLA-Simulator
```

### 配置 1：构建权威覆盖注册表

在 **Run | Edit Configurations | Add New Configuration | Python** 中填写：

```text
Name:       01 Build Authoritative Skill Coverage
Run:        Module name
Module:     tools.build_authoritative_skill_coverage
Parameters: --xlsx "C:\RobotProject\RobotProject\docs\家庭服务任务物体数据表.xlsx"
```

等价 PowerShell 命令：

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.build_authoritative_skill_coverage --xlsx "C:\RobotProject\RobotProject\docs\家庭服务任务物体数据表.xlsx"
```

配置 1 成功后会原子更新 `data/skill_coverage/generated` 下的五个 JSON 文件。

### 配置 2：独立校验注册表

```text
Name:       02 Validate Authoritative Skill Coverage
Run:        Module name
Module:     tools.validate_skill_coverage
Parameters: --registry-dir "C:\OpenVLA-Simulator\data\skill_coverage\generated"
```

等价 PowerShell 命令：

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.validate_skill_coverage --registry-dir "C:\OpenVLA-Simulator\data\skill_coverage\generated"
```

正确的当前输出包含：

```text
STRUCTURAL_VALIDATION: PASS
PROJECT_ACCEPTANCE: NOT_READY validated_objects=0 target=120 validated_tasks=0/15
```

`STRUCTURAL_VALIDATION: PASS` 表示数据结构、数量、交叉引用、适用性规则、摘要和源文件哈希均一致；它不代表 15 类任务和 120 个物体已经通过机器人闭环。

## 4. 生成文件

| 文件 | 含义 |
|---|---|
| `dataset_tasks.json` | Excel 中 15 个数据集任务组 |
| `object_candidates.json` | 123 个去重候选物体及来源行 |
| `source_relations.json` | Excel 中完整保留的 138 条任务—物体关系 |
| `task_object_matrix.json` | 15×123 个显式适用性与验证状态 |
| `coverage_summary.json` | 候选、语义映射、验证和缺口的汇总 |

矩阵中的适用性状态：

- `direct`：该物体可直接用于该服务任务；
- `device_control`：通过设备状态或接口控制完成；
- `composite_resource`：物体是复合任务需要的资源；
- `not_applicable`：该任务—物体组合不成立；
- `needs_review`：需要人工复核，不能计入验收。

验证状态：`untested`、`passed`、`failed`、`blocked`。`not_applicable` 和 `needs_review` 不能标记为 `passed`。

## 5. 后续闭环证据工作流

后续每个适用任务—物体组合都按同一顺序执行：

1. 从人类视频解析任务、物体、手部/人体运动与阶段边界，形成 Skill IR；
2. 由机器人专家、轨迹优化或遥操作把 Skill IR 转成机器人可执行示教；
3. 用示教训练或适配 OpenVLA，再在 RoboCasa 中进行独立种子闭环评测；
4. 保存评测报告、轨迹、关键帧或视频作为项目内证据文件；
5. 只有实际成功的适用组合才写入 `passed` 并关联存在的证据路径；
6. 每次证据更新后运行配置 2；校验器会重新计算通过物体数和通过任务数。

项目最终达到覆盖要求的最低条件是：

- 至少 120 个不同候选物体各有至少一个适用组合通过；
- 15 类服务任务全部至少有一个适用组合通过；
- 所有 `passed` 组合都有可访问的证据文件；
- 独立校验输出 `PROJECT_ACCEPTANCE: READY`。

重新运行配置 1 会重建基线矩阵并把验证状态恢复为 `untested`。在下一阶段实现证据合并工具之前，不要手工把生成矩阵改成 `passed`，也不要用训练损失代替机器人闭环成功证据。
