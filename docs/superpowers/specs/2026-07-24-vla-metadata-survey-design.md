# VLA 家庭任务元数据普查设计

日期：2026-07-24

## 目标

从 VLA / 机器人操作数据集中真实存在的元数据出发，建立可复现的任务与物体目录，为后续选择不少于 15 类家庭任务和 120 种物体提供证据。此阶段只下载或解析元数据，不下载大规模视频、图像或完整轨迹。

## 原则

1. 不读取或使用 `C:\RobotProject`、任务二 Excel、参考文档第五章中的任务/物体名称作为分类依据。
2. 参考 DOCX 仅可提供技术候选（如层次化技能表示、BC/DAgger、领域随机化、Sim-to-Real 和增量学习）；涉及具体模型能力与实验数字时，必须回到官方论文或代码验证。
2. 每条任务和物体记录必须带来源、来源类型和证据级别。
3. 区分“官方宣传总数”“官方任务表”“本地源码注册项”“实际轨迹指令”。
4. 只有显式参与操作的对象才计入 120 物体候选；背景、场景和纯容器结构单独统计。
5. 15 类任务是对真实任务记录的上层归类，不是机械臂原子动作列表。
6. 本轮结果是候选覆盖基线，不等同于全部任务均已具备可直接训练的轨迹。

## 数据源优先级

1. 本地 BridgeData V2：TFDS 数据集结构、分割、语言字段；后续轻量抽样指令。
2. 本地 RoboCasa：任务类、任务说明、语言指令、对象配置、对象属性注册表。
3. 官方任务级元数据：LIBERO、DROID、RoboMIND、RH20T、BEHAVIOR。
4. 仅有官方汇总数据的数据源：只登记规模，不伪造任务明细。

## 统一记录

任务记录包含：

- `dataset_id`
- `task_id`
- `task_name`
- `task_family`
- `description`
- `instructions`
- `manipulated_objects`
- `target_objects`
- `fixtures`
- `source_path_or_url`
- `source_kind`
- `evidence_level`

对象记录包含：

- `canonical_name`
- `aliases`
- `datasets`
- `task_count`
- `roles`
- `properties`
- `evidence_level`

## 证据级别

- `sampled_episode`：从真实轨迹样本读取。
- `local_task_code`：从本地任务实现、语言或对象配置读取。
- `official_task_table`：官方逐任务元数据。
- `official_registry`：官方对象或任务注册表。
- `official_aggregate`：只有规模汇总，不用于生成具体任务或对象。
- `inferred_from_name`：只能作为低置信补充，单独标记。

## 输出

- `outputs/vla_metadata_survey/unified_task_catalog.jsonl`
- `outputs/vla_metadata_survey/object_catalog.json`
- `outputs/vla_metadata_survey/dataset_summary.json`
- `outputs/vla_metadata_survey/task_family_candidates.json`
- `outputs/vla_metadata_survey/object_candidates_top120.json`
- `outputs/vla_metadata_survey/task_catalog.csv`
- `outputs/vla_metadata_survey/object_catalog.csv`
- `docs/VLA_METADATA_SURVEY_2026-07-24.md`

## 验收

1. 所有任务记录通过统一字段校验。
2. 任务、对象结果可由一个命令重新生成。
3. 15 类任务每类均列出真实任务证据及数据集来源。
4. 120 物体候选不得靠补齐虚构；若不足 120，报告真实缺口。
5. 本地数据和官方汇总数字明确分开。
