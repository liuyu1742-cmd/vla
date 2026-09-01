# 技能覆盖证据账本与确定性合并设计

**状态：** 已批准执行  
**日期：** 2026-07-24  
**范围：** 将已经通过闭环验收的正式任务—物体关系安全地计入 15 类任务 / 123 个候选物体覆盖注册表。

## 1. 问题

`organizing::toy` 已在 held-out seeds 101、102、103 上通过 300 决策步协议，
但 `data/skill_coverage/generated/coverage_summary.json` 仍显示
`validated_object_count=0`。当前不能手工把矩阵改为 `passed`，因为：

1. 重新构建权威注册表会覆盖手工状态；
2. 训练损失、目录存在和基础设施 smoke 不能替代闭环证据；
3. `organizing::toy` 的 Task-2 task id 必须通过 applicability policy 映射到
   service task `organization_storage`，不能按字符串直接查矩阵；
4. hybrid 闭环含状态感知定位恢复、抓取时序校准和 final-contact servo，
   必须与 pure autonomous VLA 证据分开表述。

## 2. 方案比较

### 方案 A：直接修改生成矩阵

实现最快，但重建注册表后证据丢失，且无法审计是谁、依据什么报告将状态改为
`passed`。不采用。

### 方案 B：持久化证据账本并确定性叠加

把通过验证的证据写入非生成文件 `data/skill_coverage/evidence_ledger.json`，
再由统一函数叠加到基线矩阵并重新计算汇总。构建注册表时也应用同一账本。
可重放、可审计、可复用，采用此方案。

### 方案 C：校验器运行时动态扫描 outputs

不修改矩阵，但 outputs 中可能同时存在 smoke、失败、旧 adapter 和不同证据等级，
自动扫描容易误计覆盖；也无法形成稳定快照。不采用。

## 3. 数据契约

### 3.1 证据账本

账本使用 `skill_coverage_evidence_ledger_v1`：

```json
{
  "schema_version": "skill_coverage_evidence_ledger_v1",
  "records": [
    {
      "relation_key": "organizing::toy",
      "dataset_task_id": "organizing",
      "service_task_id": "organization_storage",
      "object_id": "toy",
      "validation_state": "passed",
      "evidence_level": "L3",
      "evaluation_mode": "hybrid_closed_loop",
      "pure_autonomous_vla": false,
      "bundle_path": "outputs/formal_skill_dagger_r3_hybrid_v3_eval/acceptance_summary.json",
      "bundle_sha256": "<64 lowercase hex>",
      "heldout_seeds": [101, 102, 103],
      "successful_seeds": 3,
      "evaluated_seeds": 3
    }
  ]
}
```

记录按 `relation_key` 排序；同一关系再次导入时以最新的已验证 bundle 替换，
不产生重复记录。账本不修改 Task-2 的只读源文件。

### 3.2 可接受的 hybrid 验收包

第一版只接收 `formal_skill_hybrid_acceptance_summary_v1`，且必须满足：

- `relation_key` 存在于 `task2_skill_contract.json`；
- split 为 `held_out`，协议上限为每 seed 300 决策步；
- aggregate 的 `successful_seeds == evaluated_seeds >= 3`；
- `success_rate == 1.0`；
- `all_native_success_predicates_true == true`；
- `counts_toward_task2_coverage == true`；
- `counts_toward_autonomous_vla_acceptance == false`；
- `evaluation_mode.kind == hybrid_closed_loop`；
- `pure_autonomous_vla == false`；
- 每个 result 对应的 report 文件存在；
- 每份 report 的 relation、seed、split、adapter/manifest hash 与 bundle 一致；
- 每份 report 均为成功，且 simulator、placed-in-storage、released 三个谓词为 true；
- 每份 report 明确 `counts_toward_task2_coverage=true` 和
  `counts_toward_autonomous_vla_acceptance=false`。

任何一项不满足都拒绝导入，不部分更新账本或覆盖矩阵。

## 4. relation 到覆盖矩阵的映射

`relation_key = dataset_task_id::object_id`。合并器从
`applicability_policy.json.primary_by_dataset_task` 得到 `service_task_id`。

对于 `organizing::toy`：

```text
dataset_task_id = organizing
primary_by_dataset_task[organizing] = organization_storage
matrix key = organization_storage::toy
```

目标矩阵行必须存在，且 applicability 必须是 `direct`、
`device_control` 或 `composite_resource`；`not_applicable` 和
`needs_review` 永远不能被证据提升为 `passed`。

## 5. 组件边界

### `tools/skill_coverage/evidence_merge.py`

- 校验 hybrid 验收包及其引用报告；
- 生成规范化 ledger record；
- 合并或替换账本记录；
- 将账本确定性叠加到矩阵；
- 重新计算 coverage summary。

该模块不执行训练、不启动 RoboCasa、不修改外部 Task-2 文件。

### `tools/merge_skill_coverage_evidence.py`

- 提供命令行入口；
- 先在内存中完成全部校验；
- 将账本、矩阵和汇总写到同目录临时文件；
- 校验预期结构后用 `Path.replace` 原子替换单个目标文件；
- 输出 relation、service-task、validated objects/tasks 和剩余缺口。

### `tools/build_authoritative_skill_coverage.py`

生成基线矩阵后，如果账本存在，调用同一叠加函数恢复已验证证据；因此重建不会
静默把覆盖状态清零。

### `tools/validate_skill_coverage.py`

继续作为独立最终校验器。所有 `passed` 行必须引用存在的证据文件，存储汇总必须
与矩阵重新计算值一致。

## 6. 错误处理

- bundle、report、contract、policy 或矩阵字段不一致：抛出 `ValueError`；
- 证据文件不存在或 hash 不匹配：抛出 `FileNotFoundError` 或 `ValueError`；
- relation 不在 138 条冻结契约中：拒绝；
- 目标矩阵行为不可适用状态：拒绝；
- 导入失败时不得修改 ledger、matrix、summary；
- 成功导入后必须运行独立 coverage validator。

## 7. 验收标准

本阶段只有满足以下全部条件才完成：

1. 证据合并单元测试先红后绿；
2. smoke、失败报告、缺失 report、hash 不一致、未知 relation 均被拒绝；
3. `organizing::toy` 精确映射到 `organization_storage::toy`；
4. 账本记录为 L3 `hybrid_closed_loop`，`pure_autonomous_vla=false`；
5. `validated_object_count` 从 0 变为 1；
6. `validated_service_task_count` 从 0 变为 1；
7. `remaining_validation_gap` 从 120 变为 119；
8. 重新构建注册表后再次校验，以上数字保持不变；
9. 独立校验器输出 `STRUCTURAL_VALIDATION: PASS`。

## 8. 后续阶段接口

后续每个正式 relation 复用同一导入接口。L2 人类视频证据和 L4 真机证据使用独立
schema 与 evidence level，不覆盖 L3 结果；同一 relation 可保留多个证据等级。
第一版只实现已存在且可验证的 L3 hybrid pass 路径，避免提前构建没有真实证据的
通用框架。
