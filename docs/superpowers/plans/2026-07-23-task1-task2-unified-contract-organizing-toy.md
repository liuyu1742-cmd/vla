# 任务一/任务二统一契约与 organizing::toy 闭环实施计划

> **执行要求：** 实施时使用测试驱动开发；每个任务先写失败测试，再写最小实现；宣布
> 完成前使用验证优先流程重新运行本计划列出的命令并检查实际产物。

**目标：** 在不修改 `C:\RobotProject` 的前提下，将任务二的 15 个任务、123 个唯一
物体和 138 条标准动作关系冻结为任务一的只读技能契约，并完成第一条正式关系
`organizing::toy` 的 OpenVLA + RoboCasa 300 步闭环与可审计报告。

**架构：** 新的任务二契约导入器读取 Excel 和冻结动作基准，生成带双源哈希的 138 条
关系快照。`household_skill_ir_v2` 从关系快照构造执行请求。RoboCasa 通过本地注册的
彩色 toy block 和现有 `PickPlaceCounterToCabinet` 形成 `storage` 场景；通用抓放专家
采集成功示教，通用 LoRA 训练器训练适配器，Skill IR 运行器通过现有 TCP IPC 请求
OpenVLA 动作、执行安全监督并输出 300 步上限报告。

**技术栈：** Python 3.10/3.11、标准库 ZIP/XML/JSON、PyTorch、Transformers、PEFT、
OpenVLA、Gymnasium、RoboCasa、robosuite、MuJoCo、unittest、PyCharm 共享配置。

**边界：** 本计划只完成契约和第一条正式关系，不宣称 138 条关系已物理执行；旧杯子
入柜只作为基础冒烟测试。当前工作区不是 Git 仓库，因此不包含提交命令；每个检查点
用测试输出、JSON 哈希和报告文件固定。

---

## Task 1：建立任务二双源契约读取器

**文件：**

- 新建：`tools/skill_coverage/task2_contract.py`
- 新建：`tests/test_task2_skill_contract.py`
- 复用：`tools/skill_coverage/xlsx_reader.py`
- 复用：`tools/skill_coverage/registry_builder.py`

### Step 1：先写失败测试

测试必须构造小型 Excel 行和小型 benchmark JSON，并覆盖：

```python
def test_contract_requires_exact_relation_identity():
    excel = [{"task_id": "organizing", "object": "toy", ...}]
    benchmark = [{
        "relation_key": "organizing::toy",
        "acceptance_task": "organizing",
        "object": "toy",
        "instruction": "整理任务：整理并归位玩具（toy）。",
        "target": "none",
        "actions": ["locate(toy)", "grasp(toy)",
                    "move(storage)", "place(toy)"],
    }]
    contract = build_task2_contract(excel, benchmark)
    assert contract[0]["relation_key"] == "organizing::toy"

def test_contract_rejects_benchmark_relation_missing_from_excel(): ...
def test_contract_rejects_excel_relation_missing_from_benchmark(): ...
def test_contract_rejects_malformed_relation_key(): ...
def test_contract_rejects_empty_or_unordered_actions(): ...
def test_same_object_in_two_tasks_is_preserved(): ...
```

### Step 2：运行并确认失败

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest tests.test_task2_skill_contract -v
```

预期：因 `tools.skill_coverage.task2_contract` 不存在而失败。

### Step 3：实现最小契约模块

实现以下纯函数：

```python
def relation_key(task_id: str, object_id: str) -> str: ...
def parse_action_call(action: str) -> tuple[str, tuple[str, ...]]: ...
def validate_benchmark_record(record: dict[str, object]) -> dict[str, object]: ...
def build_task2_contract(
    excel_rows: list[dict[str, str]],
    benchmark_records: list[dict[str, object]],
) -> list[dict[str, object]]: ...
```

输出每条记录至少包含 `relation_key`、任务/物体中英文名、源 Excel 行号、instruction、
target、canonical_actions、legacy_task_id 和媒体统计。保持 Excel 顺序，不按对象去重
关系。

### Step 4：运行测试

运行 Step 2 命令，预期全部通过。

---

## Task 2：生成并锁定 15/123/138 契约快照

**文件：**

- 新建：`data/skill_coverage/task2_contract_lock.json`
- 新建：`tools/import_task2_skill_contract.py`
- 新建：`tools/validate_task2_skill_contract.py`
- 生成：`data/skill_coverage/generated/task2_skill_contract.json`
- 生成：`data/skill_coverage/generated/task2_contract_manifest.json`
- 修改：`tests/test_authoritative_skill_coverage_snapshot.py`
- 新建：`tests/test_validate_task2_skill_contract.py`

### Step 1：写锁文件和失败测试

锁文件固定：

```json
{
  "schema_version": "task2_contract_lock_v1",
  "xlsx_path": "C:\\RobotProject\\RobotProject\\docs\\家庭服务任务物体数据表.xlsx",
  "xlsx_sheet": "任务物体数据表",
  "xlsx_sha256": "23e39dd2803f3d90725169a6095f354b227feb4c1f54f92310679e84732e9ae0",
  "benchmark_path": "C:\\RobotProject\\RobotProject\\datasets\\acceptance_instruction_action_benchmark_15tasks.json",
  "benchmark_sha256": "97198b51f7b0ebf3e160bd6397fbaed70ae73cd966b431180a8431ad6b75e212",
  "expected_tasks": 15,
  "expected_unique_objects": 123,
  "expected_relations": 138
}
```

失败测试覆盖哈希变化、15/123/138 数量变化、重复 `relation_key`、源文件在读取期间变化、
14 个跨任务物体未保留和 `remote_control` 三关系被丢失。

### Step 2：实现导入器

导入器参数：

```text
--lock data/skill_coverage/task2_contract_lock.json
--output-dir data/skill_coverage/generated
```

先计算两个源文件哈希，再读取 Excel/benchmark，再次计算哈希；前后不一致立即失败。
使用临时文件和原子替换写入两个 JSON。manifest 记录源哈希、计数、生成器版本、生成时间、
48 个动作原语和跨任务物体列表。

### Step 3：实现独立验证器

验证器不能依赖旧 `service_tasks.json`、`applicability_policy.json` 或 1845 项 matrix。
它重新计算关系集合、动作原语、任务数、物体数和双源哈希，输出：

```text
CONTRACT_VALIDATION: PASS
TASKS: 15/15
OBJECTS: 123/123
RELATIONS: 138/138
SOURCE_HASHES: PASS
```

### Step 4：生成并验证真实快照

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.import_task2_skill_contract
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.validate_task2_skill_contract
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest tests.test_task2_skill_contract tests.test_validate_task2_skill_contract tests.test_authoritative_skill_coverage_snapshot -v
```

预期：三个命令成功；源 Excel 和 benchmark 哈希保持锁值。

---

## Task 3：实现 household_skill_ir_v2

**文件：**

- 新建：`tools/skill_transfer/__init__.py`
- 新建：`tools/skill_transfer/contracts.py`
- 新建：`tools/skill_transfer/skill_ir.py`
- 新建：`tests/test_household_skill_ir_v2.py`
- 生成夹具：`tests/fixtures/organizing_toy_skill_ir.json`

### Step 1：写失败测试

覆盖：

- `relation_key` 必须存在于快照；
- `canonical_actions` 必须与快照逐项相等；
- task/object 与 key 不一致时拒绝；
- 缺失证据使用 `null`，不能用全零关键点冒充；
- `execution_evidence` 不能在未执行时标为成功；
- 能从 `organizing::toy` 构造确定性 Skill IR。

### Step 2：实现契约

提供：

```python
def load_contract(path: Path) -> dict[str, dict[str, object]]: ...
def build_skill_ir(relation_key: str, contract: Mapping[str, dict]) -> dict: ...
def validate_skill_ir(ir: Mapping[str, object], contract: Mapping[str, dict]) -> dict: ...
def write_skill_ir(path: Path, ir: Mapping[str, object]) -> None: ...
```

执行证据中预留 `raw_openvla_actions`、`executed_actions`、`safety_interventions`、
`recovery_actions`、`canonical_action_progress` 和 `success_predicates`。

### Step 3：运行测试并生成正式输入

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest tests.test_household_skill_ir_v2 -v
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.skill_transfer.skill_ir --relation-key organizing::toy --output data\skill_coverage\generated\organizing_toy_skill_ir.json
```

预期：测试通过，JSON 的动作严格为 `locate/grasp/move(storage)/place`。

---

## Task 4：隔离旧水杯冒烟证据和正式覆盖

**文件：**

- 修改：`tools/run_openvla_robocasa_water_cup.py`
- 修改：`tools/openvla_gym_water_cup_guarded_rollout.py`
- 新建：`tools/skill_transfer/evidence.py`
- 修改：`tests/test_run_openvla_robocasa_water_cup.py`
- 新建：`tests/test_formal_skill_evidence.py`
- 修改：`docs/SKILL_COVERAGE_REGISTRY.md`

### Step 1：先写失败测试

断言所有新水杯报告包含：

```json
{
  "evidence_scope": "infrastructure_smoke",
  "relation_key": null,
  "counts_toward_task2_coverage": false
}
```

正式证据验证器必须拒绝该报告，即使 `success=true`。只有存在于 138 条快照、Skill IR
动作一致且关系级成功谓词通过的报告才能计入 L3。

### Step 2：实现并验证

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest tests.test_run_openvla_robocasa_water_cup tests.test_formal_skill_evidence -v
```

不得删除旧模型、旧数据或旧输出；只改变新报告语义和正式聚合规则。

---

## Task 5：注册本地 toy block 并建立正式 RoboCasa 场景

**文件：**

- 新建：`assets/robocasa/task2/toy/toy_block_0/model.xml`
- 新建：`tools/skill_transfer/robocasa_assets.py`
- 新建：`tools/skill_transfer/robocasa_envs.py`
- 新建：`tools/check_organizing_toy_env.py`
- 新建：`tests/test_task2_robocasa_assets.py`
- 新建：`tests/test_organizing_toy_env_contract.py`

### Step 1：写轻量失败测试

测试 MJCF 含自由关节、碰撞 geom、可见 geom、`reg_bbox`、质量/摩擦和稳定尺寸；资产
注册函数重复调用不产生重复或覆盖 RoboCasa 原类别；环境映射只接受
`organizing::toy`。

### Step 2：实现资产注册

不修改 `third_party`。在 RoboCasa 导入后，通过 `ObjCat` 向运行期
`OBJ_CATEGORIES["toy"]` 和 `OBJ_GROUPS["toy"]` 注册本地模型目录，并将注册表类型映射
到 RoboCasa 已支持的对象 registry 参数。函数必须幂等，并验证 MJCF 路径位于项目
`assets/robocasa/task2` 下。

### Step 3：实现环境工厂

```python
def create_formal_env(
    relation_key: str,
    *,
    seed: int,
    camera_name: str = "robot0_agentview_left",
): ...
```

`organizing::toy` 映射到 `PickPlaceCounterToCabinet`、`obj_groups="toy"`，cabinet 作为
`storage`。工厂返回环境和映射元数据，元数据包含 relation、目标对象、目标区域、
RoboCasa task 和成功谓词版本。

### Step 4：运行轻量和真实环境检查

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest tests.test_task2_robocasa_assets tests.test_organizing_toy_env_contract -v
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m tools.check_organizing_toy_env --seed 0 --output outputs\organizing_toy_env_check
```

真实检查必须保存首帧、对象类别、对象尺寸、目标 cabinet、相机 shape 和 reset 后状态；
环境不能只因为能 reset 就声明任务成功。

---

## Task 6：通用化抓放专家并采集 toy 成功示教

**文件：**

- 新建：`tools/pick_place_oracle.py`
- 修改：`tools/water_cup_dagger_oracle.py`（兼容重导出旧类名）
- 新建：`tools/collect_formal_skill_expert.py`
- 新建：`tools/prepare_formal_skill_manifest.py`
- 新建：`tests/test_pick_place_oracle.py`
- 新建：`tests/test_collect_formal_skill_expert.py`
- 新建：`tests/test_prepare_formal_skill_manifest.py`
- 运行生成：`datasets/formal_skills/organizing_toy/`

### Step 1：重构前写兼容测试

将现有 `WaterCupDaggerOracle` 的纯状态机测试复制为通用 `PickPlaceOracle` 测试，并断言
旧导入路径行为不变。通用状态机使用 `object_position`、destination front/center/retreat
和成功状态，不出现 `water_cup` 硬编码。

### Step 2：实现专家采集器

采集器参数：

```text
--skill-ir data/skill_coverage/generated/organizing_toy_skill_ir.json
--seed N
--frame-stride 1
--max-steps 900
--output-root datasets/formal_skills/organizing_toy
```

每个 episode 保存 RGB、7D 专家动作、canonical action/phase、夹爪状态、对象/EEF 位置、
成功谓词、relation_key、Skill IR 哈希和环境映射。只有 `env._check_success()` 与关系级
谓词都通过时 `success=true`。

### Step 3：采集并建立无泄漏划分

先跑 seed 0 冒烟；成功后采集至少 12 个成功训练种子和 3 个从未进入训练的 held-out
种子。manifest 以 seed 去重并拒绝训练/验证重叠。

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m tools.collect_formal_skill_expert --skill-ir data\skill_coverage\generated\organizing_toy_skill_ir.json --seed 0 --frame-stride 1
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m tools.prepare_formal_skill_manifest --relation-key organizing::toy --heldout-seeds 101 102 103
```

批量采集命令在 seed 0 单回合通过后再运行；失败 episode 保留诊断但不进入训练。

### Step 4：验证

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest tests.test_pick_place_oracle tests.test_collect_formal_skill_expert tests.test_prepare_formal_skill_manifest -v
```

---

## Task 7：建立通用 Skill IR LoRA 训练器和服务

**文件：**

- 新建：`tools/formal_skill_dataset.py`
- 新建：`tools/finetune_formal_skill_openvla.py`
- 新建：`tools/openvla_tcp_server_skill_adapter.py`
- 新建：`tests/test_formal_skill_dataset.py`
- 新建：`tests/test_finetune_formal_skill_openvla.py`
- 新建：`tests/test_openvla_tcp_server_skill_adapter.py`
- 运行生成：`models/openvla-organizing-toy-lora/`
- 运行日志：`outputs/organizing_toy_training.log`

### Step 1：写失败测试

覆盖：manifest 只加载成功训练 episode、held-out 不进入训练、每个样本携带 Skill IR
instruction/phase、7D 动作长度、视觉 token 标签对齐、训练更新数正确、action stats 可由
服务加载、服务不再硬编码 `water_cup`。

### Step 2：实现通用数据集与训练器

训练器参数：

```text
--manifest datasets/formal_skills/organizing_toy/training_manifest.json
--model-dir models/openvla-7b
--epochs 12
--batch-size 1
--learning-rate 1e-4
--output models/openvla-organizing-toy-lora
--log-every 1
```

沿用现有 visual-token-aligned weighted causal cross entropy，但 instruction 从 Skill IR
读取，phase 可作为提示，不从文件名推断。训练报告保存数据/Skill IR 哈希、train seeds、
样本数、更新数、loss 曲线、CUDA/模型版本和输出 adapter 哈希。

### Step 3：先 dry-run 再正式训练

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.finetune_formal_skill_openvla --manifest datasets\formal_skills\organizing_toy\training_manifest.json --epochs 12 --dry-run
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.finetune_formal_skill_openvla --manifest datasets\formal_skills\organizing_toy\training_manifest.json --epochs 12 --output models\openvla-organizing-toy-lora
```

实时查看：

```powershell
Get-Content C:\OpenVLA-Simulator\outputs\organizing_toy_training.log -Wait
```

训练进程必须由脚本本身同时写日志和控制台，不能依赖空的外部重定向文件。

### Step 4：验证服务启动

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.openvla_tcp_server_skill_adapter --adapter-dir models\openvla-organizing-toy-lora --port 8781
```

预期出现 `OPENVLA_SKILL_SERVER_READY 127.0.0.1:8781`；常驻等待是正常状态。

---

## Task 8：实现 Skill IR 驱动的正式闭环运行器

**文件：**

- 新建：`tools/run_openvla_skill_ir_rollout.py`
- 新建：`tools/skill_transfer/formal_rollout.py`
- 复用：`tools/openvla_tcp_client.py`
- 复用：`tools/openvla_simulator_adapter.py`
- 复用：`tools/water_cup_guarded_control.py`（随后泛化命名）
- 新建：`tests/test_formal_skill_rollout.py`
- 新建：`tests/test_run_openvla_skill_ir_rollout.py`

### Step 1：写 mock 失败测试

使用假环境和假预测器验证：

- 只接受快照中的关系；
- 每步请求携带当前 canonical action/phase；
- 原始 OpenVLA 动作、监督后动作和最终 RoboCasa action 分开；
- 抓取失败能恢复；
- 300 步超时为失败；
- env success 但未完成四个 canonical action 时不能计正式成功；
- 水杯冒烟报告不能传入正式聚合器。

### Step 2：实现运行器

CLI：

```text
--skill-ir <path>
--host 127.0.0.1
--port 8781
--seed 101
--steps 300
--max-translation-error 0.15
--output-dir outputs/organizing_toy_eval/seed_101
```

每 10 步或 phase 变化时 flush 控制台与 `rollout_progress.jsonl`。最终报告保存：relation、
Skill IR/adapter/source hashes、seed、executed_steps、canonical_action_progress、
ever_grasped、object_lift、inside_storage、gripper_released、stable_frames、success、
openvla_direct_fraction、干预/恢复计数和逐步记录。

### Step 3：轻量测试

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest tests.test_formal_skill_rollout tests.test_run_openvla_skill_ir_rollout -v
```

---

## Task 9：执行 300 步训练种子和 held-out 评测

**文件：**

- 新建：`tools/evaluate_formal_skill.py`
- 新建：`tests/test_evaluate_formal_skill.py`
- 运行生成：`outputs/organizing_toy_eval/summary.json`
- 运行生成：`outputs/organizing_toy_eval/summary.md`

### Step 1：实现严格汇总器

汇总器拒绝缺失 report、重复 seed、非 300 上限、关系/模型/Skill IR 哈希不一致和没有
held-out seed 的输入。分别报告 train-seed 与 held-out 成功率；不因训练 loss 下降判定
成功。

### Step 2：运行真实评测

OpenVLA 终端保持 8781 服务运行；RoboCasa 终端逐个执行：

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m tools.run_openvla_skill_ir_rollout --skill-ir data\skill_coverage\generated\organizing_toy_skill_ir.json --host 127.0.0.1 --port 8781 --seed 0 --steps 300 --output-dir outputs\organizing_toy_eval\seed_000
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m tools.run_openvla_skill_ir_rollout --skill-ir data\skill_coverage\generated\organizing_toy_skill_ir.json --host 127.0.0.1 --port 8781 --seed 101 --steps 300 --output-dir outputs\organizing_toy_eval\seed_101
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m tools.run_openvla_skill_ir_rollout --skill-ir data\skill_coverage\generated\organizing_toy_skill_ir.json --host 127.0.0.1 --port 8781 --seed 102 --steps 300 --output-dir outputs\organizing_toy_eval\seed_102
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m tools.run_openvla_skill_ir_rollout --skill-ir data\skill_coverage\generated\organizing_toy_skill_ir.json --host 127.0.0.1 --port 8781 --seed 103 --steps 300 --output-dir outputs\organizing_toy_eval\seed_103
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.evaluate_formal_skill --relation-key organizing::toy --reports outputs\organizing_toy_eval --heldout-seeds 101 102 103
```

实时查看：

```powershell
Get-Content C:\OpenVLA-Simulator\outputs\organizing_toy_eval\seed_101\rollout_progress.jsonl -Wait
```

首个里程碑通过条件：seed 0 成功，三个 held-out seed 至少 2 个成功，失败 seed 有明确
阶段诊断；每个成功回合满足六个关系级谓词，并保存非零 OpenVLA 直接动作比例。

---

## Task 10：PyCharm 配置、文档和最终验证

**文件：**

- 新建：`.run/Task2 Contract Import.run.xml`
- 新建：`.run/OpenVLA Organizing Toy Server.run.xml`
- 新建：`.run/RoboCasa Organizing Toy 300-step Eval.run.xml`
- 修改：`docs/OPENVLA_PYCHARM_SETUP.md`
- 修改：`docs/SKILL_COVERAGE_REGISTRY.md`
- 新建：`docs/ORGANIZING_TOY_FORMAL_CLOSED_LOOP.md`

### Step 1：共享 PyCharm 配置

三个配置分别固定解释器：

- 契约导入：`openvla\python.exe`；
- 模型服务：`openvla\python.exe`，端口 8781；
- 仿真评测：`robocasa\python.exe`，300 步。

工作目录均为 `$PROJECT_DIR$`，启用终端模拟。模型服务持续显示 READY 并等待连接是正常
行为；不是卡死。

### Step 2：完整自动验证

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m compileall -q tools tests
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest tests.test_task2_skill_contract tests.test_validate_task2_skill_contract tests.test_household_skill_ir_v2 tests.test_formal_skill_evidence tests.test_task2_robocasa_assets tests.test_organizing_toy_env_contract tests.test_pick_place_oracle tests.test_collect_formal_skill_expert tests.test_prepare_formal_skill_manifest tests.test_formal_skill_dataset tests.test_finetune_formal_skill_openvla tests.test_openvla_tcp_server_skill_adapter tests.test_formal_skill_rollout tests.test_run_openvla_skill_ir_rollout tests.test_evaluate_formal_skill -v
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.validate_task2_skill_contract
```

### Step 3：最终证据检查

确认以下文件实际存在且内容一致：

```text
data/skill_coverage/generated/task2_skill_contract.json
data/skill_coverage/generated/task2_contract_manifest.json
data/skill_coverage/generated/organizing_toy_skill_ir.json
models/openvla-organizing-toy-lora/training_report.json
outputs/organizing_toy_eval/summary.json
outputs/organizing_toy_eval/summary.md
```

最终报告必须明确：

- 任务二源文件未修改且哈希匹配；
- 15/123/138 契约通过；
- 杯子入柜不计正式覆盖；
- `organizing::toy` 是第一条正式 L3 关系；
- OpenVLA 直接动作比例与监督器干预比例；
- 本里程碑不等于 138/138 仿真完成，也不等于真机 L4 完成。

完成本计划后，再单独编写并执行 `remote_control`/`fan` 同物体多任务计划。
