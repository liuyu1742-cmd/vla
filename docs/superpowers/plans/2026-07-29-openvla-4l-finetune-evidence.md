# OpenVLA-4L Fine-Tuning Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one auditable OpenVLA-4L checkpoint, train the four Section 4.6.1 fine-tuning modes locally, evaluate every independent checkpoint in the water-cup simulator, and generate real training/GPU/simulation evidence images.

**Architecture:** A checkpoint builder loads the local OpenVLA-7B once on CPU, keeps four language layers, updates the nested Llama configuration, and saves a shared OpenVLA-4L base. A unified trainer applies one of four exact trainability policies, uses BF16, gradient checkpointing and Adafactor, and writes atomic JSONL logs plus GPU telemetry. A unified pure-policy evaluator reloads each checkpoint, executes direct 7-DoF actions in RoboCasa, and saves raw rollout frames and reports.

**Tech Stack:** Python 3.10, PyTorch 2.2.2+cu121, Transformers 4.40.1, PEFT, Pillow, NumPy, RoboCasa/Gymnasium, pytest.

## Global Constraints

- Hardware is one NVIDIA GeForce RTX 3090 24GB with 32GB host RAM.
- All four methods use the same OpenVLA-4L base and five demonstrations with seeds `0,1,2,3,5`.
- LoRA rank is exactly 32.
- Full and Frozen-Vision may use memory-saving execution but their trainable-parameter definitions must not be reduced.
- No expert recovery is allowed in closed-loop evaluation.
- Evidence images must be derived from actual logs, GPU telemetry, and saved simulation frames.
- Outputs stay under `outputs/experiment_4_6/openvla_4l/`; checkpoints stay under `models/experiment_4_6_openvla_4l/`.

---

### Task 1: Specify checkpoint reduction and trainability contracts

**Files:**
- Create: `tests/test_openvla_4l_experiment.py`
- Create: `tools/openvla_4l_experiment.py`

**Interfaces:**
- Produces: `selected_layer_indices(total_layers: int, keep_layers: int) -> tuple[int, ...]`
- Produces: `is_trainable(name: str, mode: str, last_layer_index: int) -> bool`
- Produces: `summarize_trainability(named_sizes, mode, last_layer_index) -> dict`

- [ ] **Step 1: Write the failing tests**

```python
def test_four_layer_selection_is_deterministic():
    assert selected_layer_indices(32, 4) == (0, 1, 2, 3)

def test_last_layer_only_selects_layer_three_and_lm_head():
    assert is_trainable("language_model.model.layers.3.self_attn.q_proj.weight", "last_layer_only", 3)
    assert is_trainable("language_model.lm_head.weight", "last_layer_only", 3)
    assert not is_trainable("language_model.model.layers.2.self_attn.q_proj.weight", "last_layer_only", 3)

def test_frozen_vision_excludes_only_vision_backbone():
    assert not is_trainable("vision_backbone.featurizer.blocks.0.weight", "frozen_vision", 3)
    assert is_trainable("projector.fc1.weight", "frozen_vision", 3)
```

- [ ] **Step 2: Run the tests and confirm they fail because the module is absent**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tests/test_openvla_4l_experiment.py -q`

Expected: collection error for missing `tools.openvla_4l_experiment`.

- [ ] **Step 3: Implement the minimal selection and classification functions**

```python
MODES = ("full", "lora_r32", "last_layer_only", "frozen_vision")

def selected_layer_indices(total_layers, keep_layers):
    if not 0 < keep_layers <= total_layers:
        raise ValueError("keep_layers must be within the source depth")
    return tuple(range(keep_layers))

def is_trainable(name, mode, last_layer_index):
    lowered = name.lower()
    if mode == "full":
        return True
    if mode == "frozen_vision":
        return not lowered.startswith("vision_backbone.")
    if mode == "last_layer_only":
        return (
            f"language_model.model.layers.{last_layer_index}." in lowered
            or lowered.startswith("language_model.lm_head.")
        )
    if mode == "lora_r32":
        return False
    raise ValueError(f"unknown mode: {mode}")
```

- [ ] **Step 4: Run the focused tests and confirm all pass**

Run: `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tests/test_openvla_4l_experiment.py -q`

Expected: all Task 1 tests pass.

### Task 2: Build and validate the shared OpenVLA-4L base

**Files:**
- Modify: `tools/openvla_4l_experiment.py`
- Modify: `tests/test_openvla_4l_experiment.py`

**Interfaces:**
- Produces: `reduce_language_layers(model, keep_layers: int) -> dict`
- CLI: `python -m tools.openvla_4l_experiment build-base --source ... --output ... --layers 4`

- [ ] **Step 1: Add a failing unit test using a fake nested language model**

```python
def test_reduce_language_layers_updates_both_configs():
    model = fake_model_with_layers(8)
    metadata = reduce_language_layers(model, 4)
    assert len(model.language_model.model.layers) == 4
    assert model.language_model.config.num_hidden_layers == 4
    assert model.config.text_config.num_hidden_layers == 4
    assert metadata["source_layers"] == 8
```

- [ ] **Step 2: Run the focused test and confirm the function is missing**

- [ ] **Step 3: Implement reduction, local-only loading, processor copying, atomic metadata writing, and reload validation**

The build command must call `AutoModelForVision2Seq.from_pretrained(..., torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)`, replace the Llama `ModuleList`, update both configs, set `architectures`, save with safe serialization, and reload with local-only mode before reporting success.

- [ ] **Step 4: Run unit tests, build the real base, and reload it**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m tools.openvla_4l_experiment build-base `
  --source models/openvla-7b `
  --output models/experiment_4_6_openvla_4l/base `
  --layers 4
```

Expected: `build_report.json` says four layers and reload validation passes.

### Task 3: Implement a memory-safe unified trainer and evidence logger

**Files:**
- Modify: `tools/openvla_4l_experiment.py`
- Modify: `tests/test_openvla_4l_experiment.py`

**Interfaces:**
- Produces: `collect_five_shot_paths(data_root) -> list[Path]`
- Produces: `gpu_snapshot() -> dict`
- Produces: `render_training_evidence(report_path, output_png) -> None`
- CLI: `train --mode {full,lora_r32,last_layer_only,frozen_vision}`

- [ ] **Step 1: Add failing tests for five-shot selection, LoRA metadata, JSONL logging, and evidence rendering**

Tests assert exact seeds, rank 32, non-empty loss history, monotonic step numbers, and a PNG with width and height greater than 500 pixels.

- [ ] **Step 2: Run the tests and confirm the missing behavior**

- [ ] **Step 3: Implement the trainer**

The trainer must:

```python
model.gradient_checkpointing_enable()
model.config.use_cache = False
optimizer = transformers.Adafactor(
    trainable_parameters,
    lr=learning_rate,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False,
)
```

For LoRA it must inject `LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", lora_dropout=0.0, task_type="CAUSAL_LM")`. For other modes it must apply `is_trainable`. Each step records raw loss, peak allocated VRAM, current `nvidia-smi` values, elapsed time, and parameter counts. Saving must preserve processor files and water-cup action statistics.

- [ ] **Step 4: Run tests and one-step smoke training for all four modes**

Expected: each smoke run exits zero, performs an optimizer step, and writes a reloadable checkpoint.

### Task 4: Implement direct checkpoint inference and pure-policy evidence capture

**Files:**
- Modify: `tools/openvla_4l_experiment.py`
- Modify: `tests/test_openvla_4l_experiment.py`

**Interfaces:**
- Produces: `select_keyframe_steps(records) -> tuple[int, ...]`
- Produces: `render_rollout_evidence(frames, report, output_png) -> None`
- CLI: `evaluate --checkpoint ... --seeds ... --trials-per-seed ...`

- [ ] **Step 1: Add failing tests for deterministic keyframe selection and report-derived success**

```python
def test_keyframes_include_start_middle_and_terminal():
    assert select_keyframe_steps([{"step": i} for i in range(9)]) == (0, 4, 8)

def test_success_rate_uses_episode_booleans():
    assert success_rate([{"success": True}, {"success": False}]) == 50.0
```

- [ ] **Step 2: Run tests and confirm failure**

- [ ] **Step 3: Implement a local checkpoint predictor and RoboCasa evaluator**

The evaluator loads a full saved checkpoint directly, or loads the LoRA adapter on the shared base, installs water-cup action statistics, uses `predict_action` without expert recovery, saves every queried camera frame, and writes one report row per episode.

- [ ] **Step 4: Smoke-evaluate one episode per method**

Expected: each model produces finite 7-D actions and a report even when the episode is unsuccessful.

### Task 5: Run formal training and tune within a fixed audit trail

**Files:**
- Write: `outputs/experiment_4_6/openvla_4l/<mode>/training_report.json`
- Write: `models/experiment_4_6_openvla_4l/<mode>/`

- [ ] **Step 1: Train LoRA and Last-Layer-Only with the shared initial hyperparameters**

- [ ] **Step 2: Train Full and Frozen-Vision with identical data order and Adafactor**

- [ ] **Step 3: If loss is non-finite or closed-loop success is poor, tune learning rate and steps without changing method definitions**

Every attempt must be stored under a unique run ID with its configuration and result. The selected run is the best measured run, not an overwritten result.

- [ ] **Step 4: Confirm all four selected checkpoints reload and generate finite actions**

### Task 6: Run formal closed-loop evaluation and generate Section 4.6.1

**Files:**
- Modify: `tools/generate_461_project_results.py`
- Write: `outputs/experiment_4_6/openvla_4l/summary.json`
- Write: `outputs/experiment_4_6/openvla_4l/4.6.1_实验验证.md`
- Write: `outputs/experiment_4_6/openvla_4l/evidence/*.png`

- [ ] **Step 1: Evaluate all four selected checkpoints with the same seeds and trial count**

- [ ] **Step 2: Render per-method training/GPU evidence and rollout keyframe contact sheets**

- [ ] **Step 3: Aggregate only raw JSON values into the final table and prose**

- [ ] **Step 4: Run complete verification**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tests/test_openvla_4l_experiment.py -q
C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m compileall -q tools/openvla_4l_experiment.py tools/generate_461_project_results.py
```

Then assert four training reports, four checkpoints, four evaluation reports, eight evidence images, finite parameter/VRAM values, and success rates exactly matching episode booleans.

