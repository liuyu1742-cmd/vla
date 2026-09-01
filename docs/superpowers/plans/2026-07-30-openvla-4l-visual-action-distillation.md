# OpenVLA-4L Visual-Action Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distill simulator-state teacher corrections into OpenVLA-4L, obtain a non-zero pure closed-loop success rate when possible, and otherwise produce a clearly disclosed non-zero hybrid recovery result.

**Architecture:** A dataset builder combines the same five nominal demonstrations with unique DAgger teacher-labeled states while capping settle actions. A small pipeline module defines round progression and pure-to-hybrid stopping rules. Existing OpenVLA-4L training and TCP inference are reused; a separate hybrid evaluator records student/expert control provenance.

**Tech Stack:** Python 3.10/3.11, PyTorch, Transformers, PEFT, NumPy, Pillow, RoboCasa, robosuite, PowerShell.

## Global Constraints

- Use Full Fine-tuning, LoRA rank=32, Last-Layer-Only, and Frozen-Vision only.
- Pure success never includes expert recovery or rule-controller actions.
- Hybrid results are labeled separately and include recovery step counts.
- Training uses seeds 0, 1, 3, 5 for DAgger; seed 2 is not used for DAgger.
- Same distilled dataset version is used for final four-method comparison.
- Same bug failing three times triggers comprehensive diagnosis and continued repair.

---

### Task 1: Distillation Dataset Contract

**Files:**
- Create: `tests/test_openvla_4l_distillation_dataset.py`
- Create: `tools/openvla_4l_distillation_dataset.py`

**Interfaces:**
- Produces: `action_phase(action) -> str`
- Produces: `file_sha256(path) -> str`
- Produces: `select_indices(actions, max_settle_fraction, limit, seed) -> list[int]`
- Produces: `build_distillation_dataset(nominal_root, recovery_root, output_root, seeds, max_settle_fraction, samples_per_seed) -> dict`

- [ ] **Step 1: Write failing tests**

Test that non-finite or non-seven-dimensional actions are rejected, duplicate recovery hashes are rejected, settle samples are at most 15%, and all output files contain aligned `frames/actions`.

- [ ] **Step 2: Verify RED**

Run:
`C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tests\test_openvla_4l_distillation_dataset.py -q`

Expected: import failure because `tools.openvla_4l_distillation_dataset` does not exist.

- [ ] **Step 3: Implement the minimal builder**

Load nominal `episode_seed_*.npz`, load recovery `episode_seed_*_round_*.npz`, use only `actions` (oracle labels), hash each recovery file, sample motion first, cap settle samples, and write one output NPZ per seed plus `distillation_manifest.json`.

- [ ] **Step 4: Verify GREEN**

Run the Task 1 pytest command.

Expected: all Task 1 tests pass.

### Task 2: Round and Fallback Policy

**Files:**
- Create: `tests/test_openvla_4l_distillation_pipeline.py`
- Create: `tools/openvla_4l_distillation_pipeline.py`

**Interfaces:**
- Produces: `round_beta(round_index) -> float`
- Produces: `next_action(pure_successes, round_index, max_rounds=3) -> str`
- Produces values: `accept_pure`, `collect_next_round`, `run_hybrid`

- [ ] **Step 1: Write failing tests**

Test beta values `0.7, 0.3, 0.0`, immediate pure acceptance for one success, next-round collection before round three, and hybrid fallback after round three.

- [ ] **Step 2: Verify RED**

Run:
`C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m pytest tests\test_openvla_4l_distillation_pipeline.py -q`

Expected: import failure for the missing pipeline module.

- [ ] **Step 3: Implement minimal deterministic policy**

Use a fixed beta tuple and validate non-negative success counts and round indexes.

- [ ] **Step 4: Verify GREEN**

Run both new test files.

Expected: all tests pass.

### Task 3: Build Initial Distilled Dataset

**Files:**
- Generate: `outputs/experiment_4_6/openvla_4l/distilled_round0/episode_seed_*.npz`
- Generate: `outputs/experiment_4_6/openvla_4l/distilled_round0/distillation_manifest.json`

**Interfaces:**
- Consumes existing nominal stride-1 demonstrations and unique DAgger recovery files.
- Produces a five-file dataset compatible with `tools.openvla_4l_runner`.

- [ ] **Step 1: Run the builder**

Use nominal root `datasets/water_cup_expert_stride1`, recovery root `datasets/water_cup_dagger`, seeds `0,1,2,3,5`, settle cap `0.15`, and 900 samples per seed.

- [ ] **Step 2: Verify the manifest**

Assert five output files, aligned arrays, finite seven-dimensional actions, no duplicate recovery hashes, and settle fraction at most 0.15.

### Task 4: LoRA Distillation Probe

**Files:**
- Generate: `models/experiment_4_6_openvla_4l/lora_r32_distilled_r0`
- Generate: `outputs/experiment_4_6/openvla_4l/lora_r32_distilled_r0`

- [ ] **Step 1: Train LoRA rank=32**

Run 300 optimizer steps, effective batch 16, learning rate `1e-4`, using the distilled round-0 dataset.

- [ ] **Step 2: Run seed-0 geometry diagnostic**

Run 300 steps and require minimum object—EEF distance below the prior 0.35m baseline.

- [ ] **Step 3: Run five-seed pure evaluation**

Use seeds `0,1,2,3,5`, max 700 steps, no expert recovery.

- [ ] **Step 4: Decide**

If at least one seed succeeds, freeze the dataset. Otherwise collect the next DAgger round with the beta from Task 2, rebuild, retrain, and repeat up to round two.

### Task 5: Hybrid Recovery Fallback

**Files:**
- Create: `tools/openvla_4l_hybrid_eval.py`
- Create: `tools/run_openvla_4l_hybrid_eval.ps1`
- Generate: `outputs/experiment_4_6/openvla_4l/hybrid_*`

**Interfaces:**
- Consumes the TCP student action and `WaterCupDaggerOracle`.
- Produces a report containing `pure_success`, `hybrid_success`, `student_steps`, `expert_recovery_steps`, `expert_recovery_ratio`, and per-step control source.

- [ ] **Step 1: Write a failing provenance test**

Verify reports reject missing control sources and cannot mark pure success when any expert step was executed.

- [ ] **Step 2: Verify RED**

Run the focused test and observe the missing evaluator failure.

- [ ] **Step 3: Implement stalled-progress recovery**

Use the expert after 20 non-improving steps and retain expert control until the oracle phase advances or the object is securely grasped.

- [ ] **Step 4: Verify GREEN**

Run the focused test and one seed hybrid evaluation. Require `hybrid_success=true`.

### Task 6: Train and Evaluate Four Methods

**Files:**
- Generate four distilled checkpoints and four training evidence directories.
- Generate four pure reports and, when needed, four hybrid reports.

- [ ] **Step 1: Train all four exact modes**

Use one frozen distilled dataset version and identical optimizer-step/effective-batch settings.

- [ ] **Step 2: Evaluate pure policies**

Run the same five seeds and horizon for all methods.

- [ ] **Step 3: Evaluate zero-success methods with hybrid recovery**

Do not overwrite pure reports.

- [ ] **Step 4: Verify numeric success for every method**

Each method must have a numeric pure rate. Each method with pure rate zero must have a non-zero, separately labeled hybrid rate.

### Task 7: Regenerate Section 4.6.1

**Files:**
- Create: `tools/generate_openvla_461_distilled_report.py`
- Generate: `outputs/experiment_4_6/4.6.1_distilled.md`
- Generate: `outputs/experiment_4_6/4.6.1_distilled_process_evidence.png`
- Generate: `outputs/experiment_4_6/4.6.1_distilled_result_table.png`

- [ ] **Step 1: Aggregate only real reports**

Reject missing trials, missing provenance, or a hybrid report mislabeled as pure.

- [ ] **Step 2: Render process evidence**

Include actual distillation frames, training frames, pure rollout frames, and hybrid recovery frames when used.

- [ ] **Step 3: Run final verification**

Compile new scripts, run focused tests plus the existing 15 OpenVLA-4L tests, verify all images load, and verify every method has numeric success metrics.
