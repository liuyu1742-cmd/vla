# Organizing Toy DAgger R2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct `organizing::toy` held-out visual-state generalization by collecting expert labels on states visited by the augmented-r2 policy, continuing LoRA training from augmented-r2, and rerunning the audited three-seed acceptance.

**Architecture:** Keep seeds 101/102/103 immutable and held out. On new training seeds, query the current OpenVLA policy, probabilistically mix its action with the adaptive safe-cabinet oracle, but always save the oracle action as the training label. Aggregate those recovery episodes with the 36 successful nominal episodes and continue training from the augmented-r2 adapter.

**Tech Stack:** Python 3.10/3.11, NumPy, RoboCasa/Gymnasium, PyTorch, PEFT LoRA, unittest, local TCP IPC.

## Global Constraints

- Never add seeds 101, 102, or 103 to training.
- Simulator state is permitted only for DAgger labels, safety mixing, and diagnostics.
- The deployed policy input remains camera image, instruction, and canonical phase.
- The formal gripper convention is `-1=open`, `+1=closed`.
- A hybrid-v2 success must not be reported as pure autonomous VLA because final-contact servo remains active.

---

### Task 1: Formal DAgger data and mixing contract

**Files:**
- Create: `tools/formal_skill_dagger.py`
- Create: `tests/test_formal_skill_dagger.py`

**Interfaces:**
- Consumes: `PickPlaceDecision`, policy 7D action, held-out seed list.
- Produces: `validate_training_seed`, `choose_executed_action`, and `save_dagger_episode`.

- [ ] **Step 1: Write failing tests**

```python
def test_heldout_seed_is_rejected():
    with self.assertRaises(ValueError):
        validate_training_seed(101, {101, 102, 103})

def test_premature_policy_close_is_fully_opened():
    result = choose_executed_action(policy, oracle, beta=0.0, rng=FixedRng(1.0))
    self.assertEqual(-1.0, result.executed_action[6])

def test_saved_actions_are_oracle_labels():
    episode, _ = save_dagger_episode(...)
    with np.load(episode) as data:
        np.testing.assert_allclose(data["actions"], oracle_actions)
```

- [ ] **Step 2: Run test and verify RED**

Run:
`C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tests.test_formal_skill_dagger -v`

Expected: import failure because `tools.formal_skill_dagger` does not exist.

- [ ] **Step 3: Implement minimal contract**

Validate finite `(7,)` actions, reject held-out seeds, force oracle actions in forced phases, use `-1` for gated open-gripper approach, and save the full formal episode schema plus policy/executed audit arrays.

- [ ] **Step 4: Run test and verify GREEN**

Run the Task 1 test command; expect all tests to pass.

### Task 2: Policy-state recovery collector

**Files:**
- Create: `tools/collect_formal_skill_dagger.py`
- Extend: `tests/test_formal_skill_dagger.py`

**Interfaces:**
- Consumes: balanced formal TCP service, `create_formal_env`, `AdaptiveLiftSafeCabinetPickPlaceOracle`.
- Produces: one recovery `.npz`, report, and sidecar manifest per non-held-out seed.

- [ ] **Step 1: Add failing parser and validation tests**

Test that the collector rejects held-out seeds from the active formal manifest and requires `0 <= beta <= 1`.

- [ ] **Step 2: Verify RED**

Run the Task 1 test command; expect missing collector behavior.

- [ ] **Step 3: Implement collector**

At each decision, save the camera frame, query OpenVLA with the oracle canonical phase, compute the oracle label, mix actions deterministically, step RoboCasa, and log policy/oracle/executed actions and recovery diagnostics.

- [ ] **Step 4: Verify GREEN and compile**

Run the Task 1 tests and:
`C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m compileall -q tools\collect_formal_skill_dagger.py`

### Task 3: Audited DAgger manifest

**Files:**
- Create: `tools/prepare_formal_skill_dagger_manifest.py`
- Create: `tests/test_prepare_formal_skill_dagger_manifest.py`

**Interfaces:**
- Consumes: current nominal manifest and recovery sidecars.
- Produces: `training_manifest_dagger_r2.json` with original held-out entries unchanged.

- [ ] **Step 1: Write failing split-integrity test**

Construct nominal and recovery fixtures, assert no held-out seed is accepted, nominal plus recovery sources are retained, and weighted sample counts are exact.

- [ ] **Step 2: Verify RED**

Run:
`C:\Users\sjtu101\miniconda3\envs\openvla\python.exe -m unittest tests.test_prepare_formal_skill_dagger_manifest -v`

- [ ] **Step 3: Implement manifest builder**

Preserve schema/relation/Skill-IR hash/held-out entries, add recovery entries with `source=dagger_recovery`, and compute exact episode/sample counts.

- [ ] **Step 4: Verify GREEN**

Run the Task 3 test command; expect all tests to pass.

### Task 4: Continue training from augmented-r2

**Files:**
- Modify: `tools/finetune_formal_skill_openvla_resumable.py`
- Create: `tests/test_finetune_formal_skill_initial_adapter.py`

**Interfaces:**
- Consumes: `--initial-adapter` directory when no same-manifest checkpoint exists.
- Produces: trainable PEFT model initialized from augmented-r2 and an audited fingerprint in `training_report.json`.

- [ ] **Step 1: Write failing argument-contract tests**

Assert an initial adapter must contain `adapter_config.json` and `adapter_model.safetensors`, and cannot be combined with a same-run resume checkpoint.

- [ ] **Step 2: Verify RED**

Run the new test; expect missing initial-adapter validation.

- [ ] **Step 3: Implement minimal initialization path**

When no checkpoint is selected, load the base model and then `PeftModel.from_pretrained(..., is_trainable=True)` from the initial adapter; retain the existing fresh-LoRA path when the option is absent.

- [ ] **Step 4: Verify GREEN and regression suite**

Run the new test plus all formal training, dataset, action-codec, and prompt tests.

### Task 5: Collection, training, and acceptance

**Files:**
- Generate: `datasets/formal_skills/organizing_toy_dagger_r2/`
- Generate: `datasets/formal_skills/organizing_toy/training_manifest_dagger_r2.json`
- Generate: `models/openvla-organizing-toy-lora-dagger-r3/`
- Generate: `outputs/formal_skill_dagger_r3_eval/`

**Interfaces:**
- Consumes: new tools from Tasks 1–4.
- Produces: recovery data, trained adapter, probes, three 300-step reports, and acceptance summary.

- [ ] **Step 1: Start balanced policy service**

Run `tools.openvla_formal_skill_tcp_server_balanced` with the augmented-r2 adapter on port 8773.

- [ ] **Step 2: Collect non-held-out recovery episodes**

Use deterministic seeds outside both the 36 nominal set and 101/102/103, beta `0.5`, and retain every valid oracle-labeled visited state.

- [ ] **Step 3: Build and audit the DAgger manifest**

Require zero train/held-out overlap and exact file/sample counts before training.

- [ ] **Step 4: Continue LoRA training**

Train one audited DAgger epoch from augmented-r2, monitor finite loss, checkpoints, OOM/Traceback, GPU memory, and temperature.

- [ ] **Step 5: Run offline probes**

Require plausible x/y/z/gripper behavior on seeds 101/102/103 before long rollouts.

- [ ] **Step 6: Run three 300-step hybrid-v2 rollouts**

Run seeds 101, 102, and 103 sequentially and aggregate success predicates, grasp/inside state, execution-mode counts, and final-contact-servo use.
