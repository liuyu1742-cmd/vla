# Water-Cup DAgger Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect expert corrective labels on states visited by the current OpenVLA policy, retrain on nominal plus recovery data, and pass the geometry-matched seed 0 closed-loop evaluation before attempting held-out seed 2.

**Architecture:** Keep the upstream OpenVLA checkpoint and action tokenizer unchanged. Add a RoboCasa ground-truth oracle used only during DAgger data collection, mix policy and oracle actions to visit recoverable off-trajectory states, label every visited image with the oracle action, aggregate these samples with stride-1 demonstrations, then fine-tune and evaluate through the existing TCP split environment.

**Tech Stack:** Python 3.10, PyTorch/PEFT, OpenVLA 7B, Gymnasium, RoboCasa/robosuite, NumPy, unittest.

## Global Constraints

- The oracle may use simulator state only for training labels and evaluation diagnostics, never for deployed policy actions.
- RoboCasa task geometry must use `object_scale=0.7`, `glass_cup`, and `PickPlaceCounterToCabinet`.
- OpenVLA actions remain 7-DoF `(x, y, z, roll, pitch, yaw, gripper)` in `[-1, 1]`.
- Seed 2 remains held out from training.
- A new training iteration is accepted only after seed 0 succeeds in a 700-step closed loop.

---

### Task 1: Recovery Oracle State Machine

**Files:**
- Create: `tools/water_cup_dagger_oracle.py`
- Test: `tests/test_water_cup_dagger_oracle.py`

**Interfaces:**
- Consumes: end-effector position, object position, grasp state, and fixed cabinet waypoints.
- Produces: `OracleDecision(action: np.ndarray, phase: str, force_expert: bool)` from `WaterCupDaggerOracle.decide(snapshot)`.

- [ ] Write tests proving the oracle keeps the gripper open while approaching, closes only at contact, and returns to approach after a failed grasp.
- [ ] Run `python -m unittest tests.test_water_cup_dagger_oracle -v` and verify failure because the module is missing.
- [ ] Implement the minimal state machine using the same controller-frame position error and thresholds as `robocasa_gt_grasp.py`.
- [ ] Run the test again and require all cases to pass.

### Task 2: DAgger Mixing and Safety Contract

**Files:**
- Create: `tools/water_cup_dagger_mixing.py`
- Test: `tests/test_water_cup_dagger_mixing.py`

**Interfaces:**
- Consumes: policy action, oracle decision, mixture probability, deterministic RNG.
- Produces: the executed action while always preserving the oracle label for the dataset.

- [ ] Write tests proving forced phases always execute the oracle and movement phases honor deterministic mixture selection.
- [ ] Run the tests and verify they fail before implementation.
- [ ] Implement `choose_executed_action(...)` and contact-before-close safety gating.
- [ ] Run the tests and require all cases to pass.

### Task 3: Policy-State Recovery Collector

**Files:**
- Create: `tools/collect_water_cup_dagger.py`
- Test: `tests/test_collect_water_cup_dagger.py`

**Interfaces:**
- Consumes: TCP policy server, RoboCasa environment, oracle, seed, and beta.
- Produces: compressed NPZ with visited frames and oracle actions plus an auditable JSON report containing phase counts, policy/oracle execution counts, minimum object distance, grasp, lift, and success.

- [ ] Write tests for the saved sample contract and seed-2 rejection.
- [ ] Verify tests fail before implementation.
- [ ] Implement collection for training seeds only and save under `datasets/water_cup_dagger/`.
- [ ] Run a short collection smoke test and verify frames/actions have matching length and seven action dimensions.

### Task 4: Aggregate and Fine-Tune

**Files:**
- Create: `tools/prepare_water_cup_dagger_manifest.py`
- Create: `tools/finish_water_cup_dagger_pipeline.ps1`
- Test: `tests/test_prepare_water_cup_dagger_manifest.py`

**Interfaces:**
- Consumes: successful stride-1 demonstrations and DAgger recovery episodes.
- Produces: an active training manifest, LoRA adapter, and reports under unique DAgger paths.

- [ ] Write a failing split test ensuring seed 2 never enters training and both nominal/recovery sources are present.
- [ ] Implement aggregation without overwriting raw episodes.
- [ ] Run one complete fine-tuning epoch using `tools.finetune_water_cup_openvla_epochs_v2`.
- [ ] Save and verify adapter weights and training report.

### Task 5: Closed-Loop Acceptance

**Files:**
- Reuse: `tools/openvla_tcp_server_water_cup.py`
- Reuse: `tools/openvla_gym_water_cup_rollout_v2.py`

**Interfaces:**
- Consumes: DAgger LoRA adapter.
- Produces: seed 0 and, only after seed 0 passes, seed 2 reports.

- [ ] Run seed 0 with `object_scale=0.7`, query interval 1, and 700 steps.
- [ ] Require `success=true`; otherwise collect another DAgger round instead of adding nominal epochs.
- [ ] After seed 0 passes, run held-out seed 2 under the same settings.
- [ ] Record exact success, executed steps, and model/action artifacts in the project status document.
