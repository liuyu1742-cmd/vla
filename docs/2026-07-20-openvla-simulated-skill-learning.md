# OpenVLA 模拟技能学习 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a truthful OpenVLA readiness-to-simulator pipeline in which simulated camera observations and task instructions produce adaptable continuous robot actions, with measurable simulator outcomes.

**Architecture:** Start from the current RoboCasa/MuJoCo capability instead of assuming working physics. Export a minimal robot demonstration contract, validate and scale continuous actions, assess OpenVLA dependencies/resources, and only then optionally download/run OpenVLA. All reports separate readiness, trajectory export, action legality, and simulator completion.

**Tech Stack:** Python 3.11; existing RoboCasa/MuJoCo installation; PyTorch; OpenVLA public code/checkpoint after approval; JSON/NPZ.

## Global Constraints

- Use simulated robot observations/actions only for the VLA layer; do not use human EPIC videos as robot action supervision.
- Do not download OpenVLA code or weights until readiness reports actual disk/VRAM/dependency requirements and the user authorizes it.
- Do not claim real-robot, OpenVLA, or simulator success without a local run.
- Keep human-video work as optional extension; no existing human-video results count toward this pipeline.

---

### Task 1: Establish a minimal VLA trajectory contract from the local simulator

**Files:**
- Create: `tools/export_robocasa_vla_trajectories.py`
- Create: `tests/test_export_robocasa_vla_trajectories.py`
- Create: `datasets/robocasa_vla/manifest.json`
- Create: `datasets/robocasa_vla/README.md`

**Interfaces:**
- Produces episodes containing `instruction`, `task_id`, `object`, `observation_paths`, `actions`, `action_dim`, `control_hz`, `success`, `failure_reason`, and `seed`.

- [ ] Write failing contract tests for synchronized observations/actions and simulator metadata.
- [ ] Run `python -m unittest tests.test_export_robocasa_vla_trajectories -v` and observe failure.
- [ ] Implement environment inspection and a minimal exporter. It must produce an explicit simulator-unavailable manifest if local MuJoCo/RoboCasa cannot start.
- [ ] Re-run the test and exporter; retain only actual exported episodes.

### Task 2: Implement and test simulator action adaptation

**Files:**
- Create: `tools/openvla_simulator_adapter.py`
- Create: `tests/test_openvla_simulator_adapter.py`

**Interfaces:**
- Produces `adapt_openvla_action(model_action, low, high, gripper_index) -> list[float]`.

- [ ] Write failing tests for 7-D scaling, clipping and incompatible dimensions.
- [ ] Run `python -m unittest tests.test_openvla_simulator_adapter -v` and observe failure.
- [ ] Implement dimension validation and normalized-to-simulator scaling.
- [ ] Re-run adapter tests and retain exact action-space metadata in the trajectory manifest.

### Task 3: Measure OpenVLA readiness before model acquisition

**Files:**
- Create: `tools/check_openvla_simulator_readiness.py`
- Create: `tests/test_check_openvla_simulator_readiness.py`
- Create: `models/openvla_simulator/readiness_report.json`

**Interfaces:**
- Produces `check_readiness() -> dict` with Python package availability, GPU name/VRAM, free disk, simulator trajectory availability, model/checkpoint availability, and `ready_for_download` / `ready_for_inference` statuses.

- [ ] Write failing tests for missing required components and a fully specified mocked environment.
- [ ] Run `python -m unittest tests.test_check_openvla_simulator_readiness -v` and observe failure.
- [ ] Implement inspection only; no model install or download in this task.
- [ ] Re-run tests and create actual local readiness report.

### Task 4: Acquire and smoke-test OpenVLA only when approved and ready

**Files:**
- Create: `tools/run_openvla_simulator_smoke.py`
- Create: `models/openvla_simulator/smoke_report.json`

**Interfaces:**
- Consumes a ready report, one simulator image/instruction, and declared action space.
- Produces an adapted action, legality result, and simulator outcome.

- [ ] Present actual readiness findings and request explicit approval for OpenVLA code/checkpoint download.
- [ ] Install/use only the approved official release and record exact model identifier/version.
- [ ] Run one deterministic simulator episode; record each input, raw output, adapted action, legality result, and success/failure.

### Task 5: Freeze evaluation and report measured simulation results

**Files:**
- Create: `docs/OPENVLA_SIMULATED_SKILL_LEARNING_REPORT.md`
- Create: `docs/OPENVLA_SIMULATED_SKILL_LEARNING_USAGE.md`
- Modify: `docs/CURRENT_PROJECT_STATUS_2026-07-04.md`

- [ ] Document the distinction between readiness, exported data, action legality, simulator outcome, and real-arm execution.
- [ ] Add only measured values to project status.
- [ ] Run `python -m unittest tests.test_export_robocasa_vla_trajectories tests.test_openvla_simulator_adapter tests.test_check_openvla_simulator_readiness -v` and rerun every report generator.
