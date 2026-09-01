# Phase Planning and Action Calibration Plan

### Task 1: Two-phase skill planner

- Create `tools/skill_phase_planner.py` and `tests/test_skill_phase_planner.py`.
- Test first: `pick` transitions to `place` only when `grasped=True`; otherwise it remains `pick`.
- Implement prompts and JSON-safe transition reports from the EPIC skill manifest.

### Task 2: RoboCasa 12D calibration probes

- Create `tools/calibrate_robocasa_action_space.py` and `tests/test_calibrate_robocasa_action_space.py`.
- Test first: a 7D source action becomes a 12D native action by appending five zeros.
- Run deterministic single-axis probes in RoboCasa, record observed end-effector deltas and output `outputs/robocasa_action_calibration.json`.

### Task 3: Calibrated IPC micro-step

- Create `tools/run_calibrated_ipc_step.py`.
- Consume the phase prompt, calibration report and IPC action response; execute one calibrated 12D action and report phase predicate state.
