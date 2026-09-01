# Water-Cup OpenVLA Zero-Shot Loop Design

## Goal

Run one auditable, closed-loop RoboCasa `water_cup` household manipulation episode in which a simulator camera observation and English instruction are sent to the local OpenVLA-7B model, then adapt and execute the predicted action in the simulator.

## Scope

- Reuse the existing local OpenVLA Conda environment and `models/openvla-7b` checkpoint.
- Reuse the existing stationary-base RoboCasa action mapping in `tools/openvla_simulator_adapter.py`.
- Use one existing `water_cup` task setup and a fixed seed.
- Record a JSON report with instruction, camera name, raw OpenVLA actions, adapted RoboCasa action groups, per-step simulator outcome, final success status, and any exception.
- Save a short video only when the simulator renderer is available.

## Non-goals

- Do not train or fine-tune OpenVLA.
- Do not read, alter, or require `datasets/oxe/bridge_orig`.
- Do not claim task success from a single action or placeholder image.
- Do not control a physical robot.

## Approach

The runner creates the known `water_cup` environment with a deterministic seed and retrieves an actual configured camera frame at each control step. It formats the existing OpenVLA prompt, calls `predict_action(..., unnorm_key="bridge_orig", do_sample=False)`, then passes the seven-element result through the existing adapter. The adapter holds base motion and control mode at zero; the runner calls the environment's native step method with the resulting action dictionary.

The first run is deliberately bounded to a small, explicit number of steps. This establishes integration correctness and captures the scale/coordinate behavior before any longer zero-shot evaluation. A simulator exception, unavailable camera image, invalid action shape, or model error terminates the run and is reported as a failure rather than being hidden.

## Acceptance criteria

1. A unit test validates that the runner rejects malformed model actions before simulator execution.
2. The runner uses a real RoboCasa image rather than the placeholder image from the prior model smoke test.
3. The report records at least one raw OpenVLA action and the corresponding adapted RoboCasa action.
4. The report clearly distinguishes runner completion from the simulator's task-success signal.
5. The command can be run from PyCharm using the `openvla` interpreter.

## Risks

BridgeData V2 and RoboCasa use different robots, cameras, task distributions, and action conventions. Zero-shot success is not expected; the intended outcome is a truthful diagnosis of the first mismatch. The report will be the basis for a later action-calibration and RoboCasa-demonstration collection task.
