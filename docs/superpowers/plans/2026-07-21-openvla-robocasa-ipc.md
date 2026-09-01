# OpenVLA–RoboCasa IPC Loop Implementation Plan

**Goal:** Execute one bounded RoboCasa water-cup episode through isolated OpenVLA inference.

### Task 1: OpenVLA action worker

- Create `tools/openvla_ipc_predict.py` and `tests/test_openvla_ipc_predict.py`.
- Test first: reject a request missing `image_path` before model loading.
- Implement a CLI taking `--request` and `--response`; it loads the local image and checkpoint in the `openvla` environment, writes `raw_action: list[float]` of length seven, or `error`.
- Verify with the existing smoke image and `C:\Users\sjtu101\miniconda3\envs\openvla\python.exe`.

### Task 2: RoboCasa IPC episode runner

- Create `tools/run_robocasa_openvla_ipc.py` and `tests/test_run_robocasa_openvla_ipc.py`.
- Test first: reject non-seven-dimensional worker responses and JSON-serialize adapted actions.
- Run from the RoboCasa environment; render `robot0_agentview_left_image` to PNG, invoke the OpenVLA worker as a subprocess, adapt with `to_robocasa_action`, execute at most ten steps, and write `outputs/openvla_robocasa_ipc/report.json`.
- Verify that the final report records each rendered path, raw action, adapted action, and termination state without reading BridgeData V2.
