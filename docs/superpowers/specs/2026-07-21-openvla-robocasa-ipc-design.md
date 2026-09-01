# OpenVLA–RoboCasa IPC Loop Design

## Goal

Run a bounded household-simulation episode while keeping OpenVLA and RoboCasa in their mutually incompatible Python environments.

## Architecture

`robocasa` owns MuJoCo, resets `PickPlaceCounterToCabinet`, renders `robot0_agentview_left_image`, and writes one request JSON containing an image file path and instruction. `openvla` reads that request, loads the local model, writes a seven-element action JSON, and exits. RoboCasa validates and adapts the action, executes it, and repeats for at most 10 steps.

The image is a PNG under `outputs/openvla_robocasa_ipc/`; JSON files are per-step and contain only paths, metadata, and numeric actions. Neither process reads BridgeData V2.

## Constraints

- RoboCasa remains on NumPy 2.2.5; OpenVLA remains on PyTorch 2.2.2+cu121 and NumPy 1.26.4.
- No physical camera or robot is used.
- Every step report records request, raw model action, adapted simulator action, and simulator transition state.
- Any subprocess failure is captured in the final report and stops the episode.

## Acceptance

One command launched from the RoboCasa environment produces an auditable report with at least one rendered image and one OpenVLA-predicted action, or reports the exact blocking subprocess error.
