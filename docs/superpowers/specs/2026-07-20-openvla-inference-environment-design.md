# OpenVLA Local Inference Environment Design

## Goal

Run the local `models/openvla-7b` checkpoint on the RTX 3090 for one GPU-backed, deterministic OpenVLA action prediction.  This is the prerequisite for connecting actions to RoboCasa household simulations.  BridgeData V2 is not required for this first milestone.

## Scope

- Create an isolated Conda environment named `openvla` with Python 3.10.
- Install the Windows-compatible CUDA build of the OpenVLA inference dependencies.
- Install the vendored OpenVLA source in editable mode.
- Add an in-repository verification command that reports Python, PyTorch/CUDA, GPU, checkpoint availability, and a small CUDA tensor operation.
- Add a PyCharm run configuration and interpreter-selection instructions that use the `openvla` environment.

## Non-goals

- Do not modify either existing `robosim` or `robocasa` environment.
- Do not stop, move, inspect, or depend on the active BridgeData V2 download.
- Do not install FlashAttention: its standard build path is Linux/CUDA-toolkit oriented and it is optional for the initial Windows inference smoke test.
- Do not begin LoRA training.  The official BridgeData recipe calls for approximately 27 GB VRAM while the installed RTX 3090 has 24 GB.
- Do not execute real-robot actions.

## Design

The new environment pins the upstream OpenVLA compatibility set: Python 3.10, PyTorch 2.2.x with a CUDA 12.1 wheel, torchvision 0.17.x, `transformers==4.40.1`, `tokenizers==0.19.1`, and `timm==0.9.10`.  NVIDIA driver 560.94 supports CUDA 12.1 runtime wheels even though it reports CUDA 12.6.

The environment will remain separate from RoboCasa because the checked-in RoboCasa environments use Python 3.11 and CPU-only PyTorch 2.7.1.  The initial runtime boundary is therefore a model-focused smoke test, followed later by an explicit simulator-bridge task that imports or serves OpenVLA without changing the simulator's dependency set.

`tools/check_openvla_environment.py` will be a read-only verifier.  It will fail clearly if CUDA is unavailable or the local model files are incomplete; it will not load the 7B model.  A later inference smoke script will load the model and call `predict_action` only after this verifier passes.

PyCharm will select `C:\\Users\\sjtu101\\miniconda3\\envs\\openvla\\python.exe` as the project interpreter.  A committed shared run configuration will expose the verifier from the PyCharm Run menu.  User-specific workspace metadata stays unmodified.

## Acceptance criteria

1. `conda run -n openvla python` reports Python 3.10.
2. `torch.cuda.is_available()` is true and reports the RTX 3090.
3. Pinned OpenVLA inference packages import successfully.
4. The vendored `third_party/openvla` package imports from its editable installation.
5. The verifier performs a CUDA tensor operation and reports model-checkpoint presence without accessing `datasets/oxe/bridge_orig`.
6. PyCharm can run the verifier through a checked-in configuration after selecting the `openvla` interpreter.

## Risks and follow-up

The initial 7B inference load may be close to the 24 GB GPU memory budget.  If it exceeds the budget, the next design iteration will evaluate CPU offload or 4-bit quantized loading; neither is part of the baseline environment.  After an inference prediction is recorded, a separate design will cover the OpenVLA-to-RoboCasa action bridge and a 3090-feasible fine-tuning method.
