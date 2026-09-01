# OpenVLA PyCharm 环境

本项目的 OpenVLA 推理使用独立 Conda 环境，避免更改现有的 `robocasa` 与 `robosim` 环境。

1. 在 PyCharm 打开 `File > Settings > Project: OpenVLA-Simulator > Python Interpreter`。
2. 选择 **Add Interpreter > Add Local Interpreter > Conda > Existing environment**。
3. 选择 `C:\\Users\\sjtu101\\miniconda3\\envs\\openvla\\python.exe` 并应用。
4. 在右上角 Run 配置中选择 `OpenVLA CUDA Environment Check`，点击运行。

该检查只验证 Python、CUDA、GPU 与本地模型文件状态；不会访问 `datasets/oxe/bridge_orig`。首次模型动作推理配置将在 CUDA 检查通过后加入。
