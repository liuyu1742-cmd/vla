# OpenVLA-Simulator

独立的视觉—语言—动作仿真项目。

## 范围

- 输入：RoboCasa/MuJoCo 的机器人相机画面与自然语言指令。
- 模型：本地 OpenVLA-7B 权重。
- 输出：7 维连续末端执行器动作；经适配后转换为 RoboCasa 的动作字典。
- 验证：仅在 MuJoCo 仿真器中执行和记录任务结果，不连接真实机械臂。

## 内容

- `models/openvla-7b/`：OpenVLA-7B 本地权重与处理器文件。
- `third_party/openvla/`：OpenVLA 源码。
- `third_party/robocasa/`、`third_party/robosuite/`：仿真运行依赖和现有仿真资源。
- `datasets/robocasa_vla/`、`datasets/robocasa_bc_pick_place/`：现有仿真轨迹清单与行为克隆样本。
- `datasets/robocasa_detection_*`、`datasets/robocasa365/`：仿真视觉检测样本与配置。
- `models/robocasa_bc_*`、`outputs/robocasa_*`：已有仿真行为克隆模型和运行记录。
- `tools/`：模型下载、轨迹清单和动作适配工具。

本项目不保存 EPIC、COCO、120/123 个物体图片库或动作解析验收数据；这些保留在 `RobotProject`。
