# VLA Go2 Robot Control Project

基于NVIDIA Isaac Lab的Go2四足机器人视觉-语言-动作（VLA）控制系统。

## 功能特性

- **文本命令控制**: 支持自然语言指令（如"go forward", "turn left", "stop"）
- **视觉规则控制**: 基于颜色的目标跟踪（红球跟随）
- **多模态控制**: 结合图像和文本输入
- **VLA模型支持**: 支持SmolVLM、Qwen2-VL、Qwen3.5等模型
- **可插拔后端**: rule/dummy_vla/real_vla三种后端

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/VLA.git
cd VLA
```

### 2. 安装IsaacLab

本项目基于NVIDIA IsaacLab，需要先安装完整环境：

```bash
# 创建conda环境
conda create -n isaac python=3.11
conda activate isaac

# 安装Isaac Sim
pip install isaacsim[all,extscache]==5.1.0

# 克隆并安装IsaacLab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
pip install -e source/isaaclab
pip install -e source/isaaclab_tasks
pip install -e "source/isaaclab_rl[all]"

# 安装RL库
pip install rsl-rl-lib
```

### 3. 安装VLA依赖

```bash
pip install transformers>=4.50.0
pip install opencv-python pillow numpy
```

### 4. 下载VLA模型（可选）

如果要使用真实VLA模型：

```bash
mkdir -p models/vla

# SmolVLM-256M
huggingface-cli download HuggingFaceTB/SmolVLM-256M-Instruct \
  --local-dir models/vla/SmolVLM-256M-Instruct

# Qwen2-VL-2B
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct \
  --local-dir models/vla/Qwen2-VL-2B-Instruct

# Qwen3.5-9B
huggingface-cli download Qwen/Qwen3.5-9B \
  --local-dir models/vla/Qwen3.5-9B
```

### 5. 运行测试

#### 基础训练
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --headless
```

#### 基础回放
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --headless \
  --num_envs 1 \
  --enable_cameras \
  --checkpoint checkpoints/model_599.pt
```

#### 文本命令控制
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --headless \
  --num_envs 1 \
  --enable_cameras \
  --checkpoint checkpoints/model_599.pt \
  --text_command "go forward"
```

#### 视觉跟踪红球（规则后端）
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --headless \
  --num_envs 1 \
  --enable_cameras \
  --checkpoint checkpoints/model_599.pt \
  --multimodal \
  --multimodal_prompt "follow the red ball" \
  --backend rule \
  --video --video_length 300
```

#### VLA模型控制（Qwen3.5）
```bash
CUDA_VISIBLE_DEVICES=2,1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --device cuda:0 \
  --num_envs 1 \
  --headless \
  --enable_cameras \
  --checkpoint checkpoints/model_599.pt \
  --multimodal \
  --backend real_vla \
  --vla_model_path models/vla/Qwen3.5-9B \
  --vla_device cuda:1 \
  --vla_action_mode discrete \
  --vla_num_frames 4 \
  --vla_frame_stride 3 \
  --vla_infer_interval 10 \
  --multimodal_prompt "follow the red ball"
```

## 项目结构

```
VLA/
├── scripts/
│   └── reinforcement_learning/
│       └── rsl_rl/
│           ├── train.py                          # 训练脚本
│           ├── play.py                           # 基础回放
│           ├── play_with_high_level_commands.py  # VLA控制（主脚本）
│           └── cli_args.py                       # CLI参数
├── checkpoints/
│   └── model_599.pt                              # 预训练checkpoint
├── docs/
│   └── qwen35_vla_status_2026-03-17.md          # VLA状态文档
└── README.md                                     # 本文件
```

## 关键参数说明

### VLA参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--backend` | 后端类型 | `rule`, `dummy_vla`, `real_vla` |
| `--vla_model_path` | VLA模型路径 | `models/vla/Qwen3.5-9B` |
| `--vla_device` | VLA设备 | `cuda:1` |
| `--vla_action_mode` | 动作模式 | `discrete`, `continuous` |
| `--vla_num_frames` | 输入帧数 | 4 |
| `--vla_frame_stride` | 帧采样间隔 | 3 |
| `--vla_infer_interval` | 推理间隔 | 10 |

### 相机与视频

| 参数 | 说明 |
|------|------|
| `--enable_cameras` | 启用相机（必须） |
| `--video` | 录制视频 |
| `--video_length` | 视频长度（步数） |

## 支持的命令

### 文本命令
- `go forward` / `forward` / `move forward`
- `go backward` / `backward` / `move backward` / `back`
- `turn left` / `left`
- `turn right` / `right`
- `stop`

### 视觉命令
- `follow the red ball`
- `follow the [color] target`

## 环境要求

- **操作系统**: Ubuntu 22.04
- **GPU**: NVIDIA GPU with CUDA support
- **内存**: 32GB+
- **显存**: 24GB+ (用于Qwen3.5)

## 常见问题

### "CUDA in bad state"
```bash
pkill -f isaac
pkill -f python
# 或重启机器
```

### 显存不足
```bash
# 使用独立GPU
CUDA_VISIBLE_DEVICES=2,1 ./isaaclab.sh ...
--device cuda:0      # Isaac Sim
--vla_device cuda:1  # VLA模型
```

### "transformers not support qwen3_5"
```bash
pip install --upgrade transformers
```

## 参考

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)

## 许可证

本项目基于BSD-3-Clause许可证。

---

**作者**: liangc001
**日期**: 2026-03-24
