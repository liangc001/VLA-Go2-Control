# VLA Go2 Control

<p align="center">
  <img src="https://img.shields.io/badge/IsaacLab-0.54.3-blue?style=flat-square" alt="IsaacLab">
  <img src="https://img.shields.io/badge/IsaacSim-5.1.0-green?style=flat-square" alt="IsaacSim">
  <img src="https://img.shields.io/badge/Python-3.11-orange?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/License-BSD--3--Clause-yellow?style=flat-square" alt="License">
</p>

<p align="center">
  <b>Vision-Language-Action Control for Unitree Go2 Quadruped Robot</b>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#demo-videos">Demos</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#usage">Usage</a>
</p>

---

## 📹 Demo Videos

### Go2 Baseline Locomotion

<p align="center">
  <img src="assets/images/go2_baseline_demo.gif" width="70%" alt="Go2 Baseline Locomotion Demo">
</p>

<p align="center">
  <em>Baseline policy walking forward with velocity commands (~4s preview)</em>
</p>

### Full Video Recordings

Higher quality videos are available in the local experiment logs:

| Video | Location | Duration |
|-------|----------|----------|
| **Baseline Playback** | `logs/rsl_rl/.../videos/play/rl-video-step-0.mp4` | ~4s |
| **Text Command Exp** | `logs/rsl_rl/.../videos/play_high_level_commands/2026-03-16_15-17-54/` | Variable |
| **Vision Control Exp** | `logs/rsl_rl/.../videos/play_high_level_commands/2026-03-16_15-31-23/` | Variable |
| **VLA Model Exp** | `logs/rsl_rl/.../videos/play_high_level_commands/2026-03-17_14-06-33/` | Variable |

> 🎬 **To view full videos**: Clone the repo and play the MP4 files locally, or record your own experiments using the commands below.

---

## 🏗️ Architecture

The VLA control system consists of the following layers:

| Layer | Description | Components |
|-------|-------------|------------|
| **Input** | Camera + Text | RGB Image, Text Prompt "go forward" |
| **Backend** | VLA Models | Rule / Dummy VLA / Real VLA (Qwen3.5) |
| **Control** | Velocity Commands | (vx, vy, wz) |
| **Execution** | Robot Policy | RSL-RL + Unitree Go2 |

**Supported VLA Models:**
- SmolVLM-256M-Instruct (3.3G)
- Qwen2-VL-2B-Instruct (4.2G)
- Qwen3.5-9B (19G) ⭐ Recommended

---

## ✨ Features
<tr>
<td width="50%">

### 🤖 Robot Control
- **Text Commands**: Natural language control
  - `go forward`, `turn left`, `stop`
  - `go backward`, `turn right`

- **Vision-Based**: Follow visual targets
  - Red ball tracking
  - Color-based detection

- **Multimodal**: Image + Text fusion
  - "Follow the red ball"
  - "Approach the target"

</td>
<td width="50%">

### 🧠 VLA Models
- **SmolVLM-256M**: Lightweight (3.3G)
- **Qwen2-VL-2B**: Balanced (4.2G)
- **Qwen3.5-9B**: Best performance (19G)

### ⚙️ Backend Options
- **Rule**: Fast, deterministic
- **Dummy VLA**: Testing & development
- **Real VLA**: Production inference

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

- Ubuntu 22.04
- NVIDIA GPU (24GB+ VRAM recommended for Qwen3.5)
- CUDA 12.x
- Conda

### Installation

```bash
# 1. Clone repository
git clone https://github.com/liangc001/VLA-Go2-Control.git
cd VLA-Go2-Control

# 2. Create conda environment
conda create -n isaac python=3.11
conda activate isaac

# 3. Install Isaac Sim
pip install isaacsim[all,extscache]==5.1.0

# 4. Install IsaacLab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
pip install -e source/isaaclab
pip install -e source/isaaclab_tasks
pip install -e "source/isaaclab_rl[all]"
pip install rsl-rl-lib
cd ..

# 5. Install VLA dependencies
pip install transformers>=4.50.0 opencv-python pillow
```

---

## 📖 Usage

### 1. Basic Playback

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --headless \
  --num_envs 1 \
  --enable_cameras \
  --checkpoint checkpoints/model_599.pt
```

### 2. Text Command Control

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --headless \
  --enable_cameras \
  --checkpoint checkpoints/model_599.pt \
  --text_command "go forward"
```

### 3. Vision-Based Control

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --headless \
  --enable_cameras \
  --checkpoint checkpoints/model_599.pt \
  --multimodal \
  --backend rule \
  --multimodal_prompt "follow the red ball" \
  --video --video_length 300
```

### 4. VLA Model Control (Qwen3.5)

```bash
CUDA_VISIBLE_DEVICES=2,1 ./isaaclab.sh \
  -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
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

---

## 🔧 Key Parameters

### VLA Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--backend` | Control backend | `rule` | `rule`, `dummy_vla`, `real_vla` |
| `--vla_model_path` | VLA model directory | - | Path to model |
| `--vla_device` | GPU for VLA | `cuda:1` | `cuda:0`, `cuda:1`, etc. |
| `--vla_action_mode` | Output format | `discrete` | `discrete`, `continuous`, `auto` |
| `--vla_num_frames` | History frames | 4 | 1-8 |
| `--vla_frame_stride` | Frame sampling | 3 | 1-10 |
| `--vla_infer_interval` | Inference frequency | 10 | Steps |

### Camera & Video

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--enable_cameras` | Enable rendering | Required |
| `--video` | Record video | - |
| `--video_length` | Video duration | 300 steps |

---

## 📊 Performance

### Model Comparison

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| SmolVLM-256M | 3.3G | ⭐⭐⭐⭐⭐ | ⭐⭐ | Prototype |
| Qwen2-VL-2B | 4.2G | ⭐⭐⭐⭐ | ⭐⭐⭐ | Balanced |
| Qwen3.5-9B | 19G | ⭐⭐ | ⭐⭐⭐⭐⭐ | Production |

### GPU Requirements

| Configuration | GPU 0 | GPU 1 | Total VRAM |
|--------------|-------|-------|------------|
| Isaac Sim Only | 8GB | - | 8GB |
| + Rule Backend | 10GB | - | 10GB |
| + SmolVLM | 10GB | 4GB | 14GB |
| + Qwen2-VL | 10GB | 6GB | 16GB |
| + Qwen3.5 | 10GB | 20GB | 30GB |

---

## 🗂️ Project Structure

```
VLA-Go2-Control/
├── 📁 scripts/
│   └── reinforcement_learning/
│       └── rsl_rl/
│           ├── 🐍 train.py                          # Training script
│           ├── 🐍 play.py                           # Basic playback
│           ├── 🐍 play_with_high_level_commands.py  # ⭐ VLA main script
│           └── 🐍 cli_args.py                       # CLI arguments
├── 📁 checkpoints/
│   └── 💾 model_599.pt                              # Pre-trained checkpoint
├── 📁 assets/
│   └── 📹 videos/                                   # Demo videos
├── 📁 docs/
│   └── 📝 qwen35_vla_status_2026-03-17.md          # Development log
├── 📄 LICENSE                                        # BSD-3-Clause
├── 📄 README.md                                      # This file
└── 📄 requirements.md                                # Dependencies
```

---

## 🐛 Troubleshooting

<details>
<summary><b>CUDA out of memory</b></summary>

```bash
# Use separate GPUs
CUDA_VISIBLE_DEVICES=2,1 ./isaaclab.sh ... \
  --device cuda:0 \
  --vla_device cuda:1
```
</details>

<details>
<summary><b>"CUDA in bad state"</b></summary>

```bash
pkill -f isaac
pkill -f python
# Or restart machine
```
</details>

<details>
<summary><b>Transformers not support qwen3_5</b></summary>

```bash
pip install --upgrade transformers
```
</details>

---

## 📚 References

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B)

---

## 📄 License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Author</b>: liangc001<br>
  <b>Date</b>: 2026-03-24<br><br>
  ⭐ Star this repo if you find it helpful!
</p>
