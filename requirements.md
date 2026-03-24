# 环境依赖

## 基础环境
- Ubuntu 22.04
- NVIDIA GPU (推荐24GB+显存)
- CUDA 12.x
- Conda

## 安装步骤

### 1. 创建Conda环境

```bash
conda create -n isaac python=3.11
conda activate isaac
```

### 2. 安装Isaac Sim

```bash
pip install isaacsim[all,extscache]==5.1.0
```

### 3. 安装IsaacLab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

pip install -e source/isaaclab
pip install -e source/isaaclab_tasks
pip install -e "source/isaaclab_rl[all]"
```

### 4. 安装RL库

```bash
pip install rsl-rl-lib
```

### 5. 安装VLA依赖

```bash
pip install transformers>=4.50.0
pip install torch torchvision
pip install opencv-python
pip install pillow
pip install numpy
```

## 模型下载

### SmolVLM-256M-Instruct
```bash
huggingface-cli download HuggingFaceTB/SmolVLM-256M-Instruct \
  --local-dir models/vla/SmolVLM-256M-Instruct
```

### Qwen2-VL-2B-Instruct
```bash
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct \
  --local-dir models/vla/Qwen2-VL-2B-Instruct
```

### Qwen3.5-9B
```bash
huggingface-cli download Qwen/Qwen3.5-9B \
  --local-dir models/vla/Qwen3.5-9B
```

## 验证安装

```bash
./isaaclab.sh -p scripts/environments/list_envs.py | grep Go2
```

## 注意事项

1. **IsaacLab路径**: 如果使用本项目脚本，需要确保IsaacLab安装在正确位置，或者修改脚本中的路径

2. **Checkpoint位置**: 将`checkpoints/model_599.pt`复制到IsaacLab的`logs/rsl_rl/unitree_go2_flat/`目录下

3. **GPU设置**: 根据你的GPU配置调整`CUDA_VISIBLE_DEVICES`
