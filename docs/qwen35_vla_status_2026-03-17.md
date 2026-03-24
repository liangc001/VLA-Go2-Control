# Qwen3.5 VLA Status

Date: 2026-03-17

## Current Goal

Integrate a real VLM/VLA into the existing layered pipeline:

`robot front camera RGB + text prompt -> VLA -> high-level velocity (vx, vy, wz) -> low-level locomotion policy`

The current target model is:

- `models/vla/Qwen3.5-9B`

The current low-level locomotion checkpoint is:

- `logs/rsl_rl/unitree_go2_flat/2026-03-16_14-26-02/model_599.pt`

## What Is Already Confirmed

### 1. Low-level locomotion is working

The base Go2 policy can walk and execute text commands.

### 2. Rule-based vision pipeline is working

The red-ball task already works with the rule backend:

- front camera image
- target detection
- turn / forward / stop

This means the core execution chain is valid.

### 3. Input image for vision/VLA is the robot front camera

It is not the global viewer camera.

Relevant files:

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/flat_env_cfg.py`
- `scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py`

## Models On Disk

Complete:

- `models/vla/SmolVLM-256M-Instruct`
- `models/vla/Qwen2-VL-2B-Instruct`
- `models/vla/Qwen3.5-9B`

Removed because download was incomplete:

- `models/vla/Qwen2.5-VL-3B-Instruct`

## What Was Learned From Model Experiments

### SmolVLM-256M-Instruct

- Loads successfully
- Can be wired into the pipeline
- Too weak for reliable red-ball following
- Often collapses to `forward`

### Qwen2-VL-2B-Instruct

- Loads successfully
- Better than the tiny model structurally
- Still did not give reliable visual servoing for the red-ball task

### Qwen3.5-9B

Current status:

- Download is complete
- `transformers` was upgraded and now recognizes `qwen3_5`
- The model can be loaded locally as:
  - processor: `Qwen3VLProcessor`
  - model: `Qwen3_5ForConditionalGeneration`

Important result:

- The earlier failures were not only "model too weak"
- There were real pipeline/runtime issues that had to be fixed first

## Code Changes Already Made

All changes are in:

- `scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py`

### 1. Manual command override fix

When overriding velocity commands, the script now disables heading/standing overrides so `wz` is not silently rewritten.

### 2. Front-camera-only multimodal input

Vision/multimodal control now requires the robot-mounted `front_camera`.

### 3. Real VLA backend added

Backend name:

- `real_vla`

Supported args already added:

- `--backend real_vla`
- `--vla_model_path`
- `--vla_action_mode`
- `--vla_trust_remote_code`
- `--vla_device`
- `--vla_num_frames`
- `--vla_frame_stride`
- `--vla_infer_interval`

### 4. Multi-frame history support

The multimodal controller now keeps recent front-camera frames and passes a frame list to the VLA backend.

### 5. Separate GPU support for VLA

The VLA model can now be placed on a different GPU from Isaac Sim via:

- `--vla_device cuda:1`

### 6. Lower VLA inference frequency

The real VLA backend now supports sparse inference:

- run generation every `N` control steps
- reuse the last action between inference steps

This is controlled by:

- `--vla_infer_interval`

### 7. No forced video recording in multimodal mode

Multimodal and vision modes still force camera access, but they no longer force `--video`.

This reduces GPU rendering pressure.

## Problems Found And Resolved

### Resolved: old `transformers` version did not support `qwen3_5`

Symptoms:

- `ValueError: model type qwen3_5 is not recognized`

Resolution:

- upgraded `transformers`

### Resolved: backend import issue

Symptoms:

- `AutoModelForVision2Seq` import failure in some runs

Resolution:

- backend import logic now tolerates missing `AutoModelForVision2Seq`
- it prefers `AutoModelForImageTextToText`

### Resolved: VLA and Isaac on same GPU

Symptoms:

- Qwen3.5 and Isaac Sim fighting for the same GPU

Resolution:

- added `--vla_device`

## Current Blocking Issue

The main blocker is now system runtime state, not a missing code path.

Observed repeatedly:

- `Skipping NVIDIA GPU due CUDA being in bad state`
- `another kit process is locking it`
- repeated out-of-memory / bad GPU state behavior after many failed runs

This means the machine currently needs a clean restart before the next meaningful end-to-end verdict on Qwen3.5.

## Process Cleanup Already Done

Stale Qwen3.5 Isaac processes were killed before writing this summary.

## Recommended First Test After Reboot

Run this exact command first.

This is the cleanest known setup:

- hide the problematic physical GPU 0
- let Isaac use a clean GPU
- let Qwen3.5 use another GPU
- no video recording
- low-frequency VLA inference

```bash
CUDA_VISIBLE_DEVICES=2,1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --device cuda:0 \
  --num_envs 1 \
  --headless \
  --enable_cameras \
  --checkpoint logs/rsl_rl/unitree_go2_flat/2026-03-16_14-26-02/model_599.pt \
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

## What To Check In The First Post-Reboot Run

Good signs:

- model loads successfully
- no `CUDA bad state`
- no `OutOfMemory`
- step logs appear regularly
- raw outputs from `real_vla` change over time

Bad signs:

- still skipping GPUs due to bad state
- still crashing before step logs
- still only generating forever without control logs

## If The Post-Reboot Run Works

Next test:

```bash
CUDA_VISIBLE_DEVICES=2,1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --device cuda:0 \
  --num_envs 1 \
  --headless \
  --enable_cameras \
  --checkpoint logs/rsl_rl/unitree_go2_flat/2026-03-16_14-26-02/model_599.pt \
  --multimodal \
  --backend real_vla \
  --vla_model_path models/vla/Qwen3.5-9B \
  --vla_device cuda:1 \
  --vla_action_mode discrete \
  --vla_num_frames 4 \
  --vla_frame_stride 3 \
  --vla_infer_interval 10 \
  --multimodal_prompt "follow the red ball" \
  --video \
  --video_length 120
```

## Current Bottom Line

At this point:

- the pipeline is much closer to correct than before
- the main remaining blocker is unstable GPU runtime state
- a clean reboot is the next meaningful checkpoint

