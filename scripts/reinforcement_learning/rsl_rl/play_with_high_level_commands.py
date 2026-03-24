# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint with high-level command control.

This script demonstrates how to override the velocity commands with a custom high-level
command sequence. It is useful for testing the robot's behavior with specific commands
or as a template for integrating with higher-level planners (e.g., VLA models).

The command sequence is hardcoded as:
- Phase 1 (0-100 steps): Move forward (vx=1.0, wz=0.0)
- Phase 2 (100-200 steps): Turn left (vx=0.5, wz=0.5)
- Phase 3 (200-300 steps): Stop (vx=0.0, wz=0.0)

NEW: Text Command Support
    This script now supports natural language text commands for controlling the robot.

    Supported text commands:
        - "go forward" / "forward" / "move forward"
        - "go backward" / "backward" / "move backward" / "back"
        - "turn left" / "left"
        - "turn right" / "right"
        - "stop"

    Usage with text commands:
        # Single text command
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
            --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 --headless \
            --text_command "go forward"

        # Text command sequence
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
            --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 --headless \
            --text_sequence "go forward|turn left|stop" --phase_steps 200

    Note: --text_command and --text_sequence take priority over --command_sequence.

NEW: Vision-Based Rule Control
    This script now supports vision-based control where the robot follows a red target.

    Vision control rules:
        - Target in left side of image -> turn left
        - Target in right side of image -> turn right
        - Target in center but far -> go forward
        - Target close/large -> stop

    Usage with vision control:
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
            --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
            --vision_control --video --video_length 500

    Note: --vision_control takes priority over all other command modes.

NEW: Multimodal Control (Image + Text)
    This script now supports unified multimodal control for VLA integration.

    The multimodal controller accepts both image and text inputs and decides
    the appropriate control strategy based on the text content:

    Control branching logic:
        - Basic motion commands (e.g., "go forward", "turn left"): Text-only branch
        - Vision-based commands (e.g., "follow the red ball"): Vision + Text branch
        - No text provided: Vision-only branch (e.g., follow any visible target)

    Usage with multimodal control:
        # Text-only mode (backward compatible)
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
            --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
            --multimodal --multimodal_prompt "go forward" --video

        # Vision + Text mode (follow target with language guidance)
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
            --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
            --multimodal --multimodal_prompt "follow the red ball" --video

        # Vision-only mode (automatic target following)
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
            --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
            --multimodal --video

    Note: --multimodal takes priority over --vision_control and --text_command.

NEW: Pluggable Backend Architecture
    This script now supports pluggable backends for the multimodal controller:

    Available backends:
        - rule: Rule-based backend using color detection and text parsing
        - dummy_vla: Dummy VLA backend (placeholder for real VLA models)

    Backend selection:
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
            --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
            --multimodal --multimodal_prompt "follow the red ball" \
            --backend rule --video

        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
            --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
            --multimodal --multimodal_prompt "go forward" \
            --backend dummy_vla --video

    To add a real VLA backend:
        1. Create a new class inheriting from BackendBase
        2. Implement predict_velocity() method
        3. Register in BACKEND_REGISTRY
        4. Use with --backend your_backend_name

Original Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_high_level_commands.py \
        --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 --headless

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import re
import sys
from collections import deque

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play with high-level command control using RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=300, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--command_sequence",
    type=str,
    default="forward,turn,stop",
    help="Command sequence: comma-separated list of commands (forward,turn,stop,backward,right).",
)
parser.add_argument(
    "--phase_steps",
    type=int,
    default=100,
    help="Number of steps for each command phase.",
)
# Text command arguments (NEW)
parser.add_argument(
    "--text_command",
    type=str,
    default=None,
    help="Single text command to execute (e.g., 'go forward', 'turn left', 'stop'). "
         "Takes priority over --command_sequence if provided.",
)
parser.add_argument(
    "--text_sequence",
    type=str,
    default=None,
    help="Sequence of text commands separated by '|' (e.g., 'go forward|turn left|stop'). "
         "Takes priority over --text_command if provided.",
)
# Vision control arguments (NEW)
parser.add_argument(
    "--vision_control",
    action="store_true",
    default=False,
    help="Enable vision-based rule control. Robot will follow a red target object. "
         "Takes priority over text commands.",
)
parser.add_argument(
    "--vision_log_interval",
    type=int,
    default=10,
    help="Number of steps between vision control logs.",
)
# Multimodal control arguments (NEW)
parser.add_argument(
    "--multimodal",
    action="store_true",
    default=False,
    help="Enable unified multimodal control (image + text). "
         "This is the recommended interface for VLA integration.",
)
parser.add_argument(
    "--multimodal_prompt",
    type=str,
    default=None,
    help="Text prompt for multimodal control (e.g., 'follow the red ball', 'go forward'). "
         "If not provided, defaults to vision-only mode.",
)
parser.add_argument(
    "--multimodal_log_interval",
    type=int,
    default=10,
    help="Number of steps between multimodal control logs.",
)
# Backend selection arguments (NEW)
parser.add_argument(
    "--backend",
    type=str,
    default="rule",
    choices=["rule", "dummy_vla", "real_vla"],
    help="Backend to use for multimodal control. "
         "'rule': Rule-based backend (color detection + text parsing). "
         "'dummy_vla': Dummy VLA backend (placeholder for real VLA models). "
         "'real_vla': Real VLA backend using a local model path. "
         "Default: 'rule'.",
)
parser.add_argument(
    "--vla_model_path",
    type=str,
    default=None,
    help="Local path to the VLA model directory/checkpoint when using --backend real_vla.",
)
parser.add_argument(
    "--vla_action_mode",
    type=str,
    default="discrete",
    choices=["discrete", "continuous", "auto"],
    help="How to decode real VLA outputs. "
         "'discrete': expect an action label. "
         "'continuous': expect JSON with vx/vy/wz. "
         "'auto': try JSON first, then discrete labels.",
)
parser.add_argument(
    "--vla_trust_remote_code",
    action="store_true",
    default=False,
    help="Pass trust_remote_code=True when loading custom VLA models.",
)
parser.add_argument(
    "--vla_device",
    type=str,
    default="cuda:1",
    help="Device used by the real VLA model. Keep Isaac Sim on cuda:0 and place the VLA on another GPU.",
)
parser.add_argument(
    "--vla_num_frames",
    type=int,
    default=4,
    help="Number of recent front-camera frames to pass to the real VLA backend.",
)
parser.add_argument(
    "--vla_frame_stride",
    type=int,
    default=3,
    help="Temporal stride when sampling the recent frame history for the real VLA backend.",
)
parser.add_argument(
    "--vla_infer_interval",
    type=int,
    default=10,
    help="Run real VLA generation once every N control steps and reuse the last action in between.",
)
parser.add_argument(
    "--vla_debug_dir",
    type=str,
    default=None,
    help="Optional directory to dump VLA debug artifacts: input frames, prompt, raw output, and parsed action.",
)
parser.add_argument(
    "--vla_debug_max_dumps",
    type=int,
    default=12,
    help="Maximum number of VLA inference events to dump to the debug directory.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Force enable cameras for vision control or multimodal. Video recording is optional.
if args_cli.vision_control or args_cli.multimodal:
    args_cli.enable_cameras = True
    mode_str = "multimodal" if args_cli.multimodal else "vision"
    print(f"[INFO] {mode_str.capitalize()} control enabled - front camera access enabled")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time
from dataclasses import MISSING
from datetime import datetime
from typing import Callable

import numpy as np
from PIL import Image

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def spawn_red_target_ball(position: tuple[float, float, float] = (3.0, 0.0, 0.5)):
    """Spawn a red ball as the visual target in the scene.

    Args:
        position: (x, y, z) position of the target ball in world coordinates.
    """
    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdGeom

    # Get the stage
    from isaaclab.sim import SimulationContext
    sim = SimulationContext.instance()
    stage = sim.stage

    # Define the path for the target
    target_path = "/World/TargetBall"

    # Check if target already exists
    if stage.GetPrimAtPath(target_path):
        print(f"[INFO] Target already exists at {target_path}")
        return target_path

    # Create a sphere (ball)
    sphere_geom = UsdGeom.Sphere.Define(stage, target_path)

    # Set radius
    sphere_geom.CreateRadiusAttr().Set(0.3)

    # Set position using Xform transform
    from pxr import Usd
    sphere_prim = stage.GetPrimAtPath(target_path)
    if sphere_prim:
        # Get or create xformable ops
        xform = UsdGeom.Xformable(sphere_prim)
        # Clear existing ops and set translate
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*position))

    # Set display color to red using displayColor primvar
    display_color_attr = sphere_geom.CreateDisplayColorAttr()
    display_color_attr.Set([Gf.Vec3f(1.0, 0.0, 0.0)])  # Red

    print(f"[INFO] Spawned red target ball at {position}")
    return target_path


def _normalize_camera_frame(frame) -> np.ndarray | None:
    """Normalize different render return types to a single RGB uint8 image."""
    if frame is None:
        return None

    # Unpack common wrapper formats
    if isinstance(frame, tuple):
        for item in frame:
            normalized = _normalize_camera_frame(item)
            if normalized is not None:
                return normalized
        return None

    if isinstance(frame, dict):
        for key in ("rgb", "image", "frame", "render", "color", "rgb_array"):
            if key in frame:
                return _normalize_camera_frame(frame[key])
        # Fall back to first value
        for value in frame.values():
            normalized = _normalize_camera_frame(value)
            if normalized is not None:
                return normalized
        return None

    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()

    if isinstance(frame, list):
        if not frame:
            return None
        return _normalize_camera_frame(frame[0])

    if not isinstance(frame, np.ndarray):
        return None

    # Batched image: (N, H, W, C) -> pick first env
    if frame.ndim == 4:
        frame = frame[0]
    # Grayscale image: (H, W) -> 3 channels
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)
    if frame.ndim != 3 or frame.shape[-1] < 3:
        return None

    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    return frame


def get_rgb_image_from_sensor(env, sensor_name: str = "front_camera") -> np.ndarray | None:
    """Read RGB output from a camera sensor attached to the environment scene."""
    if hasattr(env, "unwrapped"):
        env = env.unwrapped

    scene = getattr(env, "scene", None)
    if scene is None:
        return None

    sensors = getattr(scene, "sensors", None)
    if sensors is None or sensor_name not in sensors:
        return None

    sensor = sensors[sensor_name]
    data = getattr(sensor, "data", None)
    output = getattr(data, "output", None) if data is not None else None
    if output is None or "rgb" not in output:
        return None

    return _normalize_camera_frame(output["rgb"])


def get_camera_image_from_env(env, *, require_front_camera: bool = False) -> tuple[np.ndarray | None, str]:
    """Get an RGB image from the environment together with its source.

    Args:
        env: The environment instance.
        require_front_camera: If True, only accept the robot-mounted front camera.

    Returns:
        Tuple of (rgb_image, source_name).
    """
    # Prefer a real scene camera sensor. This is the stable path for vision control and VLA backends.
    sensor_frame = get_rgb_image_from_sensor(env, sensor_name="front_camera")
    if sensor_frame is not None:
        return sensor_frame, "robot_front_camera"

    if require_front_camera:
        return None, "missing_robot_front_camera"

    # Fall back to generic environment rendering if the sensor is unavailable.
    render_targets = [env]
    unwrapped_env = env
    while hasattr(unwrapped_env, "env") and unwrapped_env.env is not None:
        unwrapped_env = unwrapped_env.env
        render_targets.append(unwrapped_env)

    for target in render_targets:
        try:
            if hasattr(target, "render") and callable(getattr(target, "render")):
                frame = target.render()
                normalized = _normalize_camera_frame(frame)
                if normalized is not None:
                    return normalized, "render_fallback"
        except Exception:
            # Vision control is optional: try the next render target.
            continue

    return None, "unavailable"


class TextCommandParser:
    """Parser for natural language text commands to velocity commands.

    Converts text commands like "go forward", "turn left" into velocity vectors [vx, vy, wz].
    This serves as a bridge for VLA (Vision-Language-Action) models to control the robot.
    """

    # Text command mappings to internal command names
    TEXT_COMMAND_MAPPINGS = {
        # Forward commands
        "go forward": "forward",
        "move forward": "forward",
        "forward": "forward",
        "go": "forward",
        # Backward commands
        "go backward": "backward",
        "move backward": "backward",
        "backward": "backward",
        "back": "backward",
        "backwards": "backward",
        # Left turn commands
        "turn left": "spin_left",
        "left": "turn",
        "rotate left": "spin_left",
        # Right turn commands
        "turn right": "spin_right",
        "right": "turn_right",
        "rotate right": "spin_right",
        # Stop commands
        "stop": "stop",
        "halt": "stop",
        "stand still": "stop",
        "pause": "stop",
    }

    # Velocity parameters for each command (vx, vy, wz)
    COMMAND_VELOCITIES = {
        "forward": (0.8, 0.0, 0.0),
        "backward": (-0.4, 0.0, 0.0),
        "left": (0.0, 0.5, 0.0),
        "right": (0.0, -0.5, 0.0),
        # Bias turning commands toward stepping gaits instead of nearly in-place twisting.
        "turn": (0.3, 0.0, 0.8),
        "turn_right": (0.3, 0.0, -0.8),
        "spin_left": (0.3, 0.0, 0.8),
        "spin_right": (0.3, 0.0, -0.8),
        "stop": (0.0, 0.0, 0.0),
    }

    @classmethod
    def parse(cls, text_command: str) -> tuple[str, tuple[float, float, float]]:
        """Parse a text command into internal command name and velocity tuple.

        Args:
            text_command: Natural language command (e.g., "go forward", "turn left")

        Returns:
            Tuple of (internal_command_name, (vx, vy, wz))
        """
        # Normalize the input: lowercase and strip whitespace
        normalized = text_command.lower().strip()

        # Look up the command mapping
        if normalized in cls.TEXT_COMMAND_MAPPINGS:
            internal_cmd = cls.TEXT_COMMAND_MAPPINGS[normalized]
        else:
            # Try to match partial commands
            matched = False
            for key, value in cls.TEXT_COMMAND_MAPPINGS.items():
                if normalized in key or key in normalized:
                    internal_cmd = value
                    matched = True
                    print(f"[INFO] Partially matched '{text_command}' to '{key}' -> '{value}'")
                    break
            if not matched:
                print(f"[WARNING] Unknown text command: '{text_command}', using 'stop'")
                internal_cmd = "stop"

        # Get the velocity for the internal command
        velocity = cls.COMMAND_VELOCITIES.get(internal_cmd, (0.0, 0.0, 0.0))

        return internal_cmd, velocity

    @classmethod
    def parse_sequence(cls, text_sequence: str) -> list[tuple[str, tuple[float, float, float]]]:
        """Parse a sequence of text commands separated by '|'.

        Args:
            text_sequence: Commands separated by '|' (e.g., "go forward|turn left|stop")

        Returns:
            List of (command_name, velocity) tuples
        """
        commands = []
        for cmd_text in text_sequence.split("|"):
            cmd_text = cmd_text.strip()
            if cmd_text:
                internal_cmd, velocity = cls.parse(cmd_text)
                commands.append((internal_cmd, velocity))
        return commands

    @classmethod
    def get_supported_commands(cls) -> list[str]:
        """Return list of supported text command patterns."""
        return list(cls.TEXT_COMMAND_MAPPINGS.keys())


class BackendBase:
    """Base class for multimodal control backends.

    This is the abstract interface that all backends must implement.
    Backends are responsible for converting (image, text) inputs into
    velocity commands (vx, vy, wz).

    To add a new backend:
        1. Subclass BackendBase
        2. Implement predict_velocity()
        3. Register in BACKEND_REGISTRY
    """

    def __init__(self, num_envs: int, device: str, **kwargs):
        """Initialize the backend.

        Args:
            num_envs: Number of environments.
            device: Device to create tensors on.
            **kwargs: Additional backend-specific arguments.
        """
        self.num_envs = num_envs
        self.device = device
        self.backend_name = self.__class__.__name__

    def predict_velocity(
        self,
        rgb_image: np.ndarray | None = None,
        rgb_frames: list[np.ndarray] | None = None,
        text_command: str | None = None,
    ) -> tuple[float, float, float]:
        """Predict velocity command from multimodal inputs.

        This is the main interface that all backends must implement.

        Args:
            rgb_image: Optional RGB image as numpy array (H, W, 3).
            text_command: Optional text command for guidance.

        Returns:
            Tuple of (vx, vy, wz) velocity commands.
        """
        raise NotImplementedError("Subclasses must implement predict_velocity()")

    def preprocess_inputs(
        self,
        rgb_image: np.ndarray | None = None,
        rgb_frames: list[np.ndarray] | None = None,
        text_command: str | None = None,
    ) -> dict:
        """Build a minimal multimodal payload for future VLA backends."""
        if rgb_frames is None:
            rgb_frames = [rgb_image] if rgb_image is not None else []
        rgb_frames = [frame for frame in rgb_frames if frame is not None]
        return {
            "rgb": rgb_image if rgb_image is not None else (rgb_frames[-1] if rgb_frames else None),
            "rgb_frames": rgb_frames,
            "text": text_command,
        }

    def get_backend_info(self) -> dict:
        """Get backend information for logging."""
        return {
            "backend_name": self.backend_name,
            "num_envs": self.num_envs,
            "device": self.device,
        }


class RuleBasedBackend(BackendBase):
    """Rule-based backend using color detection and text parsing.

    This backend implements the original rule-based logic:
    - Text commands: Parsed using TextCommandParser
    - Vision commands: Color-based target detection
    - Multimodal: Automatic branch selection based on inputs
    """

    # Keywords that indicate vision-guided commands
    VISION_GUIDANCE_KEYWORDS = [
        "follow", "find", "go to", "approach", "track",
        "move to", "reach", "chase", "head to",
    ]

    def __init__(
        self,
        num_envs: int,
        device: str,
        image_size: tuple[int, int] = (480, 640),
        **kwargs,
    ):
        """Initialize the rule-based backend."""
        super().__init__(num_envs, device, **kwargs)
        self.image_height, self.image_width = image_size

        # Initialize vision controller for vision-based commands
        self.vision_controller = VisionRuleController(
            num_envs=num_envs,
            device=device,
            image_size=image_size,
            log_interval=kwargs.get("log_interval", 10),
        )

        # Cache for text-only commands
        self.text_cache = {}

    def predict_velocity(
        self,
        rgb_image: np.ndarray | None = None,
        rgb_frames: list[np.ndarray] | None = None,
        text_command: str | None = None,
    ) -> tuple[float, float, float]:
        """Predict velocity using rule-based logic."""
        payload = self.preprocess_inputs(rgb_image=rgb_image, rgb_frames=rgb_frames, text_command=text_command)
        rgb_image = payload["rgb"]
        text_command = payload["text"]
        has_text = text_command is not None and text_command.strip() != ""
        has_image = rgb_image is not None

        # Determine control branch
        control_mode = self._determine_control_mode(has_text, has_image, text_command)

        vx, vy, wz = 0.0, 0.0, 0.0

        if control_mode == "text_only":
            # Parse text command
            _, velocity = TextCommandParser.parse(text_command)
            vx, vy, wz = velocity

        elif control_mode in ["vision_only", "vision_text"]:
            # Use vision controller
            command = self.vision_controller.compute_command_from_vision(rgb_image)
            vx = command[0, 0].item()
            vy = command[0, 1].item()
            wz = command[0, 2].item()

        return vx, vy, wz

    def _determine_control_mode(
        self, has_text: bool, has_image: bool, text_command: str | None
    ) -> str:
        """Determine control mode based on inputs."""
        if not has_text and has_image:
            return "vision_only"
        elif has_text and not has_image:
            return "text_only"
        elif has_text and has_image:
            text_lower = text_command.lower()
            if any(keyword in text_lower for keyword in self.VISION_GUIDANCE_KEYWORDS):
                return "vision_text"
            else:
                return "text_only"
        else:
            return "text_only"


class DummyVLABackend(BackendBase):
    """Dummy VLA backend for testing and as a template for real VLA integration.

    This backend simulates a VLA model by returning predefined velocities
    based on simple heuristics. It serves as a placeholder for real VLA models.

    Future VLA models should inherit from BackendBase and implement
    predict_velocity() with actual model inference.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        default_velocity: tuple[float, float, float] = (0.5, 0.0, 0.0),
        **kwargs,
    ):
        """Initialize the dummy VLA backend.

        Args:
            num_envs: Number of environments.
            device: Device to create tensors on.
            default_velocity: Default (vx, vy, wz) to return.
        """
        super().__init__(num_envs, device, **kwargs)
        self.default_velocity = default_velocity
        self.step_count = 0

        # Simple behavior patterns for demonstration
        self.behavior_patterns = {
            "forward": (0.8, 0.0, 0.0),
            "turn_left": (0.0, 0.0, 1.0),
            "turn_right": (0.0, 0.0, -1.0),
            "stop": (0.0, 0.0, 0.0),
            "search": (0.3, 0.0, 0.6),
        }

        print(f"[DummyVLABackend] Initialized with default velocity: {default_velocity}")
        print("[DummyVLABackend] This is a placeholder backend for VLA integration")

    def predict_velocity(
        self,
        rgb_image: np.ndarray | None = None,
        rgb_frames: list[np.ndarray] | None = None,
        text_command: str | None = None,
    ) -> tuple[float, float, float]:
        """Predict velocity using dummy VLA logic.

        In a real VLA backend, this would:
        1. Preprocess the image
        2. Tokenize the text
        3. Run model inference
        4. Parse model output to velocity
        """
        payload = self.preprocess_inputs(rgb_image=rgb_image, rgb_frames=rgb_frames, text_command=text_command)
        rgb_image = payload["rgb"]
        text_command = payload["text"]
        self.step_count += 1

        # Simple text-based pattern matching (simulating VLA output)
        if text_command:
            text_lower = text_command.lower()

            # Check for specific patterns
            if any(word in text_lower for word in ["left", "左转"]):
                return self.behavior_patterns["turn_left"]
            elif any(word in text_lower for word in ["right", "右转"]):
                return self.behavior_patterns["turn_right"]
            elif any(word in text_lower for word in ["stop", "停", "站住"]):
                return self.behavior_patterns["stop"]
            elif any(word in text_lower for word in ["forward", "go", "前进", "走"]):
                return self.behavior_patterns["forward"]
            elif any(word in text_lower for word in ["search", "find", "找", "跟随"]):
                # Simulate searching behavior with periodic turning
                if self.step_count % 100 < 50:
                    return self.behavior_patterns["forward"]
                else:
                    return self.behavior_patterns["turn_left"]

        # If image is available but no text, simulate visual servoing
        if rgb_image is not None:
            # Dummy: alternate between forward and turn to simulate exploration
            if self.step_count % 200 < 150:
                return self.behavior_patterns["forward"]
            else:
                return self.behavior_patterns["turn_right"]

        # Default fallback
        return self.default_velocity


class RealVLABackend(BackendBase):
    """Real VLA backend scaffold for local multimodal models.

    This backend keeps the existing locomotion policy as the low-level controller and
    asks the VLA model to predict a high-level action or velocity command from:
    - the robot-mounted front camera RGB image
    - the user text prompt

    First version goals:
    - load a local model directory/checkpoint
    - support discrete action labels and JSON velocity outputs
    - fail safely to stop if inference is unavailable or malformed
    """

    ACTION_TO_VELOCITY = {
        "forward": (0.8, 0.0, 0.0),
        "go_forward": (0.8, 0.0, 0.0),
        "turn_left": (0.3, 0.0, 0.8),
        "left": (0.3, 0.0, 0.8),
        "turn_right": (0.3, 0.0, -0.8),
        "right": (0.3, 0.0, -0.8),
        "stop": (0.0, 0.0, 0.0),
    }

    def __init__(
        self,
        num_envs: int,
        device: str,
        model_path: str | None = None,
        model_device: str | None = None,
        action_mode: str = "discrete",
        trust_remote_code: bool = False,
        max_new_tokens: int = 32,
        num_frames: int = 4,
        frame_stride: int = 3,
        infer_interval: int = 10,
        debug_dir: str | None = None,
        debug_max_dumps: int = 12,
        **kwargs,
    ):
        super().__init__(num_envs, device, **kwargs)
        if not model_path:
            raise ValueError("--vla_model_path is required when using --backend real_vla")

        self.model_path = model_path
        self.model_device = model_device or device
        self.action_mode = action_mode
        self.trust_remote_code = trust_remote_code
        self.max_new_tokens = max_new_tokens
        self.num_frames = max(1, int(num_frames))
        self.frame_stride = max(1, int(frame_stride))
        self.infer_interval = max(1, int(infer_interval))
        self.debug_dir = debug_dir
        self.debug_max_dumps = max(0, int(debug_max_dumps))
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_type = "unloaded"
        self.load_error = None
        self.last_raw_output = ""
        self.step_count = 0
        self.last_velocity = (0.0, 0.0, 0.0)
        self.last_prompt = ""
        self.debug_dump_count = 0

        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)

        self._load_model()

    def _dump_debug_artifacts(
        self,
        sampled_frames: list[np.ndarray],
        prompt: str,
        raw_output: str,
        parsed_action: str,
        velocity: tuple[float, float, float],
    ) -> None:
        if not self.debug_dir or self.debug_dump_count >= self.debug_max_dumps:
            return

        dump_index = self.debug_dump_count
        dump_dir = os.path.join(self.debug_dir, f"infer_{dump_index:03d}_step_{self.step_count:05d}")
        os.makedirs(dump_dir, exist_ok=True)

        for frame_idx, frame in enumerate(sampled_frames):
            Image.fromarray(frame).save(os.path.join(dump_dir, f"frame_{frame_idx:02d}.png"))

        payload = {
            "step": self.step_count,
            "model_path": self.model_path,
            "model_type": self.model_type,
            "model_device": self.model_device,
            "prompt": prompt,
            "raw_output": raw_output,
            "parsed_action": parsed_action,
            "velocity": {"vx": velocity[0], "vy": velocity[1], "wz": velocity[2]},
            "num_frames": len(sampled_frames),
        }
        with open(os.path.join(dump_dir, "debug.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self.debug_dump_count += 1

    def _load_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
        except Exception as exc:
            self.load_error = f"transformers import failed: {exc}"
            print(f"[RealVLABackend] {self.load_error}")
            return

        try:
            from transformers import AutoModelForVision2Seq
        except Exception:
            AutoModelForVision2Seq = None

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
            )
        except Exception:
            self.processor = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
            )
        except Exception:
            self.tokenizer = None

        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": torch.float16 if "cuda" in str(self.model_device) else torch.float32,
        }

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, **model_kwargs)
            self.model_type = "image_text_to_text"
        except Exception:
            try:
                if AutoModelForVision2Seq is None:
                    raise RuntimeError("AutoModelForVision2Seq is unavailable in this transformers build")
                self.model = AutoModelForVision2Seq.from_pretrained(self.model_path, **model_kwargs)
                self.model_type = "vision2seq"
            except Exception:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
                    self.model_type = "causal_lm"
                except Exception as exc:
                    self.load_error = f"model load failed: {exc}"
                    print(f"[RealVLABackend] {self.load_error}")
                    self.model = None
                    return

        self.model.to(self.model_device)
        self.model.eval()
        print(f"[RealVLABackend] Loaded model from: {self.model_path}")
        print(f"[RealVLABackend] Model type: {self.model_type}")
        print(f"[RealVLABackend] Action mode: {self.action_mode}")
        print(f"[RealVLABackend] Model device: {self.model_device}")

    def _build_prompt(self, text_command: str | None, num_frames: int) -> str:
        text_command = (text_command or "move safely").strip()
        text_lower = text_command.lower()
        if self.action_mode == "continuous":
            return (
                "You control a quadruped robot. "
                f"You are given {num_frames} recent front-camera frames ordered from oldest to newest. "
                "Use the frame history and the text instruction. "
                "Reply with JSON only in the format "
                '{"vx": float, "vy": float, "wz": float}. '
                f"Instruction: {text_command}"
            )

        if "follow the red ball" in text_lower or "follow red ball" in text_lower or "red ball" in text_lower:
            return (
                "You control a quadruped robot from recent front-camera frames. "
                f"You are given {num_frames} recent robot-mounted front camera frames ordered from oldest to newest. "
                "Your task is to follow the red ball. "
                "Choose exactly one action label and output only that label. "
                "Allowed labels: forward, turn_left, turn_right, stop. "
                "Decision rules: "
                "If the red ball is mostly on the left side of the image, output turn_left. "
                "If the red ball is mostly on the right side of the image, output turn_right. "
                "If the red ball is centered and still far away, output forward. "
                "If the red ball is very close, large in the image, or already reached, output stop. "
                "If the red ball is not visible, output stop. "
                "Do not explain your reasoning. "
                "Do not repeat the instruction. "
                "Output one label only."
            )

        return (
            "You control a quadruped robot. "
            f"You are given {num_frames} recent robot-mounted front camera frames ordered from oldest to newest. "
            "Use the frame history and the text instruction. "
            "Reply with exactly one label and nothing else. "
            "Allowed labels: forward, turn_left, turn_right, stop. "
            "Examples: "
            "instruction='turn left' -> turn_left; "
            "instruction='turn right' -> turn_right; "
            "instruction='go forward' -> forward; "
            "instruction='stop' -> stop. "
            f"Instruction: {text_command}"
        )

    def _build_chat_messages(self, prompt: str, num_frames: int) -> list[dict]:
        content = []
        for index in range(num_frames):
            content.append({"type": "image"})
            content.append({"type": "text", "text": f"Frame {index + 1}."})
        content.append({"type": "text", "text": prompt})
        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def _run_generation(self, rgb_frames: list[np.ndarray], prompt: str) -> str:
        if self.model is None:
            raise RuntimeError(self.load_error or "VLA model is not loaded")

        generation_kwargs = {
            "max_new_tokens": min(self.max_new_tokens, 8 if self.action_mode != "continuous" else self.max_new_tokens),
            "do_sample": False,
        }

        pil_images = [Image.fromarray(frame) for frame in rgb_frames]
        image_payload = pil_images[0] if len(pil_images) == 1 else pil_images

        if self.processor is not None:
            try:
                if hasattr(self.processor, "apply_chat_template") and pil_images:
                    messages = self._build_chat_messages(prompt, len(pil_images))
                    formatted_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = self.processor(text=formatted_prompt, images=pil_images, return_tensors="pt")
                elif pil_images:
                    inputs = self.processor(images=image_payload, text=prompt, return_tensors="pt")
                else:
                    inputs = self.processor(text=prompt, return_tensors="pt")
            except Exception:
                if pil_images:
                    inputs = self.processor(images=image_payload, text=prompt, return_tensors="pt")
                else:
                    inputs = self.processor(text=prompt, return_tensors="pt")
        elif self.tokenizer is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        else:
            raise RuntimeError("No processor/tokenizer available for VLA backend")

        inputs = {key: value.to(self.model_device) if hasattr(value, "to") else value for key, value in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        prompt_ids = inputs.get("input_ids")
        if prompt_ids is not None and output_ids.shape[-1] > prompt_ids.shape[-1]:
            generated_ids = output_ids[:, prompt_ids.shape[-1]:]
        else:
            generated_ids = output_ids

        decode_source = self.processor if self.processor is not None else self.tokenizer
        if decode_source is None:
            raise RuntimeError("No decode source available for VLA backend")
        return decode_source.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def _parse_json_velocity(self, text: str) -> tuple[float, float, float] | None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return None

        if not all(key in payload for key in ("vx", "vy", "wz")):
            return None
        return self._sanitize_velocity(payload["vx"], payload["vy"], payload["wz"])

    def _normalize_action_text(self, text: str) -> str:
        normalized = text.lower().strip()
        normalized = normalized.replace("-", "_").replace(" ", "_")
        normalized = re.sub(r"[^a-z_]", "", normalized)
        return normalized

    def _extract_action_from_free_text(self, text: str, text_command: str | None = None) -> str | None:
        """Extract an action label from explanatory model output.

        The VLM often responds with natural language such as:
        - "The user wants me to control a quadruped to turn left..."
        - "I should move forward toward the red ball."
        - "Stop because the target is close."

        This method looks for action-bearing phrases in the raw text and maps them
        back to one of the allowed discrete control labels.
        """
        text_lower = (text or "").lower()
        if not text_lower:
            return None

        phrase_groups = [
            ("turn_left", [r"\bturn\s+left\b", r"\bleft\b", r"\brotate\s+left\b", r"\bveer\s+left\b"]),
            ("turn_right", [r"\bturn\s+right\b", r"\bright\b", r"\brotate\s+right\b", r"\bveer\s+right\b"]),
            ("stop", [r"\bstop\b", r"\bhalt\b", r"\bstand\s+still\b", r"\bdo\s+not\s+move\b", r"\bfreeze\b"]),
            ("forward", [r"\bgo\s+forward\b", r"\bmove\s+forward\b", r"\bforward\b", r"\bgo\s+straight\b", r"\bapproach\b"]),
        ]

        matches: list[tuple[int, str]] = []
        for action, patterns in phrase_groups:
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match is not None:
                    matches.append((match.start(), action))
                    break

        if not matches:
            return None

        instruction_prior = self._instruction_prior(text_command)
        if instruction_prior is not None:
            for _, action in matches:
                if action == instruction_prior:
                    return action

        matches.sort(key=lambda item: item[0])
        return matches[0][1]

    def _instruction_prior(self, text_command: str | None) -> str | None:
        if not text_command:
            return None
        text = text_command.lower()
        if "turn left" in text or text.strip() == "left":
            return "turn_left"
        if "turn right" in text or text.strip() == "right":
            return "turn_right"
        if "stop" in text:
            return "stop"
        if "forward" in text or "go " in text:
            return "forward"
        return None

    def _is_simple_text_command(self, text_command: str | None) -> bool:
        if not text_command:
            return False
        text = text_command.lower().strip()
        complex_keywords = [
            "follow",
            "find",
            "track",
            "red ball",
            "target",
            "object",
            "approach",
            "go to",
            "move to",
            "reach",
            "chase",
        ]
        if any(keyword in text for keyword in complex_keywords):
            return False
        return self._instruction_prior(text_command) is not None

    def _parse_discrete_action(self, text: str, text_command: str | None = None) -> tuple[float, float, float] | None:
        normalized = self._normalize_action_text(text)
        instruction_prior = self._instruction_prior(text_command)

        direct_aliases = {
            "forward": "forward",
            "goforward": "forward",
            "go_forward": "forward",
            "moveforward": "forward",
            "turnleft": "turn_left",
            "turn_left": "turn_left",
            "left": "turn_left",
            "turnright": "turn_right",
            "turn_right": "turn_right",
            "right": "turn_right",
            "stop": "stop",
        }

        if normalized in direct_aliases:
            return self.ACTION_TO_VELOCITY[direct_aliases[normalized]]

        if instruction_prior is not None and instruction_prior in normalized:
            return self.ACTION_TO_VELOCITY[instruction_prior]

        for alias, canonical in direct_aliases.items():
            if alias in normalized:
                if instruction_prior in ("turn_left", "turn_right") and canonical == "forward":
                    continue
                return self.ACTION_TO_VELOCITY[canonical]

        free_text_action = self._extract_action_from_free_text(text, text_command=text_command)
        if free_text_action is not None:
            return self.ACTION_TO_VELOCITY[free_text_action]
        return None

    def _label_from_velocity(self, velocity: tuple[float, float, float] | None) -> str:
        if velocity is None:
            return "none"
        for label, candidate in self.ACTION_TO_VELOCITY.items():
            if tuple(candidate) == tuple(velocity):
                return label
        return "custom"

    def _fallback_from_text_command(self, text_command: str | None) -> tuple[float, float, float] | None:
        instruction_prior = self._instruction_prior(text_command)
        if instruction_prior is None:
            return None
        return self.ACTION_TO_VELOCITY[instruction_prior]

    def _sanitize_velocity(self, vx, vy, wz) -> tuple[float, float, float]:
        vx = float(np.clip(vx, -1.0, 1.0))
        vy = float(np.clip(vy, -0.5, 0.5))
        wz = float(np.clip(wz, -1.0, 1.0))
        if not np.isfinite(vx) or not np.isfinite(vy) or not np.isfinite(wz):
            return 0.0, 0.0, 0.0
        return vx, vy, wz

    def predict_velocity(
        self,
        rgb_image: np.ndarray | None = None,
        rgb_frames: list[np.ndarray] | None = None,
        text_command: str | None = None,
    ) -> tuple[float, float, float]:
        self.step_count += 1
        payload = self.preprocess_inputs(rgb_image=rgb_image, rgb_frames=rgb_frames, text_command=text_command)
        rgb_image = payload["rgb"]
        rgb_frames = payload["rgb_frames"]
        text_command = payload["text"]

        # Keep simple motion commands deterministic. Real VLA is used for richer visual/language tasks.
        if self._is_simple_text_command(text_command):
            fallback = self._fallback_from_text_command(text_command)
            if fallback is not None:
                self.last_raw_output = f"rule_fallback:{self._instruction_prior(text_command)}"
                self.last_velocity = fallback
                return fallback

        if self.step_count > 1 and (self.step_count - 1) % self.infer_interval != 0:
            self.last_raw_output = f"reuse_last_action:{self.last_raw_output or 'none'}"
            return self.last_velocity

        if not rgb_frames:
            print("[RealVLABackend] Missing front camera image, returning stop")
            self.last_velocity = (0.0, 0.0, 0.0)
            return 0.0, 0.0, 0.0

        try:
            sampled_frames = rgb_frames[-self.num_frames * self.frame_stride::self.frame_stride]
            if not sampled_frames:
                sampled_frames = [rgb_image]
            prompt = self._build_prompt(text_command, len(sampled_frames))
            self.last_prompt = prompt
            raw_output = self._run_generation(sampled_frames, prompt)
            self.last_raw_output = raw_output
        except Exception as exc:
            print(f"[RealVLABackend] Inference failed: {exc}")
            return 0.0, 0.0, 0.0

        if self.action_mode == "continuous":
            velocity = self._parse_json_velocity(raw_output)
            self.last_velocity = velocity if velocity is not None else (0.0, 0.0, 0.0)
            self._dump_debug_artifacts(sampled_frames, self.last_prompt, raw_output, self._label_from_velocity(self.last_velocity), self.last_velocity)
            return self.last_velocity

        if self.action_mode == "discrete":
            velocity = self._parse_discrete_action(raw_output, text_command=text_command)
            if velocity is not None:
                self.last_velocity = velocity
                self._dump_debug_artifacts(sampled_frames, self.last_prompt, raw_output, self._label_from_velocity(self.last_velocity), self.last_velocity)
                return velocity
            fallback = self._fallback_from_text_command(text_command)
            self.last_velocity = fallback if fallback is not None else (0.0, 0.0, 0.0)
            self._dump_debug_artifacts(sampled_frames, self.last_prompt, raw_output, self._label_from_velocity(self.last_velocity), self.last_velocity)
            return self.last_velocity

        velocity = self._parse_json_velocity(raw_output)
        if velocity is not None:
            self.last_velocity = velocity
            self._dump_debug_artifacts(sampled_frames, self.last_prompt, raw_output, self._label_from_velocity(self.last_velocity), self.last_velocity)
            return velocity
        velocity = self._parse_discrete_action(raw_output, text_command=text_command)
        if velocity is not None:
            self.last_velocity = velocity
            self._dump_debug_artifacts(sampled_frames, self.last_prompt, raw_output, self._label_from_velocity(self.last_velocity), self.last_velocity)
            return velocity
        fallback = self._fallback_from_text_command(text_command)
        self.last_velocity = fallback if fallback is not None else (0.0, 0.0, 0.0)
        self._dump_debug_artifacts(sampled_frames, self.last_prompt, raw_output, self._label_from_velocity(self.last_velocity), self.last_velocity)
        return self.last_velocity

    def get_backend_info(self) -> dict:
        info = super().get_backend_info()
        info.update(
            {
                "model_path": self.model_path,
                "model_device": self.model_device,
                "model_type": self.model_type,
                "action_mode": self.action_mode,
                "num_frames": self.num_frames,
                "frame_stride": self.frame_stride,
                "infer_interval": self.infer_interval,
                "debug_dir": self.debug_dir or "none",
                "load_error": self.load_error or "none",
                "last_raw_output": self.last_raw_output or "none",
            }
        )
        return info


# Backend registry for command-line selection
BACKEND_REGISTRY = {
    "rule": RuleBasedBackend,
    "dummy_vla": DummyVLABackend,
    "real_vla": RealVLABackend,
}


def create_backend(
    backend_name: str,
    num_envs: int,
    device: str,
    **kwargs,
) -> BackendBase:
    """Factory function to create a backend instance.

    Args:
        backend_name: Name of the backend to create.
        num_envs: Number of environments.
        device: Device to create tensors on.
        **kwargs: Additional backend-specific arguments.

    Returns:
        Backend instance.

    Raises:
        ValueError: If backend_name is not recognized.
    """
    if backend_name not in BACKEND_REGISTRY:
        available = ", ".join(BACKEND_REGISTRY.keys())
        raise ValueError(f"Unknown backend: {backend_name}. Available: {available}")

    backend_class = BACKEND_REGISTRY[backend_name]
    return backend_class(num_envs=num_envs, device=device, **kwargs)


class MultimodalCommandController:
    """Unified multimodal controller using pluggable backends.

    This controller provides a single interface for VLA integration by delegating
    all prediction logic to a pluggable backend. Backends can be:
    - RuleBasedBackend: Original rule-based logic
    - DummyVLABackend: Placeholder for VLA models
    - Future: RealVLABackend with actual model inference

    Usage:
        controller = MultimodalCommandController(
            num_envs=1,
            device="cuda:0",
            backend_name="rule",  # or "dummy_vla"
        )
        velocity = controller.compute_high_level_command(image, text)
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        backend_name: str = "rule",
        text_prompt: str | None = None,
        image_size: tuple[int, int] = (480, 640),
        log_interval: int = 10,
        num_frames: int = 4,
        frame_stride: int = 3,
        **backend_kwargs,
    ):
        """Initialize the multimodal command controller.

        Args:
            num_envs: Number of environments.
            device: Device to create tensors on.
            backend_name: Name of the backend to use ("rule" or "dummy_vla").
            text_prompt: Optional initial text prompt.
            image_size: (height, width) of the camera image.
            log_interval: Number of steps between logs.
            **backend_kwargs: Additional arguments for backend initialization.
        """
        self.num_envs = num_envs
        self.device = device
        self.backend_name = backend_name
        self.text_prompt = text_prompt
        self.image_height, self.image_width = image_size
        self.log_interval = log_interval
        self.step_count = 0
        self.num_frames = max(1, int(num_frames))
        self.frame_stride = max(1, int(frame_stride))
        self.frame_history = deque(maxlen=self.num_frames * self.frame_stride)

        # Create the backend
        try:
            self.backend = create_backend(
                backend_name=backend_name,
                num_envs=num_envs,
                device=device,
                image_size=image_size,
                log_interval=log_interval,
                num_frames=self.num_frames,
                frame_stride=self.frame_stride,
                **backend_kwargs,
            )
        except ValueError as e:
            print(f"[ERROR] Failed to create backend: {e}")
            raise

        # Create command tensor
        self.vel_command_b = torch.zeros(num_envs, 3, device=device)

        # Status tracking
        self.last_decision = {
            "backend": backend_name,
            "text_prompt": text_prompt,
            "velocity": (0.0, 0.0, 0.0),
        }

        self._print_controller_summary()

    def _print_controller_summary(self):
        """Print controller initialization summary."""
        print("-" * 60)
        print("Multimodal Command Controller Initialized")
        print("-" * 60)
        print(f"Backend: {self.backend_name}")
        print(f"Backend Class: {self.backend.__class__.__name__}")
        print(f"Text Prompt: {self.text_prompt or 'None'}")
        print(f"Image Size: ({self.image_height}, {self.image_width})")
        print(f"Log Interval: {self.log_interval}")
        print("")
        print("Backend Info:")
        info = self.backend.get_backend_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        print("-" * 60)

    def compute_high_level_command(
        self,
        rgb_image: np.ndarray | None = None,
        text_command: str | None = None,
    ) -> torch.Tensor:
        """Compute velocity command from multimodal inputs.

        This is the unified interface for VLA integration.

        Args:
            rgb_image: Optional RGB image as numpy array (H, W, 3).
            text_command: Optional text command for guidance.

        Returns:
            Velocity command tensor of shape (num_envs, 3).
        """
        self.step_count += 1

        # Update text prompt if provided
        if text_command is not None:
            self.text_prompt = text_command

        if rgb_image is not None:
            self.frame_history.append(np.array(rgb_image, copy=True))

        rgb_frames = list(self.frame_history)[::self.frame_stride]
        if not rgb_frames and rgb_image is not None:
            rgb_frames = [rgb_image]

        # Delegate to backend for velocity prediction
        vx, vy, wz = self.backend.predict_velocity(rgb_image, rgb_frames, self.text_prompt)

        # Set command for all environments
        self.vel_command_b[:, 0] = vx
        self.vel_command_b[:, 1] = vy
        self.vel_command_b[:, 2] = wz

        # Update status
        self.last_decision = {
            "backend": self.backend_name,
            "text_prompt": self.text_prompt,
            "velocity": (vx, vy, wz),
            "raw_output": getattr(self.backend, "last_raw_output", ""),
            "num_frames": len(rgb_frames),
        }

        return self.vel_command_b.clone()

    def get_current_command_name(self, step: int) -> str:
        """Get the current command name for logging."""
        return f"{self.backend_name}_step{step}"

    def log_status(self, step: int):
        """Print current multimodal control status."""
        if step % self.log_interval == 0:
            info = self.last_decision
            backend = info.get("backend", "unknown")
            velocity = info.get("velocity", (0.0, 0.0, 0.0))
            vx, vy, wz = velocity
            text = info.get("text_prompt", "None") or "None"
            raw_output = info.get("raw_output", "") or ""
            num_frames = info.get("num_frames", 0)
            # Truncate text if too long
            text_short = text[:30] + "..." if len(text) > 30 else text
            raw_short = raw_output[:40] + "..." if len(raw_output) > 40 else raw_output

            print(f"[Step {step:4d}] Backend: {backend:12s} | "
                  f"Text: '{text_short:20s}' | "
                  f"frames={num_frames:2d} | "
                  f"vx={vx:+.2f}, vy={vy:+.2f}, wz={wz:+.2f}"
                  + (f" | raw='{raw_short}'" if raw_short else ""))


class VisionRuleController:
    """Rule-based vision controller for following a colored target.

    This controller processes camera images to detect a target object (e.g., red ball)
    and generates velocity commands based on the target's position in the image.

    Control rules:
        - Target in left 1/3 of image -> turn left (wz > 0)
        - Target in right 1/3 of image -> turn right (wz < 0)
        - Target in center 1/3 -> go forward (vx > 0)
        - Target is large/close -> stop (vx = 0, wz = 0)

    This serves as a minimal example for VLA integration where vision inputs
    are processed to generate high-level velocity commands.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        target_color_range: dict | None = None,
        image_size: tuple[int, int] = (480, 640),
        log_interval: int = 10,
    ):
        """Initialize the vision rule controller.

        Args:
            num_envs: Number of environments.
            device: Device to create tensors on.
            target_color_range: HSV color range for target detection.
                Defaults to detecting red objects.
            image_size: (height, width) of the camera image.
            log_interval: Number of steps between logs.
        """
        self.num_envs = num_envs
        self.device = device
        self.image_height, self.image_width = image_size
        self.log_interval = log_interval
        self.step_count = 0
        self.image_source = "unknown"

        # Default color range for red objects in HSV
        # Red wraps around in HSV, so we need two ranges
        if target_color_range is None:
            self.target_color_range = {
                "lower1": np.array([0, 100, 100]),
                "upper1": np.array([10, 255, 255]),
                "lower2": np.array([160, 100, 100]),
                "upper2": np.array([180, 255, 255]),
            }
        else:
            self.target_color_range = target_color_range

        # Control parameters
        self.forward_vel = 0.8
        self.min_forward_vel = 0.18
        self.max_turn_vel = 0.9
        self.search_turn_vel = 0.35
        self.steer_kp = 0.9
        self.center_tolerance = 0.12
        self.stop_area_threshold = 0.15
        self.slow_area_threshold = 0.08
        self.min_detection_pixels = 50
        self.lost_target_patience = 12
        self.last_seen_direction = 1.0
        self.steps_since_seen = 0

        # Create command tensor
        self.vel_command_b = torch.zeros(num_envs, 3, device=device)

        # Debug info storage
        self.last_detection_info = {
            "target_found": False,
            "target_center": None,
            "target_area_ratio": 0.0,
            "decision": "stop",
            "image_source": "unknown",
        }

        print("-" * 60)
        print("Vision Rule Controller Initialized")
        print("-" * 60)
        print(f"Image size: {image_size}")
        print(f"Forward velocity: {self.forward_vel}")
        print(f"Max turn velocity: {self.max_turn_vel}")
        print(f"Search turn velocity: {self.search_turn_vel}")
        print(f"Stop threshold (area ratio): {self.stop_area_threshold}")
        print(f"Slow approach threshold: {self.slow_area_threshold}")
        print("Image source: robot-mounted front camera only")
        print("Control regions:")
        print(f"  Left  : x < {self.image_width // 3}")
        print(f"  Center: {self.image_width // 3} <= x < {2 * self.image_width // 3}")
        print(f"  Right : x >= {2 * self.image_width // 3}")
        print("-" * 60)

    def detect_target(self, rgb_image: np.ndarray) -> tuple[bool, tuple[int, int] | None, float]:
        """Detect the target object in the RGB image using simple color thresholding.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3) with values in [0, 255].

        Returns:
            Tuple of (target_found, target_center, target_area_ratio).
            target_center is (x, y) in pixel coordinates.
        """
        # Simple red detection in RGB space
        # Red has high R channel and low G, B channels
        r = rgb_image[:, :, 0].astype(np.float32)
        g = rgb_image[:, :, 1].astype(np.float32)
        b = rgb_image[:, :, 2].astype(np.float32)

        # Create mask: R > 150 and R > G + 30 and R > B + 30
        red_mask = (r > 150) & (r > g + 30) & (r > b + 30)

        # Find connected components using simple clustering
        # Get coordinates of red pixels
        red_coords = np.argwhere(red_mask)

        if len(red_coords) < self.min_detection_pixels:
            return False, None, 0.0

        # Calculate centroid
        center_y = int(np.mean(red_coords[:, 0]))
        center_x = int(np.mean(red_coords[:, 1]))

        # Calculate area ratio with actual frame shape.
        img_height, img_width = rgb_image.shape[:2]
        total_area = img_width * img_height
        area_ratio = len(red_coords) / total_area

        return True, (center_x, center_y), area_ratio

    def compute_command_from_vision(self, rgb_image: np.ndarray | None, image_source: str = "unknown") -> torch.Tensor:
        """Compute velocity command based on vision input.

        Args:
            rgb_image: RGB image as numpy array, or None if not available.
            image_source: Human-readable image source identifier.

        Returns:
            Velocity command tensor of shape (num_envs, 3).
        """
        self.step_count += 1

        # Default: stop
        vx, vy, wz = 0.0, 0.0, 0.0
        decision = "stop"
        self.image_source = image_source

        if rgb_image is not None:
            # Adapt to the actual render resolution.
            self.image_height, self.image_width = rgb_image.shape[:2]
            # Detect target
            target_found, target_center, area_ratio = self.detect_target(rgb_image)

            self.last_detection_info = {
                "target_found": target_found,
                "target_center": target_center,
                "target_area_ratio": area_ratio,
                "decision": decision,
                "image_source": image_source,
            }

            if target_found:
                self.steps_since_seen = 0
                center_x, center_y = target_center
                image_mid_x = self.image_width / 2.0
                normalized_error_x = (center_x - image_mid_x) / max(image_mid_x, 1.0)
                self.last_seen_direction = -1.0 if normalized_error_x < 0.0 else 1.0

                # Check if target is close enough to stop
                if area_ratio > self.stop_area_threshold:
                    decision = "stop (target close)"
                    vx, vy, wz = 0.0, 0.0, 0.0
                else:
                    wz = float(np.clip(-self.steer_kp * normalized_error_x, -self.max_turn_vel, self.max_turn_vel))
                    abs_error = abs(normalized_error_x)

                    if area_ratio > self.slow_area_threshold:
                        target_forward_vel = self.forward_vel * 0.45
                    else:
                        target_forward_vel = self.forward_vel

                    if abs_error <= self.center_tolerance:
                        decision = "go_forward"
                        vx = target_forward_vel
                    elif abs_error <= 0.45:
                        decision = "track_target"
                        vx = max(self.min_forward_vel, target_forward_vel * (1.0 - abs_error))
                    else:
                        decision = "turn_left" if normalized_error_x < 0.0 else "turn_right"
                        vx = self.min_forward_vel

                self.last_detection_info["decision"] = decision
            else:
                self.steps_since_seen += 1

        if not self.last_detection_info["target_found"] and self.steps_since_seen <= self.lost_target_patience:
            decision = "search_left" if self.last_seen_direction < 0.0 else "search_right"
            vx = 0.0
            wz = self.search_turn_vel * self.last_seen_direction
            self.last_detection_info["decision"] = decision
            self.last_detection_info["image_source"] = image_source

        # Set command for all environments
        self.vel_command_b[:, 0] = vx
        self.vel_command_b[:, 1] = vy
        self.vel_command_b[:, 2] = wz

        return self.vel_command_b.clone()

    def get_current_command_name(self, step: int) -> str:
        """Get the current command name for logging."""
        return self.last_detection_info.get("decision", "unknown")

    def log_status(self, step: int):
        """Print current vision status."""
        if step % self.log_interval == 0:
            info = self.last_detection_info
            found = "YES" if info["target_found"] else "NO"
            center = info["target_center"]
            center_str = f"({center[0]}, {center[1]})" if center else "N/A"
            area = info["target_area_ratio"]
            decision = info["decision"]
            source = info.get("image_source", "unknown")

            print(f"[Vision Step {step:4d}] Source: {source:24s} | Target: {found} | "
                  f"Center: {center_str:15s} | Area: {area:.3f} | Decision: {decision}")


class HighLevelCommandController:
    """High-level command controller that generates velocity commands.

    This controller generates a sequence of velocity commands (vx, vy, wz) based on
    predefined high-level commands like "forward", "turn", "stop", etc.

    The commands are in the robot's base frame:
    - vx: linear velocity in x direction (forward/backward)
    - vy: linear velocity in y direction (left/right)
    - wz: angular velocity around z axis (yaw)
    """

    # Command definitions: (vx, vy, wz)
    COMMAND_VELOCITIES = {
        "forward": (0.8, 0.0, 0.0),
        "backward": (-0.4, 0.0, 0.0),
        "left": (0.0, 0.5, 0.0),
        "right": (0.0, -0.5, 0.0),
        "turn": (0.3, 0.0, 0.8),
        "turn_right": (0.3, 0.0, -0.8),
        "spin_left": (0.3, 0.0, 0.8),
        "spin_right": (0.3, 0.0, -0.8),
        "stop": (0.0, 0.0, 0.0),
    }

    def __init__(
        self,
        num_envs: int,
        device: str,
        command_sequence: list[str],
        phase_steps: int,
        command_details: list[tuple[str, tuple[float, float, float]]] | None = None
    ):
        """Initialize the high-level command controller.

        Args:
            num_envs: Number of environments.
            device: Device to create tensors on.
            command_sequence: List of high-level command names.
            phase_steps: Number of steps for each command phase.
            command_details: Optional list of (command_name, velocity) tuples for detailed logging.
        """
        self.num_envs = num_envs
        self.device = device
        self.command_sequence = command_sequence
        self.phase_steps = phase_steps
        self.current_step = 0
        self.command_details = command_details  # Store detailed command info

        # Create command tensor: (num_envs, 3) for [vx, vy, wz]
        self.vel_command_b = torch.zeros(num_envs, 3, device=device)

        # Print command summary on initialization
        self._print_command_summary()

    def _print_command_summary(self):
        """Print a summary of the command sequence."""
        print("-" * 60)
        print("High-Level Command Controller Initialized")
        print("-" * 60)
        print(f"Total phases: {len(self.command_sequence)}")
        print(f"Phase duration: {self.phase_steps} steps")
        print(f"Total sequence duration: {len(self.command_sequence) * self.phase_steps} steps")
        print("")
        print("Command Sequence:")
        for i, (cmd_name, details) in enumerate(zip(self.command_sequence, self._get_command_details())):
            start_step = i * self.phase_steps
            end_step = (i + 1) * self.phase_steps
            _, velocity = details
            vx, vy, wz = velocity
            print(f"  Phase {i+1} (steps {start_step:4d}-{end_step:4d}): "
                  f"'{cmd_name}' -> vx={vx:+.2f}, vy={vy:+.2f}, wz={wz:+.2f}")
        print("-" * 60)

    def _get_command_details(self) -> list[tuple[str, tuple[float, float, float]]]:
        """Get command details, either from stored details or from command sequence."""
        if self.command_details is not None:
            return self.command_details
        # Otherwise, construct from command sequence and default velocities
        details = []
        for cmd in self.command_sequence:
            velocity = self.COMMAND_VELOCITIES.get(cmd, (0.0, 0.0, 0.0))
            details.append((cmd, velocity))
        return details

    def compute_command(self, step: int) -> torch.Tensor:
        """Compute the velocity command for the current step.

        Args:
            step: Current simulation step.

        Returns:
            Velocity command tensor of shape (num_envs, 3).
        """
        # Determine which phase we're in
        phase_idx = (step // self.phase_steps) % len(self.command_sequence)
        command_name = self.command_sequence[phase_idx]

        # Get the velocity for this command
        if command_name in self.COMMAND_VELOCITIES:
            vx, vy, wz = self.COMMAND_VELOCITIES[command_name]
        else:
            print(f"[WARNING] Unknown command: {command_name}, using stop")
            vx, vy, wz = self.COMMAND_VELOCITIES["stop"]

        # Set the command for all environments
        self.vel_command_b[:, 0] = vx  # vx
        self.vel_command_b[:, 1] = vy  # vy
        self.vel_command_b[:, 2] = wz  # wz

        return self.vel_command_b.clone()

    def get_current_command_name(self, step: int) -> str:
        """Get the name of the current command for logging."""
        phase_idx = (step // self.phase_steps) % len(self.command_sequence)
        return self.command_sequence[phase_idx]


def override_velocity_command(env, command_tensor: torch.Tensor):
    """Override the velocity command in the environment's command manager.

    Args:
        env: The environment instance.
        command_tensor: The velocity command tensor of shape (num_envs, 3) with [vx, vy, wz].
    """
    # Access the base_velocity command term
    # The command manager is in env.unwrapped.command_manager for wrapped environments
    if hasattr(env, "unwrapped"):
        env = env.unwrapped

    # Get the base_velocity command term
    if hasattr(env, "command_manager"):
        command_manager = env.command_manager
        if "base_velocity" in command_manager.active_terms:
            # Get the velocity command term
            vel_cmd_term = command_manager.get_term("base_velocity")
            # Disable command-generator post-processing that can overwrite manual yaw commands.
            if hasattr(vel_cmd_term, "is_heading_env"):
                vel_cmd_term.is_heading_env[:] = False
            if hasattr(vel_cmd_term, "is_standing_env"):
                vel_cmd_term.is_standing_env[:] = False
            # Override the velocity command
            # vel_command_b is the buffer used by UniformVelocityCommand
            # Ensure command_tensor is on the same device as vel_command_b
            if vel_cmd_term.vel_command_b.device != command_tensor.device:
                command_tensor = command_tensor.to(vel_cmd_term.vel_command_b.device)
            vel_cmd_term.vel_command_b[:] = command_tensor
        else:
            print("[WARNING] 'base_velocity' not found in command manager")
    else:
        print("[WARNING] Command manager not found in environment")


def parse_command_sequence_with_text_support(args_cli) -> tuple[list[str], list[tuple[str, tuple[float, float, float]]], VisionRuleController | None, MultimodalCommandController | None]:
    """Parse command sequence with support for text commands and vision control.

    Priority:
        1. --multimodal (highest)
        2. --vision_control
        3. --text_sequence
        4. --text_command
        5. --command_sequence (lowest)

    Args:
        args_cli: Parsed command line arguments.

    Returns:
        Tuple of (command_name_list, command_details_list, vision_controller, multimodal_controller)
        vision_controller and multimodal_controller are None if not enabled.
    """
    print("\n" + "=" * 60)
    print("Command Input Parsing")
    print("=" * 60)

    command_details = []
    vision_controller = None
    multimodal_controller = None

    # Priority -1: Multimodal control (highest priority)
    if args_cli.multimodal:
        print("[INPUT MODE] Multimodal control (unified interface)")
        print(f"[INFO] Text prompt: {args_cli.multimodal_prompt or 'None'}")
        print("[INFO] This is the recommended mode for VLA integration")
        command_names = []
        command_details = []
        print("=" * 60 + "\n")
        return command_names, command_details, vision_controller, multimodal_controller

    # Priority 0: Vision control
    if args_cli.vision_control:
        print("[INPUT MODE] Vision-based rule control")
        print("[INFO] Robot will follow red target using vision")
        # Vision control uses its own controller, return empty command sequence
        command_names = []
        command_details = []
        print("=" * 60 + "\n")
        return command_names, command_details, vision_controller, multimodal_controller

    # Priority 1: Text sequence
    if args_cli.text_sequence is not None:
        print(f"[INPUT MODE] Text sequence: '{args_cli.text_sequence}'")
        parsed = TextCommandParser.parse_sequence(args_cli.text_sequence)
        command_names = [name for name, _ in parsed]
        command_details = parsed
        print(f"[PARSED] {len(command_names)} commands from text sequence")

    # Priority 2: Single text command
    elif args_cli.text_command is not None:
        print(f"[INPUT MODE] Single text command: '{args_cli.text_command}'")
        internal_cmd, velocity = TextCommandParser.parse(args_cli.text_command)
        command_names = [internal_cmd]
        command_details = [(internal_cmd, velocity)]
        print(f"[PARSED] '{args_cli.text_command}' -> '{internal_cmd}' with velocity {velocity}")

    # Priority 3: Legacy command sequence
    else:
        print(f"[INPUT MODE] Legacy command sequence: '{args_cli.command_sequence}'")
        command_names = [cmd.strip() for cmd in args_cli.command_sequence.split(",")]
        command_details = []
        for cmd in command_names:
            # Try to parse as text command first
            internal_cmd, velocity = TextCommandParser.parse(cmd)
            command_details.append((internal_cmd, velocity))
        print(f"[PARSED] {len(command_names)} commands from legacy sequence")

    print("=" * 60 + "\n")
    return command_names, command_details, vision_controller, multimodal_controller


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with high-level command control."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_high_level_commands", run_stamp),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # Force enable cameras for vision control or multimodal
        if args_cli.vision_control or args_cli.multimodal:
            mode_str = "multimodal" if args_cli.multimodal else "vision"
            print(f"[INFO] Cameras enabled for {mode_str} control")

    # Avoid noisy rsl-rl warnings about implicit observation-group fallback.
    if agent_cfg.obs_groups is MISSING or agent_cfg.obs_groups is None:
        agent_cfg.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    elif hasattr(agent_cfg.obs_groups, "setdefault"):
        agent_cfg.obs_groups.setdefault("actor", ["policy"])
        agent_cfg.obs_groups.setdefault("critic", ["policy"])
    else:
        agent_cfg.obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt

    # Parse command sequence with text/vision/multimodal support
    command_sequence, command_details, _, _ = parse_command_sequence_with_text_support(args_cli)

    # Check control mode
    use_multimodal = args_cli.multimodal
    use_vision_control = args_cli.vision_control and not use_multimodal  # Multimodal takes precedence

    # Initialize controller
    if use_multimodal:
        print("=" * 80)
        print("Multimodal Control Mode Enabled")
        print("=" * 80)

        # Spawn the red target ball in the scene (for vision-guided modes)
        if args_cli.multimodal_prompt is None or "follow" in (args_cli.multimodal_prompt or "").lower():
            target_path = spawn_red_target_ball(position=(4.0, 0.0, 0.5))

        # Initialize multimodal controller
        multimodal_controller = MultimodalCommandController(
            num_envs=env_cfg.scene.num_envs,
            device=env.unwrapped.device,
            backend_name=args_cli.backend,
            text_prompt=args_cli.multimodal_prompt,
            log_interval=args_cli.multimodal_log_interval,
            num_frames=args_cli.vla_num_frames,
            frame_stride=args_cli.vla_frame_stride,
            infer_interval=args_cli.vla_infer_interval,
            model_path=args_cli.vla_model_path,
            model_device=args_cli.vla_device,
            debug_dir=args_cli.vla_debug_dir,
            debug_max_dumps=args_cli.vla_debug_max_dumps,
            action_mode=args_cli.vla_action_mode,
            trust_remote_code=args_cli.vla_trust_remote_code,
        )

        print("[INFO] Multimodal controller ready")
        print("[INFO] Multimodal image input source is fixed to the robot-mounted front_camera")
        print("[INFO] Unified interface for VLA integration")
        print("=" * 80)

    elif use_vision_control:
        print("=" * 80)
        print("Vision Control Mode Enabled")
        print("=" * 80)

        # Spawn the red target ball in the scene
        target_path = spawn_red_target_ball(position=(4.0, 0.0, 0.5))

        # Initialize vision controller
        vision_controller = VisionRuleController(
            num_envs=env_cfg.scene.num_envs,
            device=env.unwrapped.device,
            log_interval=args_cli.vision_log_interval,
        )

        print("[INFO] Vision controller will process camera images")
        print("[INFO] Vision input source is fixed to the robot-mounted front_camera")
        print("[INFO] Make sure to enable cameras with --video flag for visualization")
        print("=" * 80)

    else:
        # Initialize high-level command controller for text/command sequence
        high_level_controller = HighLevelCommandController(
            num_envs=env_cfg.scene.num_envs,
            device=env.unwrapped.device,
            command_sequence=command_sequence,
            phase_steps=args_cli.phase_steps,
            command_details=command_details,
        )

    # reset environment
    obs = env.get_observations()
    timestep = 0

    print("=" * 80)
    print("Starting high-level command control playback")
    print("Commands will override the default velocity command sampling")
    print("=" * 80)

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        if use_multimodal:
            # Multimodal control (unified interface)
            # Get camera image from the robot-mounted front camera only.
            rgb_image, image_source = get_camera_image_from_env(env, require_front_camera=True)
            if rgb_image is None and timestep == 0:
                print(f"[WARNING] Multimodal input source unavailable: {image_source}")

            # Compute command from multimodal inputs
            command = multimodal_controller.compute_high_level_command(
                rgb_image=rgb_image,
                text_command=args_cli.multimodal_prompt,
            )
            current_cmd_name = multimodal_controller.get_current_command_name(timestep)

            # Override velocity command
            override_velocity_command(env, command)

            # Log multimodal status
            multimodal_controller.log_status(timestep)

        elif use_vision_control:
            # Vision-based control
            # Get camera image from the robot-mounted front camera only.
            rgb_image, image_source = get_camera_image_from_env(env, require_front_camera=True)
            if rgb_image is None and timestep == 0:
                print(f"[WARNING] Vision input source unavailable: {image_source}")

            # Compute command from vision
            command = vision_controller.compute_command_from_vision(rgb_image, image_source=image_source)
            current_cmd_name = vision_controller.get_current_command_name(timestep)

            # Override velocity command
            override_velocity_command(env, command)

            # Log vision status
            vision_controller.log_status(timestep)
        else:
            # Sequence-based control (text or command sequence)
            command = high_level_controller.compute_command(timestep)
            current_cmd_name = high_level_controller.get_current_command_name(timestep)

            # Override the velocity command in the command manager
            override_velocity_command(env, command)

            # Log command at the start of each phase
            if timestep % args_cli.phase_steps == 0:
                vx, vy, wz = command[0, 0].item(), command[0, 1].item(), command[0, 2].item()
                print(f"[Step {timestep:4d}] ACTIVE Command: '{current_cmd_name}' | "
                      f"vx={vx:+.2f}, vy={vy:+.2f}, wz={wz:+.2f}")

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                # extract the neural network for older versions
                if version.parse(installed_version) >= version.parse("2.3.0"):
                    policy_nn = runner.alg.policy
                else:
                    policy_nn = runner.alg.actor_critic
                policy_nn.reset(dones)

        timestep += 1

        if args_cli.video:
            # Exit the play loop after recording the specified video length
            if timestep == args_cli.video_length:
                print(f"[INFO] Finished recording video of {args_cli.video_length} steps")
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    print("=" * 80)
    print(f"Playback completed. Total steps: {timestep}")
    print("=" * 80)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
