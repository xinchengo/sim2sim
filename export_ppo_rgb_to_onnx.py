import argparse
import os
import sys
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

# ManiSkill imports
import mani_skill.envs  # registers envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Import the Agent from the ManiSkill PPO baseline file
# Adjust this path if needed so Python can import the file where ppo_rgb.py lives.
# Example: repo_root/examples/baselines/ppo/ppo_rgb.py
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PP0_RGB_DIR = os.path.join(CUR_DIR, "examples", "baselines", "ppo")
if PP0_RGB_DIR not in sys.path:
    sys.path.insert(0, PP0_RGB_DIR)

from ppo_rgb import Agent  # uses NatureCNN internally

class OnnxActor(nn.Module):
    """
    A wrapper to expose deterministic actor_mean and value for ONNX export.
    Inputs:
      - rgb: float32 [N, H, W, C] with values in [0, 255]
      - state: float32 [N, S] if present during training
    Outputs:
      - action: float32 [N, A] (actor mean)
      - value: float32 [N, 1]
    """
    def __init__(self, agent: Agent, include_state: bool):
        super().__init__()
        self.agent = agent
        self.include_state = include_state

    def forward(self, rgb: torch.Tensor, state: torch.Tensor = None):
        obs = {"rgb": rgb}
        if self.include_state:
            if state is None:
                raise RuntimeError("State input is required but missing.")
            obs["state"] = state
        # Use the same feature path as training
        x = self.agent.get_features(obs)
        action_mean = self.agent.actor_mean(x)
        value = self.agent.critic(x)
        return action_mean, value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt/.pth checkpoint (state_dict) saved by ppo_rgb.py")
    parser.add_argument("--env_id", type=str, default="PickCube-v1")
    parser.add_argument("--output", type=str, default=None, help="Output .onnx path")
    parser.add_argument("--control_mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.output is None:
        base, _ = os.path.splitext(args.checkpoint)
        args.output = base + ".onnx"

    # Build a tiny env to get the exact obs/action specs as training
    env_kwargs = dict(obs_mode="rgb", render_mode="none", sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    base_env = gym.make(args.env_id, num_envs=1, reconfiguration_freq=None, **env_kwargs)
    base_env = FlattenRGBDObservationWrapper(base_env, rgb=True, depth=False, state=True)  # include_state was True in your run
    if isinstance(base_env.action_space, gym.spaces.Dict):
        base_env = FlattenActionSpaceWrapper(base_env)
    venv = ManiSkillVectorEnv(base_env, num_envs=1, ignore_terminations=True, record_metrics=False)
    obs, _ = venv.reset(seed=0)

    include_state = "state" in obs
    # Build Agent with proper sample_obs to size conv/MLP layers
    agent = Agent(venv, sample_obs=obs).to(args.device)
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    agent.load_state_dict(state_dict)
    agent.eval()

    # Prepare dummy inputs using real shapes
    dummy_rgb = obs["rgb"].to(torch.float32)  # [1, H, W, C], float32. Model divides by 255 internally.
    dummy_state = obs["state"].to(torch.float32) if include_state else None

    wrapper = OnnxActor(agent, include_state=include_state).to(args.device)
    wrapper.eval()

    input_names = ["rgb"] + (["state"] if include_state else [])
    output_names = ["action", "value"]
    dynamic_axes = {"rgb": {0: "batch"}}
    if include_state:
        dynamic_axes["state"] = {0: "batch"}
    dynamic_axes["action"] = {0: "batch"}
    dynamic_axes["value"] = {0: "batch"}

    with torch.no_grad():
        if include_state:
            torch.onnx.export(
                wrapper,
                (dummy_rgb, dummy_state),
                args.output,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=17,
            )
        else:
            torch.onnx.export(
                wrapper,
                (dummy_rgb,),
                args.output,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=17,
            )
    print(f"Exported ONNX model to: {args.output}")

if __name__ == "__main__":
    main()