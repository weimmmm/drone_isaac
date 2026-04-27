from __future__ import annotations

import argparse
import os
import sys

import torch

from quadcopter_obstacles_teacher.agents.rsl_rl_ppo_cfg import QuadcopterObstaclesPPORunnerCfg


TASK_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_DIR = os.path.abspath(os.path.join(TASK_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(TASK_DIR, "..", ".."))

for path in (ROOT_DIR, ENV_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a frozen teacher actor-only checkpoint for the refiner.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--activation", type=str, default=None)
    args = parser.parse_args()

    loaded = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    state_dict = loaded["model_state_dict"]

    teacher_cfg = QuadcopterObstaclesPPORunnerCfg()
    activation = args.activation or str(teacher_cfg.policy.activation)

    actor_weight_keys = sorted(
        [key for key in state_dict.keys() if key.startswith("actor.") and key.endswith(".weight")],
        key=lambda key: int(key.split(".")[1]),
    )
    if not actor_weight_keys:
        raise RuntimeError(f"No actor weights found in checkpoint: {args.checkpoint}")

    layer_dims = [int(state_dict[actor_weight_keys[0]].shape[1])]
    actor_state_dict: dict[str, torch.Tensor] = {}
    linear_idx = 0
    for weight_key in actor_weight_keys:
        bias_key = weight_key.replace(".weight", ".bias")
        actor_state_dict[f"{linear_idx}.weight"] = state_dict[weight_key].detach().cpu()
        actor_state_dict[f"{linear_idx}.bias"] = state_dict[bias_key].detach().cpu()
        layer_dims.append(int(state_dict[weight_key].shape[0]))
        linear_idx += 2

    bundle = {
        "export_format": "frozen_teacher_actor_v1",
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "activation": activation,
        "layer_dims": layer_dims,
        "actor_state_dict": actor_state_dict,
        "obs_mean": state_dict["actor_obs_normalizer._mean"].detach().cpu(),
        "obs_std": state_dict["actor_obs_normalizer._std"].detach().cpu(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(bundle, args.output)
    print(f"exported_teacher_actor={os.path.abspath(args.output)}")
    print(f"activation={activation}")


if __name__ == "__main__":
    main()
