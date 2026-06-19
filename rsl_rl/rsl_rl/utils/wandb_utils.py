# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import re
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        # Get the run name in the same grouped style as NavRL:
        # <experiment>/<MM-DD_HH-MM>
        log_name = os.path.split(log_dir)[-1]
        match = re.match(r"\d{4}-(\d{2}-\d{2})_(\d{2}-\d{2})", log_name)
        time_name = f"{match.group(1)}_{match.group(2)}" if match else log_name
        run_name = f"{cfg.get('run_name', log_name)}/{time_name}"

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        init_kwargs = {
            "project": project,
            "entity": entity,
            "name": run_name,
            "dir": log_dir,
        }
        run_id = os.environ.get("WANDB_RUN_ID")
        if run_id:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = os.environ.get("WANDB_RESUME", "allow")

        # Initialize wandb
        wandb.init(**init_kwargs)
        try:
            wandb.define_metric("Train/iteration")
            wandb.define_metric("*", step_metric="Train/iteration")
        except Exception as err:
            print(f"[WARN] Failed to define WandB metrics: {err}", flush=True)
        if wandb.run is not None:
            print(f"[INFO] WandB run id: {wandb.run.id}", flush=True)
            if getattr(wandb.run, "url", None):
                print(f"[INFO] WandB run URL: {wandb.run.url}", flush=True)

        # Add log directory to wandb
        wandb.config.update({"log_dir": log_dir})

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

    def _to_wandb_scalar(self, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            if value.numel() == 1:
                return value.item()
            return value
        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                return value
        return value

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        wandb.config.update({"runner_cfg": runner_cfg})
        wandb.config.update({"policy_cfg": policy_cfg})
        wandb.config.update({"alg_cfg": alg_cfg})
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)})

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb_scalars = {self._map_path(tag): self._to_wandb_scalar(scalar_value)}
        if global_step is not None:
            global_step = int(global_step)
            wandb_scalars["Train/iteration"] = global_step
            wandb.log(wandb_scalars, step=global_step)
        else:
            wandb.log(wandb_scalars)

    def add_scalars(self, scalars: dict, global_step=None):
        for tag, scalar_value in scalars.items():
            super().add_scalar(tag, scalar_value, global_step=global_step)
        wandb_scalars = {self._map_path(tag): self._to_wandb_scalar(value) for tag, value in scalars.items()}
        if global_step is not None:
            global_step = int(global_step)
            wandb_scalars["Train/iteration"] = global_step
            wandb.log(wandb_scalars, step=global_step)
        else:
            wandb.log(wandb_scalars)

    def stop(self):
        try:
            super().flush()
            super().close()
        finally:
            wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        wandb.save(path, base_path=os.path.dirname(path))

    def add_video_file(self, tag, path, global_step=None, fps=50):
        if global_step is not None:
            wandb.log(
                {tag: wandb.Video(path, fps=fps, format="mp4"), "Train/iteration": int(global_step)},
                step=int(global_step),
            )
        else:
            wandb.log({tag: wandb.Video(path, fps=fps, format="mp4")})

    """
    Private methods.
    """

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path
