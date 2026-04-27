# MBPO Teacher: Obstacle Avoidance & Single Target Navigation
This task folder now targets a staged **MBPO + SAC** teacher setup for the obstacle-navigation quadcopter environment.

<p align="center">
<img src="../../docs/figures/obstacles.gif" alt="Obstacle Avoidance Demo" width="600">





<em>Teacher environment for model-based obstacle navigation experiments.</em>
</p>

## 🎯 Task Objective

The agent must fly to **one random target point** without colliding.

* **Navigation:** Reach within 0.8m of the sampled target to finish the episode.
* **Perception:** Detect and evade 50 randomly placed pillars (Static Obstacles).
* **Constraint:** The obstacles are generated procedurally every reset, preventing map memorization.


## 🧠 Observation Space

The current policy/model state is a compact **36-D** vector:

- Body-frame linear velocity, angular velocity, and projected gravity
- Goal represented as `target_dist / angle_to_target / z_diff`
- Current command velocity
- Altitude and closest-obstacle distance
- Forward obstacle distance and goal-direction obstacle distance
- **16** forward 180-degree horizontal rays
- Episode progress



## 📉 Reward Function: The Safety-Speed Trade-off

The reward function is designed to balance the "Greedy" desire to reach the goal with the "Fear" of collision.

1. **Target Navigation:**


* Includes dense progress/velocity shaping and a smaller sparse bonus for reaching the target.


2. **Obstacle Repulsion (Safety Bubble):**
We use an exponential penalty that activates sharply only when getting too close (< 1.0m).





## Training

Use the MBPO trainer in this folder:

```bash
python train_mbpo.py --headless --device cuda:0 --epochs 200
```

Recommended stable bring-up command:

```bash
python train_mbpo.py \
  --headless \
  --device cuda:0 \
  --epochs 10 \
  --steps_per_epoch 200 \
  --buffer_min 512 \
  --rollout_batch_size 64 \
  --model_initial_steps 200 \
  --model_steps 100 \
  --solver_updates_per_step 1
```

Notes:

- Alpha autotuning is disabled by default for now because the Isaac GPU run was unstable with the original temperature-update path.
- Re-enable it only if you explicitly want to test it: `--enable_alpha_autotune`
- Detailed rollout / solver debug logs stay off by default and can be enabled with `--debug_step_logs`

## MBPO Core Migration

This task folder contains a staged `mbpo_core/` package with the reusable pieces migrated from `Safe-MBPO-main`:

- `dynamics.py`: probabilistic ensemble dynamics model
- `sac.py`: soft actor-critic core
- `sampling.py`: generic replay buffer
- `smbpo.py`: environment-agnostic `SMBPOCore` skeleton prepared for later integration

These modules are now the primary training path for this task. The previous PPO-specific runner config has been removed from this package.

Current status:

- `train_mbpo.py` is a staged single-environment trainer intended for algorithm bring-up, not large-scale Isaac parallel training yet.
- The MBPO adapter directly uses the environment's compact policy observation as the model state.
