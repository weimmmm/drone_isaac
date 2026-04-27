# rsl_rl SAC Teacher: Obstacle Avoidance & Target Navigation

This task trains the compact lidar obstacle-navigation quadcopter environment with a standard Soft Actor-Critic path built into the vendored `rsl_rl` library.

## Task

The policy controls high-level body-frame velocity commands:

- `vx`, `vy` in the drone body frame
- `vz` in the world vertical axis

The observation is the environment's compact `policy` vector:

- body-frame linear velocity
- yaw rate
- target distance, target bearing, and target height delta
- current command velocity
- altitude
- 35 normalized horizontal lidar/raycast distances

## Training

```bash
cd /home/wei/End_to_end/drone_isaac/quadcopter_sac
/home/wei/IsaacLab/isaaclab.sh -p train.py --headless --device cuda:0 --num_envs 1024 --max_iterations 5000
```

Small bring-up command:

```bash
cd /home/wei/End_to_end/drone_isaac/quadcopter_sac
/home/wei/IsaacLab/isaaclab.sh -p train.py \
  --headless \
  --device cuda:0 \
  --num_envs 32 \
  --max_iterations 5 \
  --random_steps 128 \
  --learning_starts 128 \
  --batch_size 128 \
  --replay_buffer_size 20000 \
  --gradient_steps_per_iteration 10
```

Logs and checkpoints are written under:

```text
/home/wei/End_to_end/logs/rsl_rl/quadcopter_sac/
```

The default replay buffer stores `10_000_000` transitions on CPU memory. Sampled batches are moved to the training device before SAC updates.
Actor and critics are updated at the same frequency. Warmup exploration has only a light forward bias in the drone body frame (`vx` mean 0.20) so early replay data remains diverse while still containing attempts to enter the map.

The SAC policy uses a NavRL-style CNN+MLP encoder:

- low-dimensional state: first 11 observation values
- 2-D lidar: last 35 values reshaped to `(N, 1, 35, 1)`
- lidar CNN: Conv2d `1->4`, Conv2d `4->16`, Conv2d `16->16`, then Linear to 128
- fusion MLP: `[lidar_feature_128, state_11] -> 256 -> 256`
- SAC actor and twin critics use separate encoders; target critics keep separate soft-updated target encoders.

## Implementation

The SAC path uses:

- `rsl_rl.modules.ActorCriticSAC`
- `rsl_rl.algorithms.SAC`
- `rsl_rl.storage.ReplayBuffer`
- `rsl_rl.runners.OffPolicyRunner`
- `quadcopter_sac/agents/rsl_rl_sac_cfg.py`
