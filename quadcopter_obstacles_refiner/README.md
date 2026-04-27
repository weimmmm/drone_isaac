# Quadcopter Obstacles Student

This package now trains a standalone student policy for obstacle navigation.

The setup is:

- teacher: privileged policy used only during training to provide target actions
- student: consumes deployable observations only
- executed action: student action directly

The student is trained online with PPO plus an annealed distillation loss:

- early training: stronger teacher guidance
- later training: weaker teacher guidance

Current student observations:

- `policy_image`: forward first-person depth image
- `policy_state`: low-dimensional onboard state from the task

## Train

```bash
python /home/wei/End_to_end/drone_isaac/quadcopter_obstacles_refiner/train.py \
  --num_envs 64 \
  --headless --device cuda:0
```

## Export Teacher Actor

```bash
python /home/wei/End_to_end/drone_isaac/quadcopter_obstacles_refiner/export_teacher_actor.py \
  --checkpoint /home/wei/End_to_end/logs/rsl_rl/quadcopter_obstacles_v5/2026-03-14_02-50-17/model_3000.pt \
  --output /home/wei/End_to_end/logs/rsl_rl/quadcopter_obstacles_refiner/teacher_actor_model_3000.pt
```

The exported actor-only file can be passed back through `--teacher_checkpoint`.

## Evaluate

Student:

```bash
python /home/wei/End_to_end/drone_isaac/quadcopter_obstacles_refiner/eval_refiner.py \
  --mode student \
  --checkpoint /home/wei/End_to_end/logs/rsl_rl/quadcopter_obstacles_refiner/<run>/model_500.pt \
  --num_envs 128 \
  --headless --device cuda:0
```

Teacher baseline:

```bash
python /home/wei/End_to_end/drone_isaac/quadcopter_obstacles_refiner/eval_refiner.py \
  --mode teacher \
  --num_envs 128 \
  --headless --device cuda:0
```

## Notes

- Deployment uses only the student checkpoint.
- Teacher actions are not part of the student input.
- Teacher is used only for the annealed distillation loss during training and optional evaluation comparison.
