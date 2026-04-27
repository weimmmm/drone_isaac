# Quadcopter Safety

Stage 1 trains a Go2-style high-level safety command policy with PPO.
It uses a dedicated safety-only environment that removes navigation objectives but keeps the same Isaac Sim scene, depth camera, and fixed low-level controller.

The trained checkpoint is intended to be frozen and loaded by `quadcopter_camera` during stage 2.

Example:

```bash
python drone_isaac/quadcopter_safety/train.py --headless --device cuda:0
```
