import torch
from pathlib import Path

PRECISION = 2
OPTIMIZER = torch.optim.Adam
BATCH_SIZE = 256
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3

# Default local output root for future MBPO experiments inside this task package.
ROOT_DIR = str(Path(__file__).resolve().parents[1] / "logs" / "mbpo")
