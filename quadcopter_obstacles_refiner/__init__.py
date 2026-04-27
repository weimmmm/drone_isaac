import gymnasium as gym

from .quadcopter_obstacles_refiner_env import QuadcopterObstaclesRefinerEnv, QuadcopterObstaclesRefinerEnvCfg
from . import agents


gym.register(
    id="Isaac-Quadcopter-Obstacles-Refiner-v0",
    entry_point=f"{__name__}.quadcopter_obstacles_refiner_env:QuadcopterObstaclesRefinerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_obstacles_refiner_env:QuadcopterObstaclesRefinerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_refiner_cfg:QuadcopterObstaclesRefinerPPORunnerCfg",
    },
)
