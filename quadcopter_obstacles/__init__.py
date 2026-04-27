import gymnasium as gym

from .quadcopter_obstacles_env import QuadcopterObstaclesEnv, QuadcopterObstaclesEnvCfg
from . import agents

gym.register(
    id="Isaac-Quadcopter-Obstacles-v0",
    entry_point=f"{__name__}.quadcopter_obstacles_env:QuadcopterObstaclesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_obstacles_env:QuadcopterObstaclesEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterObstaclesPPORunnerCfg",
    },
)