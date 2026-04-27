import gymnasium as gym

from .quadcopter_obstacles_env import QuadcopterObstaclesEnv, QuadcopterObstaclesEnvCfg
from . import agents

gym.register(
    id="Isaac-Quadcopter-Obstacles-SAC-v0",
    entry_point=f"{__name__}.quadcopter_obstacles_env:QuadcopterObstaclesEnv",
    kwargs={
        "env_cfg_entry_point": QuadcopterObstaclesEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_sac_cfg:QuadcopterObstaclesSACRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Quadcopter-Obstacles-MBPO-v0",
    entry_point=f"{__name__}.quadcopter_obstacles_env:QuadcopterObstaclesEnv",
    kwargs={
        "env_cfg_entry_point": QuadcopterObstaclesEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_sac_cfg:QuadcopterObstaclesSACRunnerCfg",
    },
    disable_env_checker=True,
)
