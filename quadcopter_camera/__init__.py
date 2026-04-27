import gymnasium as gym

from .quadcopter_obstacles_env import QuadcopterObstaclesEnv, QuadcopterObstaclesEnvCfg
from .quadcopter_camera_shared_env import QuadcopterSharedMapEnv, QuadcopterSharedMapEnvCfg
from .quadcopter_single_world_depth_env import (
    QuadcopterSingleWorldDepthEnvCfg,
    QuadcopterSingleWorldDepthGymEnv,
)
from . import agents

gym.register(
    id="Isaac-Quadcopter-Camera-v0",
    entry_point=f"{__name__}.quadcopter_obstacles_env:QuadcopterObstaclesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_obstacles_env:QuadcopterObstaclesEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterObstaclesPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Quadcopter-Camera-SharedMap-v0",
    entry_point=f"{__name__}.quadcopter_camera_shared_env:QuadcopterSharedMapEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_camera_shared_env:QuadcopterSharedMapEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterObstaclesPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Quadcopter-Camera-SingleWorld-v0",
    entry_point=f"{__name__}.quadcopter_single_world_depth_env:QuadcopterSingleWorldDepthGymEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_single_world_depth_env:QuadcopterSingleWorldDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterObstaclesPPORunnerCfg",
    },
)
