import gymnasium as gym

from .quadcopter_obstacles_env import QuadcopterObstaclesEnv, QuadcopterObstaclesEnvCfg
from .quadcopter_obstacles_depth_eval_env import QuadcopterObstaclesDepthEvalEnv, QuadcopterObstaclesDepthEvalEnvCfg
from . import agents

gym.register(
    id="Isaac-Quadcopter-Obstacles-Teacher-v0",
    entry_point=f"{__name__}.quadcopter_obstacles_env:QuadcopterObstaclesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_obstacles_env:QuadcopterObstaclesEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterObstaclesPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Quadcopter-Obstacles-Teacher-DepthEval-v0",
    entry_point=f"{__name__}.quadcopter_obstacles_depth_eval_env:QuadcopterObstaclesDepthEvalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_obstacles_depth_eval_env:QuadcopterObstaclesDepthEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterObstaclesPPORunnerCfg",
    },
)
