import gymnasium as gym

from .quadcopter_obstacles_env import QuadcopterObstaclesEnv, QuadcopterObstaclesEnvCfg
from .quadcopter_obstacles_depth_eval_env import QuadcopterObstaclesDepthEvalEnv, QuadcopterObstaclesDepthEvalEnvCfg

gym.register(
    id="Isaac-Quadcopter-Obstacles-MBPO-v0",
    entry_point=f"{__name__}.quadcopter_obstacles_env:QuadcopterObstaclesEnv",
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Quadcopter-Obstacles-MBPO-DepthEval-v0",
    entry_point=f"{__name__}.quadcopter_obstacles_depth_eval_env:QuadcopterObstaclesDepthEvalEnv",
    disable_env_checker=True,
)
