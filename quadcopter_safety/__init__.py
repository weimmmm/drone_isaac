import gymnasium as gym

from .filter import SafetyActionFilter, SafetyFilterConfig, SafetyFilterNetwork
from .safety_env import QuadcopterSafetyEnv, QuadcopterSafetyEnvCfg
from . import agents

gym.register(
    id="Isaac-Quadcopter-Safety-v0",
    entry_point=f"{__name__}.safety_env:QuadcopterSafetyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.safety_env:QuadcopterSafetyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterSafetyPPORunnerCfg",
    },
)

__all__ = [
    "SafetyActionFilter",
    "SafetyFilterConfig",
    "SafetyFilterNetwork",
    "QuadcopterSafetyEnv",
    "QuadcopterSafetyEnvCfg",
]
