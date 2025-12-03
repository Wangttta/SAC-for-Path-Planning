# 路径规划环境
from .env import NormalizedActionsWrapper, LidarModel
__all__ = ["NormalizedActionsWrapper", "LidarModel"]

import gymnasium as gym

gym.register(
    "PathPlan-v0",
    entry_point=f"{__name__}.env:DynamicPathPlanning",
    disable_env_checker=True, 
    kwargs=dict(
        vectorization_mode="async"
    )
)

gym.register(
    "PathSearch-v0",
    entry_point=f"{__name__}.env:StaticPathPlanning",
    disable_env_checker=True, 
    kwargs=dict(
        vectorization_mode="async"
    )
)