from dataclasses import dataclass, field, fields
from typing import Callable, Dict, Any

import numpy as np
from dependency_injector.providers import Configuration


@dataclass(frozen=True)
class EnvWrapperSpec:
    wrapper_cls: Callable
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvWrappersParams:
    max_episode_steps: int
    min_action: np.float32 = 0
    max_action: np.float32 = 1
    min_obs: np.float32 = -1
    max_obs: np.float32 = 1

    @classmethod
    def from_config(cls, config: Configuration):
        params = {}
        for field_info in fields(cls):
            field_name = field_info.name
            if field_name in config:
                params[field_name] = config[field_name]

        return cls(**params)
