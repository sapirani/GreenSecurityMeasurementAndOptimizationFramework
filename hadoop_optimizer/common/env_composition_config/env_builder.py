import itertools
from typing import List, Optional, Callable
import gymnasium as gym
from hadoop_optimizer.common.env_composition_config.env_wrapper_spec import EnvWrappersParams, EnvWrapperSpec
from hadoop_optimizer.common.env_composition_config.wrappers_config import get_basic_env_wrappers


def build_env(
        base_env: gym.Env,
        basic_wrappers_params: EnvWrappersParams,
        extended_wrappers_params: Optional[EnvWrappersParams],
        extended_env_wrappers: Optional[Callable[[EnvWrappersParams], List[EnvWrapperSpec]]] = None,
) -> gym.Env:
    if extended_env_wrappers is None:
        extended_env_wrappers = lambda _: []

    env = base_env
    for env_spec in itertools.chain(
            get_basic_env_wrappers(basic_wrappers_params),
            extended_env_wrappers(extended_wrappers_params)
    ):
        env = env_spec.wrapper_cls(env, **env_spec.kwargs)
    return env
