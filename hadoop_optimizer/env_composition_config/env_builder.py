import gymnasium as gym
from hadoop_optimizer.env_composition_config.wrappers_config import get_env_wrappers
from hadoop_optimizer.env_composition_config.env_wrapper_spec import EnvWrappersParams


def build_env(base_env: gym.Env, wrappers_params: EnvWrappersParams) -> gym.Env:
    env = base_env
    for env_spec in get_env_wrappers(wrappers_params):
        env = env_spec.wrapper_cls(env, **env_spec.kwargs)
    return env
