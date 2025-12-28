from gymnasium import spaces

from hadoop_optimizer.drl_envs.consts import TERMINATE_ACTION_NAME, NEW_CONFIG_ACTION_NAME


def hadoop_config_as_gymnasium_dict_space() -> spaces.Dict:
    # TODO: extend this implementation with all the flags:
    return spaces.Dict(
        {
            "number_of_mappers": spaces.Box(low=1, high=15, shape=(), dtype=int),
            "number_of_reducers": spaces.Box(low=1, high=15, shape=(), dtype=int),
            "memory_mb": spaces.Box(low=128, high=8192, shape=(), dtype=int),
            "compress": spaces.Discrete(2),
            "map_vcores": spaces.Box(low=1, high=4, shape=(), dtype=int),
            "reduce_vcores": spaces.Box(low=1, high=4, shape=(), dtype=int),
        }
    )


def job_properties_as_gymnasium_dict_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "input_size_gb": spaces.Box(low=0, high=300, shape=(), dtype=float),
            "cpu_bound_scale": spaces.Box(low=0, high=1, shape=(), dtype=float),
            "io_bound_scale": spaces.Box(low=0, high=1, shape=(), dtype=float),
        }
    )


def add_termination_action(action_space: spaces.Space) -> spaces.Dict:
    return spaces.Dict(
        {
            NEW_CONFIG_ACTION_NAME: action_space,
            TERMINATE_ACTION_NAME: spaces.Discrete(2)
        }
    )
