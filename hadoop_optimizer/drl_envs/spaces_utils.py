from typing import Set
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType
from hadoop_optimizer.DTOs.hadoop_job_config import HadoopJobConfig
from hadoop_optimizer.drl_envs.consts import TERMINATE_ACTION_NAME


def hadoop_config_as_gymnasium_dict_space() -> spaces.Dict:
    # TODO: extend this implementation with all the flags:
    return spaces.Dict(
        {
            "number_of_mappers": spaces.Box(low=1, high=15, shape=(), dtype=np.float32),
            "number_of_reducers": spaces.Box(low=1, high=15, shape=(), dtype=np.float32),
            "map_memory_mb": spaces.Box(low=100, high=1500, shape=(), dtype=np.float32),
            "should_compress": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
            "map_vcores": spaces.Box(low=1, high=4, shape=(), dtype=np.float32),
            "reduce_vcores": spaces.Box(low=1, high=4, shape=(), dtype=np.float32),
        }
    )


def job_properties_as_gymnasium_dict_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "input_size_gb": spaces.Box(low=0, high=300, shape=(), dtype=np.float32),
            "cpu_bound_scale": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
            "io_bound_scale": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
        }
    )


def dict_to_ndarrays(d):
    """
    Recursively convert all values in a nested dictionary into np.array([value], dtype=np.float32)
    """
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_dict[k] = dict_to_ndarrays(v)  # recursive call
        else:
            new_dict[k] = np.array([v], dtype=np.float32)
    return new_dict


def decode_action_types(action: ActType, supported_fields: Set[str]):
    decoded_action = {}
    current_job_config = action["current_job_config"]

    for field_name, field_info in HadoopJobConfig.model_fields.items():
        if field_name not in supported_fields:
            continue

        if field_info.annotation is float:
            current_job_config[field_name] = float(current_job_config[field_name])
        else:
            current_job_config[field_name] = field_info.annotation(np.round(current_job_config[field_name]))

    decoded_action["current_job_config"] = current_job_config
    decoded_action[TERMINATE_ACTION_NAME] = bool(np.round(action[TERMINATE_ACTION_NAME]))

    return decoded_action
