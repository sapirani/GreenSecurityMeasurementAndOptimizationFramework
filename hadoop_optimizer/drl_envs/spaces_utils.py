from typing import Dict, Any
import numpy as np

from gymnasium import spaces
from gymnasium.core import ActType

from hadoop_optimizer.DTOs.hadoop_job_config import HadoopJobConfig
from hadoop_optimizer.DTOs.job_properties import JobProperties
from hadoop_optimizer.drl_envs.consts import TERMINATE_ACTION_NAME


def hadoop_config_as_gymnasium_dict_space() -> spaces.Dict:
    # TODO: extend this implementation with all the flags:
    return spaces.Dict(
        {
            "number_of_mappers": spaces.Discrete(15),
            "number_of_reducers": spaces.Discrete(15),
            "map_memory_mb": spaces.Discrete(20),
            "should_compress": spaces.Discrete(2),
            "map_vcores": spaces.Discrete(4),
            "reduce_vcores": spaces.Discrete(4),
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


def decode_action(action: ActType) -> Dict[str, Any]:
    return {
        "number_of_mappers": int(action[0]) + 1,        # ge=1
        "number_of_reducers": int(action[1]) + 1,       # ge=1
        "map_memory_mb": 250 + int(action[2]) * 50,     # 250, 300,...
        "should_compress": bool(action[3]),             # boolean
        "map_vcores": int(action[4]) + 1,               # ge=1, le=4
        "reduce_vcores": int(action[5]) + 1,            # ge=1, le=4
        TERMINATE_ACTION_NAME: bool(action[6]),         # boolean
    }


# def encode_job_config_state(job_config: HadoopJobConfig) -> Dict[str, Any]:
#     print("encode_job_config_state")
#     print({
#         "number_of_mappers": job_config.number_of_mappers - 1,  # 1..15 -> 0..14
#         "number_of_reducers": job_config.number_of_reducers - 1,  # 1..15 -> 0..14
#         "map_memory_mb": (job_config.map_memory_mb - 250) // 50,  # 250, 300, ... -> 0..20
#         "should_compress": int(job_config.should_compress),  # True/False -> 1/0
#         "map_vcores": job_config.map_vcores - 1,  # 1..4 -> 0..3
#         "reduce_vcores": job_config.reduce_vcores - 1,  # 1..4 -> 0..3
#     })
#
#     return {
#         "number_of_mappers": job_config.number_of_mappers - 1,  # 1..15 -> 0..14
#         "number_of_reducers": job_config.number_of_reducers - 1,  # 1..15 -> 0..14
#         "map_memory_mb": (job_config.map_memory_mb - 250) // 50,  # 250, 300, ... -> 0..20
#         "should_compress": int(job_config.should_compress),  # True/False -> 1/0
#         "map_vcores": job_config.map_vcores - 1,  # 1..4 -> 0..3
#         "reduce_vcores": job_config.reduce_vcores - 1,  # 1..4 -> 0..3
#     }


def flatten_observation(job_properties: JobProperties, job_config: HadoopJobConfig) -> np.ndarray:
    return np.array([
        job_properties.input_size_gb,
        job_properties.cpu_bound_scale,
        job_properties.io_bound_scale,
        job_config.number_of_mappers - 1,
        job_config.number_of_reducers - 1,
        (job_config.map_memory_mb - 150) // 50,
        int(job_config.should_compress),
        job_config.map_vcores - 1,
        job_config.reduce_vcores - 1,
    ], dtype=np.float32)
