from typing import SupportsFloat, Any, Optional, Dict
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame, ActType, ObsType
from pydantic import ValidationError
from hadoop_optimizer.DTOs.hadoop_job_config import HadoopJobConfig
from hadoop_optimizer.DTOs.job_properties import JobProperties
from hadoop_optimizer.drl_envs.consts import TERMINATE_ACTION_NAME
from hadoop_optimizer.drl_envs.spaces_utils import hadoop_config_as_gymnasium_dict_space, \
    job_properties_as_gymnasium_dict_space, decode_action, flatten_observation


class DummyEnv(gym.Env):
    """
    State:
        1. job properties
        2. cluster's load
        3. current hadoop job configuration
    """
    def __init__(self):
        super().__init__()

        # self.observation_space = spaces.Box(
        #     low=np.array([0, 0.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        #     high=np.array([300, 1.0, 1.0, 14, 14, 19, 1, 3, 3], dtype=np.float32),
        #     dtype=np.float32
        # )

        self.observation_space = spaces.Dict({
            "job_properties": spaces.Dict({
                "input_size": spaces.Box(low=0, high=300, shape=(), dtype=np.float32)
            }),
            "current_configuration": spaces.Dict({
                "number_of_mappers": spaces.Box(low=0, high=15, shape=(), dtype=np.uint16)
            }),
        })

        self.action_space = self.action_space = spaces.MultiDiscrete([
            15,     # number_of_mappers     (1–15)
            15,     # number_of_reducers    (1–15)
            20,     # map_memory_mb bins    (250, 300,...)
            2,      # should_compress       (0/1)
            4,      # map_vcores            (1–4)
            4,      # reduce_vcores         (1–4)
            2,      # terminate             (0/1)
        ])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        return {
            "job_properties": {
                "input_size": 10
            },
            "current_configuration": {
                "number_of_mappers": 3
            }
        }, {}

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return {
            "job_properties": {
                "input_size": 3
            },
            "current_configuration": {
                "number_of_mappers": 2
            }
        }, 0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return None
