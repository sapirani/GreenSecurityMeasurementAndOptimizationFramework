from abc import ABC, abstractmethod
from typing import Any, Dict, Set
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType
from DTOs.hadoop.drl.job_properties import JobProperties
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.range import Range
from hadoop_optimizer.drl_envs.consts import NEXT_JOB_CONFIG_KEY


class AbstractOptimizationMode(ABC):

    @property
    def job_config_space(self) -> spaces.Dict:
        """
        Note: this space is the major part of the action space. It *must* suit the resource capabilities of the server
        in which the tasks are executed at.
        Do not ask for more resources (e.g., large amount of CPU cores) than what is configured in the server as
        the limits.
        """
        # TODO: extend this implementation with all the flags:
        return spaces.Dict({
            "number_of_mappers": spaces.Box(low=1, high=15, shape=(), dtype=np.float32),
            "number_of_reducers": spaces.Box(low=1, high=15, shape=(), dtype=np.float32),
            "map_memory_mb": spaces.Box(low=256, high=4096, shape=(), dtype=np.float32),
            "should_compress": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
            "map_vcores": spaces.Box(low=1, high=5, shape=(), dtype=np.float32),
            "reduce_vcores": spaces.Box(low=1, high=5, shape=(), dtype=np.float32),
        })

    @property
    def job_properties_space(self) -> spaces.Dict:
        return spaces.Dict({
            "input_size_gb": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
            "cpu_bound_scale": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
            "io_bound_scale": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
        })

    @staticmethod
    def get_space_ranges(space: spaces.Dict) -> Dict[str, Range]:
        return {
            config_name: Range(low=box_space.low, high=box_space.high)
            for config_name, box_space in space.spaces.items()
        }

    @property
    def supported_configurations(self) -> Set[str]:
        return set(self.job_config_space.keys())

    @staticmethod
    def parse_action_to_config(action: ActType) -> HadoopJobExecutionConfig:
        """ Apply action to modify next hadoop configuration """
        # TODO: if actions become deltas: start from self._current_hadoop_config, instead of using default configuration
        default_config = HadoopJobExecutionConfig()
        return default_config.model_copy(
            update=action[NEXT_JOB_CONFIG_KEY],
            deep=True,
        )

    @abstractmethod
    def get_observation_space(self) -> spaces.Dict:
        pass

    @abstractmethod
    def get_action_space(self) -> spaces.Dict:
        pass

    @abstractmethod
    def construct_observation(
            self,
            episodic_job_properties: JobProperties,
            current_hadoop_config: HadoopJobExecutionConfig
    ) -> Dict[str, Any]:
        pass


    @abstractmethod
    def should_terminate(self, action: ActType) -> bool:
        pass
