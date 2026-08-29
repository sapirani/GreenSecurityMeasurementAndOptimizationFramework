from typing import Dict, Any

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType

from DTOs.hadoop.drl.job_properties import JobProperties
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from hadoop_optimizer.drl_envs.consts import JOB_PROPERTIES_KEY, CURRENT_JOB_CONFIG_KEY, TERMINATE_ACTION_NAME, \
    NEXT_JOB_CONFIG_KEY
from hadoop_optimizer.optimization_mode.abstract_optimization_mode import AbstractOptimizationMode


class RLMode(AbstractOptimizationMode):

    def get_observation_space(self):
        return spaces.Dict({
            JOB_PROPERTIES_KEY: self.job_properties_space,
            CURRENT_JOB_CONFIG_KEY: self.job_config_space,
        })

    def get_action_space(self):
        return spaces.Dict({
            NEXT_JOB_CONFIG_KEY: self.job_config_space,
            TERMINATE_ACTION_NAME: spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
        })

    def construct_observation(
            self,
            episodic_job_properties: JobProperties,
            current_hadoop_config:HadoopJobExecutionConfig
    ) -> Dict[str, Any]:
        return {
            JOB_PROPERTIES_KEY: episodic_job_properties.model_dump(),
            CURRENT_JOB_CONFIG_KEY: current_hadoop_config.model_dump(
                include=self.supported_configurations
            ),
        }

    def should_terminate(self, action: ActType):
        return bool(action[TERMINATE_ACTION_NAME])
