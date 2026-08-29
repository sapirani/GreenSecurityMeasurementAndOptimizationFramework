from typing import Dict, Any

from gymnasium import spaces
from gymnasium.core import ActType

from DTOs.hadoop.drl.job_properties import JobProperties
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from hadoop_optimizer.drl_envs.consts import JOB_PROPERTIES_KEY, NEXT_JOB_CONFIG_KEY
from hadoop_optimizer.optimization_mode.abstract_optimization_mode import AbstractOptimizationMode


class ContextualBanditMode(AbstractOptimizationMode):

    def get_observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            JOB_PROPERTIES_KEY: self.job_properties_space,
        })

    def get_action_space(self) -> spaces.Dict:
        return spaces.Dict({
            NEXT_JOB_CONFIG_KEY: self.job_config_space,
        })

    def construct_observation(
            self,
            episodic_job_properties: JobProperties,
            current_hadoop_config: HadoopJobExecutionConfig
    ) -> Dict[str, Any]:
        return {
            JOB_PROPERTIES_KEY: episodic_job_properties.model_dump()
        }

    def should_terminate(self, action: ActType):
        # Every contextual-bandit decision is one step.
        return True
