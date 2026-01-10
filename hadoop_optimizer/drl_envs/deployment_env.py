from typing import Any
from pydantic import ValidationError

from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_properties import JobProperties
from hadoop_optimizer.drl_envs.abstract_hadoop_optimizer_env import AbstractOptimizerEnvInterface


class OptimizerDeploymentEnv(AbstractOptimizerEnvInterface):
    def _init_episodic_job(self, options: dict[str, Any] | None) -> JobProperties:
        if not options:
            raise ValueError("Expected to retrieve the job properties on reset")

        try:
            return JobProperties.model_validate(options)
        except ValidationError as e:
            raise ValueError("Received unexpected job properties") from e

    def _compute_reward(self, job_config: HadoopJobExecutionConfig, terminated: bool, truncated: bool) -> float:
        """
        :return: 0, since upon deployment there is no need to compute rewards
        """
        return 0

    def _custom_rendering(self):
        pass
