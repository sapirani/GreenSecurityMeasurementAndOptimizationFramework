from typing import Any

from pydantic import ValidationError

from DTOs.hadoop.drl.job_properties import JobProperties
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from hadoop_optimizer.drl_envs.abstract_hadoop_optimizer_env import AbstractOptimizerEnvInterface


class OptimizerDeploymentEnv(AbstractOptimizerEnvInterface):
    DEFAULT_REWARD = 0

    def _init_episodic_job(self, options: dict[str, Any] | None) -> JobProperties:
        """
        :param options: Should contain the job properties that we want to optimize its Hadoop config params
            through this episode
        """
        if not options:
            raise ValueError("Expected to retrieve the job properties on reset")

        try:
            return JobProperties.model_validate(options)
        except ValidationError as e:
            raise ValueError("Received unexpected job properties") from e

    def _extra_step_init(self):
        pass

    def _compute_reward(self, job_config: HadoopJobExecutionConfig, *, terminated: bool, truncated: bool) -> float:
        """
        :return: some default reward, since there is no need to compute rewards in the deployment phase
        """
        return OptimizerDeploymentEnv.DEFAULT_REWARD

    def _custom_rendering(self):
        pass
