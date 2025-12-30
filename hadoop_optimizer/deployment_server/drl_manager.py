from stable_baselines3.common.base_class import BaseAlgorithm

from hadoop_optimizer.DTOs.hadoop_job_execution_config import HadoopJobExecutionConfig
from hadoop_optimizer.DTOs.job_properties import JobProperties
from hadoop_optimizer.drl_envs.consts import CURRENT_JOB_CONFIG_KEY, ELAPSED_STEPS_KEY, MAX_STEPS_KEY
from hadoop_optimizer.drl_envs.deployment_env import OptimizerDeploymentEnv
from hadoop_optimizer.erros import EnvironmentTruncatedException


class DRLManager:
    def __init__(self, deployment_agent: BaseAlgorithm, deployment_env: OptimizerDeploymentEnv):
        self.deployment_agent = deployment_agent
        self.deployment_env = deployment_env

    def determine_best_job_configuration(
        self,
        job_properties: JobProperties,
    ) -> HadoopJobExecutionConfig:
        with self.deployment_env:
            obs, _ = self.deployment_env.reset(options=job_properties.model_dump())
            while True:
                action, _states = self.deployment_agent.predict(obs)
                obs, rewards, terminated, truncated, info = self.deployment_env.step(action)
                self.deployment_env.render()

                if terminated:
                    return HadoopJobExecutionConfig.model_validate(info[CURRENT_JOB_CONFIG_KEY])

                if truncated:
                    raise EnvironmentTruncatedException(
                        HadoopJobExecutionConfig.model_validate(info[CURRENT_JOB_CONFIG_KEY]),
                        info[ELAPSED_STEPS_KEY],
                        info[MAX_STEPS_KEY],
                    )
