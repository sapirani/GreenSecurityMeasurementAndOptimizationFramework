import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, ActType
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from hadoop_optimizer.drl_envs.consts import NEXT_JOB_CONFIG_KEY, TERMINATE_ACTION_NAME
from hadoop_optimizer.drl_envs.deployment_env import OptimizerDeploymentEnv


class ActionTypesDecoder(gym.ActionWrapper):
    def action(self, action: WrapperActType) -> ActType:
        assert isinstance(self.unwrapped, OptimizerDeploymentEnv), \
            "This action decoder wrapper is dedicated for the OptimizerDeploymentEnv"

        decoded_action = {}
        next_job_config = action[NEXT_JOB_CONFIG_KEY]

        for field_name, field_info in HadoopJobExecutionConfig.model_fields.items():
            if field_name not in self.unwrapped.supported_configurations:
                continue

            if field_info.annotation is float:
                next_job_config[field_name] = float(next_job_config[field_name])
            else:
                next_job_config[field_name] = field_info.annotation(np.round(next_job_config[field_name]))

        decoded_action[NEXT_JOB_CONFIG_KEY] = next_job_config
        decoded_action[TERMINATE_ACTION_NAME] = bool(np.round(action[TERMINATE_ACTION_NAME]))

        return decoded_action
