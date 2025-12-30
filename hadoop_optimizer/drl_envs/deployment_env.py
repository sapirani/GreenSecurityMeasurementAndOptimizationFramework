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
    job_properties_as_gymnasium_dict_space, dict_to_ndarrays, \
    decode_action_types


class OptimizerDeploymentEnv(gym.Env):
    """
    State:
        1. job properties
        2. cluster's load
        3. current hadoop job configuration
    """
    def __init__(self):
        super().__init__()
        job_properties_state_dict_space = job_properties_as_gymnasium_dict_space()
        hadoop_config_state_dict_space = hadoop_config_as_gymnasium_dict_space()
        # TODO: SUPPORT CURRENT CLUSTER LOAD
        self.observation_space: spaces.Dict = spaces.Dict({
            "job_properties": job_properties_state_dict_space,
            "current_job_config": hadoop_config_state_dict_space,
        })

        # TODO: consider actions as delta increments (not absolute configuration)
        self.action_space = spaces.Dict({
            "current_job_config": hadoop_config_state_dict_space,
            TERMINATE_ACTION_NAME: spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
        })

        self._current_hadoop_config = HadoopJobConfig()
        self._episodic_job_properties: Optional[JobProperties] = None
        self._supported_job_config_values = set(hadoop_config_state_dict_space.keys())
        self._last_action: Optional[Dict[str, Any]] = None
        self.step_count = 0

    def _construct_observation(
            self,
    ) -> Dict[str, Any]:
        # TODO: return full observation (job properties, load, updated hadoop configuration)

        raw_observation = {
            "job_properties": self._episodic_job_properties.model_dump(),
            "current_job_config": self._current_hadoop_config.model_dump(include=self._supported_job_config_values),
        }

        return dict_to_ndarrays(raw_observation)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        # TODO: CONSIDER ADDING A DEPLOYMENT PARAMETER (BOOLEAN) INSIDE THE CONSTRUCTOR AND
        #   ACT DIFFERENTLY IN THIS FUNCTION ACCORDINGLY
        if not options:
            raise ValueError("Expected to retrieve the job properties on reset")

        self.step_count = 0

        try:
            self._episodic_job_properties = JobProperties.model_validate(options)
        except ValidationError as e:
            raise ValueError("Received unexpected job properties") from e

        # TODO: CONSIDER RETURNING DEBUGGING INFO, such as the current cluster load
        self._current_hadoop_config = HadoopJobConfig()
        info = {"default_config": True}
        return self._construct_observation(), info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._current_hadoop_config is None:
            raise RuntimeError("Environment must be reset before calling the step function")

        truncated = False
        info = {}
        reward = 0  # there is no meaning for the reward in the deployment environment
        self.step_count += 1

        decoded_action = decode_action_types(action, self._supported_job_config_values)
        self._last_action = decoded_action.copy()

        terminated = decoded_action[TERMINATE_ACTION_NAME]     # TODO: FIND WHY ALL ELEMENTS ARE NDARRAYS
        if not terminated and not truncated:
            # apply action to modify selected hadoop configuration
            # TODO: if actions are becoming deltas: start from self._current_hadoop_config,
            #   instead of the default configuration
            default_config = HadoopJobConfig()
            self._current_hadoop_config = default_config.model_copy(
                update=decoded_action["current_job_config"], deep=True
            )

        # TODO: CONSIDER RETURNING MORE DEBUGGING INFO, such as the current cluster load
        info.update({"current_hadoop_config": self._current_hadoop_config.model_dump()})

        return self._construct_observation(), reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        print(f"****************** Current Step: {self.step_count} ******************")

        print("Episodic Job Properties:")
        print(self._current_hadoop_config)

        print("Selected Action:")
        print(self._last_action)

        print(f"------------ Current Hadoop Config (step {self.step_count}) ------------")
        print(self._current_hadoop_config)
        print()
        print()

        return None
