from typing import SupportsFloat, Any, cast, Optional, Dict

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame, ActType, ObsType
from pydantic import ValidationError

from hadoop_optimizer.DTOs.hadoop_job_config import HadoopJobConfig
from hadoop_optimizer.DTOs.job_properties import JobProperties
from hadoop_optimizer.drl_envs.consts import TERMINATE_ACTION_NAME
from hadoop_optimizer.drl_envs.spaces_utils import hadoop_config_as_gymnasium_dict_space, \
    job_properties_as_gymnasium_dict_space, add_termination_action


class OptimizerDeploymentEnv(gym.Env):
    """
    State:
        1. job properties
        2. cluster's load
        3. current hadoop job configuration
    """
    def __init__(self):
        job_properties_state_dict_space = job_properties_as_gymnasium_dict_space()
        hadoop_config_state_dict_space = hadoop_config_as_gymnasium_dict_space()
        # TODO: SUPPORT CURRENT CLUSTER LOAD
        self.observation_space: spaces.Dict = spaces.Dict(
            **job_properties_state_dict_space,
            **hadoop_config_state_dict_space
        )

        # TODO: consider actions as delta increments (not absolute configuration)
        self.action_space = add_termination_action(hadoop_config_as_gymnasium_dict_space())

        self._current_hadoop_config = HadoopJobConfig()
        self._episodic_job_properties: Optional[JobProperties] = None

    def _construct_observation(
            self,
    ) -> Dict[str, Any]:
        # TODO: return full observation (job properties, load, updated hadoop configuration)

        configuration_as_dict = self._current_hadoop_config.model_dump()

        # take the configuration fields that are relevant to the observation space
        job_configuration_as_state = {key: configuration_as_dict[key] for key in self.observation_space.keys()}

        return {**self._episodic_job_properties.model_dump(), **job_configuration_as_state}

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

        truncated = False   # TODO: SUPPORT MAXIMUM NUMBER OF UPDATES
                                # (end the episode prematurely before a terminal state is reached)
        reward = 0  # there is no meaning for the reward upon deployment anymore
        terminated = action[TERMINATE_ACTION_NAME] == 1

        if not terminated:
            # apply action to modify selected hadoop configuration
            # TODO: if actions are becoming deltas: start from self._current_hadoop_config,
            #   instead of the default configuration
            default_config = HadoopJobConfig()
            self._current_hadoop_config = default_config.model_copy(update=action, deep=True)

        # TODO: CONSIDER RETURNING DEBUGGING INFO, such as the current cluster load
        info = {"current_hadoop_config": self._current_hadoop_config}

        return self._construct_observation(), reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        print("****************** Episodic Job Properties ******************")
        print(self._current_hadoop_config)

        print("****************** Current Hadoop Config ******************")
        print(self._current_hadoop_config)
        print()
        print()

        return None
