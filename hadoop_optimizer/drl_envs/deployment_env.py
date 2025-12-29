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
        # self.observation_space: spaces.Dict = spaces.Dict(
        #     **job_properties_state_dict_space,
        #     **hadoop_config_state_dict_space
        # )

        self.observation_space = spaces.Box(
            low=np.array([0, 0.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([300, 1.0, 1.0, 14, 14, 19, 1, 3, 3], dtype=np.float32),
            dtype=np.float32
        )

        # TODO: consider actions as delta increments (not absolute configuration)
        # TODO: DICT IS NOT SUPPORTED, should be other spaces
        self.action_space = self.action_space = spaces.MultiDiscrete([
            15,     # number_of_mappers     (1–15)
            15,     # number_of_reducers    (1–15)
            20,     # map_memory_mb bins    (250, 300,...)
            2,      # should_compress       (0/1)
            4,      # map_vcores            (1–4)
            4,      # reduce_vcores         (1–4)
            2,      # terminate             (0/1)
        ])

        self._current_hadoop_config = HadoopJobConfig()
        self._episodic_job_properties: Optional[JobProperties] = None
        self._supported_job_config_values = hadoop_config_state_dict_space.keys()
        self._last_action: Optional[Dict[str, Any]] = None
        self.step_count = 0

    def _construct_observation(
            self,
    ) -> np.ndarray:
        # TODO: return full observation (job properties, load, updated hadoop configuration)

        return flatten_observation(self._episodic_job_properties, self._current_hadoop_config)

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
        self.step_count += 1

        action_dict = decode_action(action)
        self._last_action = action_dict.copy()
        reward = 0  # there is no meaning for the reward in the deployment environment
        terminated = action_dict[TERMINATE_ACTION_NAME] == 1

        if not terminated and not truncated:
            # apply action to modify selected hadoop configuration
            # TODO: if actions are becoming deltas: start from self._current_hadoop_config,
            #   instead of the default configuration
            action_dict.pop(TERMINATE_ACTION_NAME)
            default_config = HadoopJobConfig()
            self._current_hadoop_config = default_config.model_copy(update=action_dict, deep=True)

        # TODO: CONSIDER RETURNING MORE DEBUGGING INFO, such as the current cluster load
        info.update({"current_hadoop_config": self._current_hadoop_config})

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
