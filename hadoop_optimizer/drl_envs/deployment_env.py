from typing import SupportsFloat, Any, Optional, Dict
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame, ActType, ObsType
from pydantic import ValidationError
from hadoop_optimizer.DTOs.hadoop_job_execution_config import HadoopJobExecutionConfig
from hadoop_optimizer.DTOs.job_properties import JobProperties
from hadoop_optimizer.drl_envs.consts import TERMINATE_ACTION_NAME, CURRENT_JOB_CONFIG_KEY, NEXT_JOB_CONFIG_KEY, \
    JOB_PROPERTIES_KEY


class OptimizerDeploymentEnv(gym.Env):
    """
    State:
        1. job properties
        2. cluster's load
        3. current hadoop job configuration
    """
    def __init__(self):
        super().__init__()
        # TODO: SUPPORT CURRENT CLUSTER LOAD
        self.observation_space: spaces.Dict = spaces.Dict({
            JOB_PROPERTIES_KEY: self.job_properties_space,
            CURRENT_JOB_CONFIG_KEY: self.job_config_space,
        })

        # TODO: consider actions as delta increments (not absolute configuration)
        self.action_space = spaces.Dict({
            NEXT_JOB_CONFIG_KEY: self.job_config_space,
            TERMINATE_ACTION_NAME: spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
        })

        self._current_hadoop_config = HadoopJobExecutionConfig()
        self._episodic_job_properties: Optional[JobProperties] = None
        self._last_action: Optional[Dict[str, Any]] = None
        self.step_count = 0

    @property
    def job_config_space(self):
        # TODO: extend this implementation with all the flags:
        return spaces.Dict({
            "number_of_mappers": spaces.Box(low=1, high=15, shape=(), dtype=np.float32),
            "number_of_reducers": spaces.Box(low=1, high=15, shape=(), dtype=np.float32),
            "map_memory_mb": spaces.Box(low=100, high=1500, shape=(), dtype=np.float32),
            "should_compress": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
            "map_vcores": spaces.Box(low=1, high=4, shape=(), dtype=np.float32),
            "reduce_vcores": spaces.Box(low=1, high=4, shape=(), dtype=np.float32),
        })

    @property
    def job_properties_space(self):
        return spaces.Dict({
            "input_size_gb": spaces.Box(low=0, high=300, shape=(), dtype=np.float32),
            "cpu_bound_scale": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
            "io_bound_scale": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
        })

    @property
    def supported_configurations(self):
        return set(self.job_config_space.keys())

    def _construct_observation(
            self,
    ) -> Dict[str, Any]:
        # TODO: return full observation (job properties, load, updated hadoop configuration)
        return {
            JOB_PROPERTIES_KEY: self._episodic_job_properties.model_dump(),
            CURRENT_JOB_CONFIG_KEY: self._current_hadoop_config.model_dump(include=self.supported_configurations),
        }

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
        self._current_hadoop_config = HadoopJobExecutionConfig()
        info = {"default_config": True}
        return self._construct_observation(), info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._current_hadoop_config is None:
            raise RuntimeError("Environment must be reset before calling the step function")

        truncated = False
        info = {}
        reward = 0  # there is no meaning for the reward in the deployment environment
        self.step_count += 1

        self._last_action = action.copy()

        terminated = action[TERMINATE_ACTION_NAME]
        if not terminated and not truncated:
            # apply action to modify selected hadoop configuration
            # TODO: if actions are becoming deltas: start from self._current_hadoop_config,
            #   instead of the default configuration
            default_config = HadoopJobExecutionConfig()
            self._current_hadoop_config = default_config.model_copy(
                update=action[NEXT_JOB_CONFIG_KEY], deep=True
            )
        # TODO: CONSIDER RETURNING MORE DEBUGGING INFO, such as the current cluster load
        info.update({CURRENT_JOB_CONFIG_KEY: self._current_hadoop_config.model_dump()})

        return self._construct_observation(), reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        print(f"****************** Current Step: {self.step_count} ******************")

        print("Episodic Job Properties:")
        print(self._episodic_job_properties)
        print()

        print("Selected Action:")
        print(self._last_action)
        print()

        print(f"------------ Current Hadoop Config (step {self.step_count}) ------------")
        print(self._current_hadoop_config)
        print()
        print()

        return None
