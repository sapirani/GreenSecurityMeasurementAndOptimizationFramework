from abc import ABC, abstractmethod
from typing import SupportsFloat, Any, Optional, Dict

from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_properties import JobProperties
from hadoop_optimizer.drl_envs.consts import TERMINATE_ACTION_NAME, CURRENT_JOB_CONFIG_KEY, NEXT_JOB_CONFIG_KEY, \
    JOB_PROPERTIES_KEY, DEFAULT_JOB_CONFIG_KEY
from hadoop_optimizer.drl_telemetry.telemetry_aggregator import TelemetryAggregator
import gymnasium as gym
from gymnasium.core import RenderFrame, ActType, ObsType
import numpy as np
from gymnasium import spaces


class AbstractOptimizerEnvInterface(gym.Env, ABC):
    """
    This environment defines:
        1. How state space looks like (what are the allowed values?
        2. How action space looks like
        3. What is the initial state (reset function)
        4. What are the next state and reward, given an action is taken (step function)

    State is generally composed of:
        1. job properties
        2. cluster's load
        3. current hadoop job configuration

    Action defines whether to:
        1. stop the episode (I.e., we found the optimal job configuration)
        2. keep trying another job configuration, which the action itself defines

    The reward pushes the DRL towards the optimal job configuration
        (in terms of minimal running time and energy consumption), while performing minimal number of steps.
    """

    def __init__(self, telemetry_aggregator: TelemetryAggregator):
        super().__init__()
        self.render_mode = "human"  # must be defined for successful rendering in training
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

        self.telemetry_aggregator = telemetry_aggregator  # TODO: LEVERAGE TELEMETRY MANAGER INSIDE THE OBSERVATION SPACE
        # TODO: THINK ABOUT WHAT TO DO WITH TELEMETRY IN THE TRAINING ENV
        #  (AS IT SHOULD BE THE SAME ACROSS THE EPISODE, BUT EACH STEP AFFECTS IT BY ITSELF)
        self.episodic_telemetry = None

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

    @staticmethod
    def _get_next_execution_config(action: ActType) -> HadoopJobExecutionConfig:
        """ Apply action to modify next hadoop configuration """
        # TODO: if actions are becoming deltas: start from self._current_hadoop_config,
        #   instead of the default configuration
        default_config = HadoopJobExecutionConfig()
        return default_config.model_copy(
            update=action[NEXT_JOB_CONFIG_KEY],
            deep=True,
        )

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0

        self._episodic_job_properties = self._init_episodic_job(options)

        # TODO: CONSIDER RETURNING DEBUGGING INFO, such as the current cluster load
        self._current_hadoop_config = HadoopJobExecutionConfig()
        self.episodic_telemetry = self.telemetry_aggregator.get_telemetry()
        info = {DEFAULT_JOB_CONFIG_KEY: True}
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
            self._current_hadoop_config = self._get_next_execution_config(action)

        self._compute_reward(self._current_hadoop_config, terminated, truncated)

        # TODO: CONSIDER RETURNING MORE DEBUGGING INFO, such as the current cluster load
        info.update({CURRENT_JOB_CONFIG_KEY: self._current_hadoop_config.model_dump()})

        return self._construct_observation(), reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        print(f"****************** Current Step: {self.step_count} ******************")

        self._custom_rendering()

        print("Episodic Job Properties:")
        print(self._episodic_job_properties)
        print()

        print("Episodic Telemetry:")
        print(self.episodic_telemetry.to_string())
        print()

        print("Selected Action:")
        print(self._last_action)
        print()

        print(f"------------ Current Hadoop Config (step {self.step_count}) ------------")
        print(self._current_hadoop_config)
        print()
        print()

        return None

    @abstractmethod
    def _init_episodic_job(self, options: dict[str, Any] | None) -> JobProperties:
        """
        This function performs all required initialization related to the episodic job.
        Note: seed can be accessed through self._np_random_seed
        :param options: additional parameters that are passed into the "reset" function of the environment
        """
        pass

    @abstractmethod
    def _compute_reward(self, job_config: HadoopJobExecutionConfig, terminated: bool, truncated: bool) -> float:
        pass

    @abstractmethod
    def _custom_rendering(self):
        pass
