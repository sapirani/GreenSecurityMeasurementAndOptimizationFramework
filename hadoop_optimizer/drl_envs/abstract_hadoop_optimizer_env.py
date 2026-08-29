from abc import ABC, abstractmethod
from datetime import datetime
from typing import SupportsFloat, Any, Optional, Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame, ActType, ObsType

from DTOs.hadoop.drl.job_properties import JobProperties
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from hadoop_optimizer.common.drl_telemetry.telemetry_aggregator import TelemetryAggregator
from hadoop_optimizer.drl_envs.consts import CURRENT_JOB_CONFIG_KEY, JOB_PROPERTIES_KEY, DEFAULT_JOB_CONFIG_KEY, \
    RenderMode
from hadoop_optimizer.optimization_mode.abstract_optimization_mode import AbstractOptimizationMode


# TODO: THINK ABOUT THE CORRECT TREATMENT OPTION FOR THE LAST STEP (TO BEST ALIGN WITH THE MARKOV PROPERTY):
#   1. RIGHT NOW - IT IS TREATED AS STOP SIGNAL AND THE ACTUAL CONFIGURATION IS THE ONE WE RECEIVED IN THE PREVIOUS STEP
#       THE REWARD OF THE LAST STEP IS COMPUTED BASED ON WHAT WE SAW IN THE PREVIOUS STATE (BREAKS THE MARKOV PROPERTY)
#   2. WHAT WE WANT TO DO: TREAT THE CONFIGURATION WHERE THE DRL DECIDES TO TERMINATE AS THE FINAL CONFIGURATION
#        THIS -100 REWARD IF WE ARE SEEING 'TERMINATE' RIGHT AWAY SHOULD BE DELETED
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

    def __init__(self, telemetry_aggregator: TelemetryAggregator, optimization_mode: AbstractOptimizationMode):
        super().__init__()
        self.render_mode = RenderMode.HUMAN
        self.optimization_mode = optimization_mode
        # TODO: SUPPORT CURRENT CLUSTER LOAD
        self.observation_space: spaces.Dict = self.optimization_mode.get_observation_space()

        # TODO: consider actions as delta increments (not absolute configuration)
        self.action_space = self.optimization_mode.get_action_space()
        self.telemetry_aggregator = telemetry_aggregator  # TODO: LEVERAGE TELEMETRY MANAGER INSIDE THE OBSERVATION SPACE
        # TODO: THINK ABOUT WHAT TO DO WITH TELEMETRY IN THE TRAINING ENV
        #  (AS IT SHOULD BE THE SAME ACROSS THE EPISODE, BUT EACH STEP AFFECTS IT BY ITSELF)
        self.episodic_telemetry = None

        self._current_hadoop_config = HadoopJobExecutionConfig()
        self._episodic_job_properties: Optional[JobProperties] = None
        self._last_action: Optional[Dict[str, Any]] = None
        self.step_count = 0
        self.episode_counter = 0
        self._cumulative_reward = 0

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        self.episode_counter += 1
        self._cumulative_reward = 0

        self._episodic_job_properties, info = self._init_episodic_job(options)
        assert self._episodic_job_properties is not None

        # TODO: CONSIDER RETURNING DEBUGGING INFO, such as the current cluster load
        self._current_hadoop_config = HadoopJobExecutionConfig()
        self.episodic_telemetry = self.telemetry_aggregator.get_telemetry()
        info.update({DEFAULT_JOB_CONFIG_KEY: True})

        observation = self.optimization_mode.construct_observation(
            self._episodic_job_properties,
            self._current_hadoop_config
        )
        return observation, info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._current_hadoop_config is None or self._episodic_job_properties is None:
            raise RuntimeError("Environment must be reset before calling the step function")

        truncated = False
        reward = 0  # there is no meaning for the reward in the deployment environment
        self.step_count += 1

        self._last_action = action.copy()
        terminated = self.optimization_mode.should_terminate(action)
        self._current_hadoop_config = self.optimization_mode.parse_action_to_config(action)

        self._extra_step_init()
        step_reward, info = self._compute_reward(
            self._current_hadoop_config, terminated=terminated, truncated=truncated
        )
        self._cumulative_reward += step_reward

        # TODO: CONSIDER RETURNING MORE DEBUGGING INFO, such as the current cluster load
        info.update({CURRENT_JOB_CONFIG_KEY: self._current_hadoop_config})

        observation = self.optimization_mode.construct_observation(
            self._episodic_job_properties,
            self._current_hadoop_config
        )
        return observation, reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        print(f"****************** "
              f"Current Episode: {self.episode_counter}, Current Step: {self.step_count} "
              f"(at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}) "
              f"******************")

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
    def _init_episodic_job(self, options: dict[str, Any] | None) -> Tuple[JobProperties, Dict[str, Any]]:
        """
        This function performs all required initialization related to the episodic job.
        Note: seed can be accessed through self._np_random_seed
        :param options: additional parameters that are passed into the "reset" function of the environment
        :return: job properties and extra information (if needed, otherwise an empty dictionary is returned).
        """
        pass

    @abstractmethod
    def _extra_step_init(self):
        """
        Additional initiation that the subclass can implement
        """
        pass

    @abstractmethod
    def _compute_reward(self, job_config: HadoopJobExecutionConfig, *, terminated: bool, truncated: bool) -> float:
        """
        This function is applied whenever a step is performed
        :param job_config: the current job config to run, measure its performance and compute reward accordingly
        :param terminated: Whether the DRL agent decided to finish the current episode
        :param truncated: Whether the truncation condition outside the scope of the MDP is satisfied (e.g., timelimit)
        :return: the step reward
        """
        pass

    @abstractmethod
    def _custom_rendering(self):
        """
        Additional debugging information that could be printed by the subclass after performing each step
        """
        pass
