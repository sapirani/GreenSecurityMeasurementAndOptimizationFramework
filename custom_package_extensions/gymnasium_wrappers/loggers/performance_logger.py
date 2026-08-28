from logging import Logger
from typing import SupportsFloat, Any, cast
import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType, ObsType, ActType, Env
from DTOs.hadoop.drl.training.training_step_results import TrainingStepResults
from drl_envs.consts import PERFORMANCE_RESULTS_KEY


class PerformanceLoggerWrapper(gym.Wrapper):
    def __init__(self, env: Env[ObsType, ActType], logger: Logger):
        super().__init__(env)
        self.logger = logger

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        performance_results = cast(TrainingStepResults, info[PERFORMANCE_RESULTS_KEY])

        self.logger.info(
            "Summarized Training Step Results",
            extra=performance_results.model_dump()
        )

        return obs, reward, terminated, truncated, info
