from typing import SupportsFloat, Any, cast
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers import TimeLimit
import gymnasium as gym

from DTOs.hadoop.drl.training.training_step_results import TrainingStepResults
from hadoop_optimizer.drl_envs.consts import ELAPSED_STEPS_KEY, MAX_STEPS_KEY, PERFORMANCE_RESULTS_KEY


class TimeLimitWrapper(TimeLimit):
    def __init__(self, env: gym.Env, max_episode_steps: int,  truncated_penalty: float):
        super().__init__(env, max_episode_steps)
        self.truncated_penalty = truncated_penalty

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)

        info.update({ELAPSED_STEPS_KEY: self._elapsed_steps, MAX_STEPS_KEY: self._max_episode_steps})
        if truncated:
            reward = self.truncated_penalty

            if info[PERFORMANCE_RESULTS_KEY] is not None:
                results = cast(TrainingStepResults, info[PERFORMANCE_RESULTS_KEY])
                current_step_reward = cast(float, results.step_reward)
                current_cumulative_reward = cast(float, results.cumulative_reward)
                preceding_cumulative_reward = current_cumulative_reward - current_step_reward
                results = results.model_copy(
                    update={
                        "step_reward": self.truncated_penalty,
                        "cumulative_reward": preceding_cumulative_reward + self.truncated_penalty
                    }
                )
                info[PERFORMANCE_RESULTS_KEY] = results

        return observation, reward, terminated, truncated, info


