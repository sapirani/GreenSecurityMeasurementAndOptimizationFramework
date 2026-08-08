from typing import SupportsFloat, Any
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers import TimeLimit

from hadoop_optimizer.drl_envs.consts import ELAPSED_STEPS_KEY, MAX_STEPS_KEY


class TimeLimitWrapper(TimeLimit):
    def __init__(self, truncated_penalty):
        super().__init__()
        self.truncated_penalty = truncated_penalty

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)

        info.update({ELAPSED_STEPS_KEY: self._elapsed_steps, MAX_STEPS_KEY: self._max_episode_steps})
        if truncated:
            reward = self.truncated_penalty
        return observation, reward, terminated, truncated, info


