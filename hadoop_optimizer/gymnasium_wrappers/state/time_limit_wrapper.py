from typing import SupportsFloat, Any
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers import TimeLimit


class TimeLimitWrapper(TimeLimit):
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)

        info.update({"elapsed_steps": self._elapsed_steps, "max_steps": self._max_episode_steps})
        return observation, reward, terminated, truncated, info


