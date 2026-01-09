from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium.core import ObsType, ActType

from hadoop_optimizer.erros import OutOfBoundObservation


class EnforceObservationBounds(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if not self.env.observation_space.contains(obs):
            raise OutOfBoundObservation(observation=obs, space=self.observation_space)
        return obs, info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self.env.observation_space.contains(obs):    # TODO: AVOID NUMERIC ERRORS
            raise OutOfBoundObservation(observation=obs, space=self.observation_space)
        return obs, reward, terminated, truncated, info

