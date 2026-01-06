from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium.core import ObsType, ActType, WrapperObsType
from gymnasium.error import ResetNeeded


class ResetEnforcer(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    def __init__(self, env: gym.Env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        self.should_reset = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        self.should_reset = False
        return super().reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.should_reset:
            raise ResetNeeded("step() called after episode has terminated. Call reset() first")

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.should_reset = terminated or truncated
        return obs, reward, terminated, truncated, info
