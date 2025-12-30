from typing import SupportsFloat, Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame, ActType, ObsType


class DummyEnv(gym.Env):
    """
    State:
        1. job properties
        2. cluster's load
        3. current hadoop job configuration
    """
    def __init__(self):
        super().__init__()

        # self.observation_space = spaces.Box(
        #     low=np.array([0, 0.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        #     high=np.array([300, 1.0, 1.0, 14, 14, 19, 1, 3, 3], dtype=np.float32),
        #     dtype=np.float32
        # )

        self.observation_space = spaces.Dict({
            "job_properties": spaces.Dict({
                "input_size": spaces.Box(low=0, high=300, shape=(), dtype=np.float32)
            }),
            "current_configuration": spaces.Dict({
                "number_of_mappers": spaces.Box(low=0, high=15, shape=(), dtype=np.uint16)  # TODO: SHOULD I GET RID OF THE UNIT AND PERFORM CASTING WITH NUMPY?
            }),
        })

        self.action_space = self.action_space = spaces.Dict({
            "number_of_mappers": spaces.Box(low=0, high=15, shape=(), dtype=np.uint16),
            "number_of_reducers": spaces.Box(low=50, high=70, shape=(), dtype=np.uint16)
        })

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        return {
            "job_properties": {
                "input_size": 10
            },
            "current_configuration": {
                "number_of_mappers": 3
            }
        }, {}

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        print("selected action:", action)
        return {
            "job_properties": {
                "input_size": 3
            },
            "current_configuration": {
                "number_of_mappers": 2
            }
        }, 0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return None
