from typing import SupportsFloat, Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType


class DictLeafsAsNumpy(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    @staticmethod
    def _observation_to_ndarrays(observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively convert all values in a nested dictionary into np.array([value], dtype=np.float32)
        """
        new_dict = {}
        for k, v in observation.items():
            if isinstance(v, dict):
                new_dict[k] = DictLeafsAsNumpy._observation_to_ndarrays(v)  # recursive call
            else:
                new_dict[k] = np.array([v], dtype=np.float32)
        return new_dict

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        return self._observation_to_ndarrays(observation), reward, terminated, truncated, info


