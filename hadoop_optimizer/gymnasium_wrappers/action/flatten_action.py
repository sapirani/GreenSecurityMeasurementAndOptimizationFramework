import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, WrapperActType, ActType
from gymnasium.wrappers import TransformAction


class FlattenAction(
    TransformAction[ObsType, WrapperActType, ActType],
    gym.utils.RecordConstructorArgs,
):
    """
    Similar implementation to Gymnasium's FlattenObservation
    Another possible implementation: https://gist.github.com/colllin/1172e042edf267d5ec667fa9802673cf
    """
    def __init__(self, env: gym.Env[ObsType, ActType]):
        gym.utils.RecordConstructorArgs.__init__(self)
        TransformAction.__init__(
            self,
            env=env,
            func=lambda act: spaces.utils.unflatten(env.action_space, act),
            action_space=spaces.utils.flatten_space(env.action_space),
        )
