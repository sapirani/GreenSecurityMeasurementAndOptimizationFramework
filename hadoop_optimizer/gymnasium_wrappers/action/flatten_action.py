import gymnasium as gym
from gymnasium import spaces

# TODO: CONSIDER USING TRANSFORM ACTION, SIMILAR TO WHAT IS GOING ON WITH FLATTEN OBSERVATION
from gymnasium.core import ObsType, WrapperActType, ActType
from gymnasium.wrappers import TransformAction


# class FlattenAction(gym.ActionWrapper):
#     """
#     Translate raw Box results (continuous values) into meaningful spaces (such as Dict space)
#     """
#     def __init__(self, env):
#         super().__init__(env)
#         self.action_space = spaces.utils.flatten_space(self.env.action_space)
#
#     def action(self, action):
#         return gym.spaces.utils.unflatten(self.env.action_space, action)
#
#     def reverse_action(self, action):
#         return gym.spaces.utils.flatten(self.env.action_space, action)
#



class FlattenAction(
    TransformAction[ObsType, WrapperActType, ActType],
    gym.utils.RecordConstructorArgs,
):
    def __init__(self, env: gym.Env[ObsType, ActType]):
        gym.utils.RecordConstructorArgs.__init__(self)
        TransformAction.__init__(
            self,
            env=env,
            func=lambda act: spaces.utils.unflatten(env.action_space, act),
            action_space=spaces.utils.flatten_space(env.action_space),
        )
