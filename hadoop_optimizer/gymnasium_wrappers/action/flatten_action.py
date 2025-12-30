import gymnasium as gym
from gymnasium import spaces


class FlattenAction(gym.ActionWrapper):
    """
    Translate raw Box results (continuous values) into meaningful spaces (such as Dict space)
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.utils.flatten_space(self.env.action_space)
        print("flatten action space: ", self.action_space)
        print("flatten action space shape: ", self.action_space.shape)
        print("flatten state space: ", self.observation_space)
        print("flatten state space shape: ", self.observation_space.shape)

    def action(self, action):
        print("original action:", action)
        return gym.spaces.utils.unflatten(self.env.action_space, action)

    def reverse_action(self, action):
        return gym.spaces.utils.flatten(self.env.action_space, action)
