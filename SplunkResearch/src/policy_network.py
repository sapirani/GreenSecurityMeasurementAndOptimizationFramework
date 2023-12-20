from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
import torch as th

# Custom policy
class CustomPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, obs):
        # Get DQN policy output
        q_values = super().forward(obs)
        
        # Create distribution 
        action_dim = q_values.shape[1]
        print(action_dim)
        dist = CategoricalDistribution(action_dim)
        
        # Set distribution probabilities 
        dist.proba = th.softmax(q_values, dim=1)
        
        # Sample action
        action = dist.sample()
        print(action)
        return action