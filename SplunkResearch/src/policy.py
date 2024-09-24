import torch as th
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet
from stable_baselines3.common.distributions import Distribution
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
import gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.ppo.policies import MlpPolicy

from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import Tensor, nn
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    DirichletDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from sb3_contrib.common.recurrent.type_aliases import RNNStates

class NormalizedDiagGaussianDistribution(DiagGaussianDistribution):
    def __init__(self, action_dim):
        super(NormalizedDiagGaussianDistribution, self).__init__(action_dim)
        self.action_dim = action_dim
        
    def get_actions(self, deterministic: bool = False) -> Tensor:
         distribution = super().get_actions(deterministic)
         return distribution/sum(distribution)
        
    

class DirichletDistribution(Distribution):
    def __init__(self, action_dim):
        super(DirichletDistribution, self).__init__()
        self.action_dim = action_dim
        self.eps = 1e-7  # Small epsilon to avoid zeros

    def proba_distribution_net(self, latent_dim):
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(self, action_logits):
        # Ensure positive concentration parameters
        concentration = th.exp(action_logits) + self.eps
        self.distribution = Dirichlet(concentration=concentration)
        return self

    def log_prob(self, actions):
        # Clamp actions to be within the valid range
        actions_clamped = th.clamp(actions, self.eps, 1 - self.eps)
        actions_clamped = actions_clamped / actions_clamped.sum(dim=-1, keepdim=True)
        return self.distribution.log_prob(actions_clamped)

    def entropy(self):
        return self.distribution.entropy()

    def sample(self):
        return self.distribution.sample()

    def mode(self):
        concentration = self.distribution.concentration
        return (concentration - 1).clamp(min=0) / (concentration.sum() - self.action_dim).clamp(min=self.eps)

    def actions_from_params(self, action_logits, deterministic=False):
        self.proba_distribution(action_logits)
        if deterministic:
            return self.mode(), None
        return self.sample(), None

    def log_prob_from_params(self, action_logits):
        actions = self.actions_from_params(action_logits)[0]
        log_prob = self.log_prob(actions)
        return log_prob, actions
    
class CustomLSTMActor(MlpLstmPolicy):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, lr_schedule: Schedule, net_arch: Optional[List[int]] = None,
                    activation_fn: Type[nn.Module] = nn.ReLU, device: Union[th.device, str] = 'auto', use_sde: bool = False,
                    log_std_init: float = 0.0, sde_net_arch: Optional[List[int]] = None, use_expln: bool = False,
                    clip_mean_action: bool = True, target_entropy: Optional[float] = None, use_combined_action_model: bool = False,
                    **kwargs):
        super(CustomLSTMActor, self).__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, device, use_sde, log_std_init,
                                        sde_net_arch, use_expln, clip_mean_action, target_entropy, use_combined_action_model, **kwargs)
   
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        elif isinstance(self.action_dist, DirichletDistribution):
            # For Dirichlet, mean_actions are used as the concentration parameters
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")
        
class CustomActor(MlpPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(CustomActor, self).__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, ortho_init, use_sde, log_std_init,
                                        full_std, use_expln, squash_output, features_extractor_class, features_extractor_kwargs, share_features_extractor,
                                        normalize_images, optimizer_class, optimizer_kwargs)
   
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        elif isinstance(self.action_dist, DirichletDistribution):
            # For Dirichlet, mean_actions are used as the concentration parameters
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")