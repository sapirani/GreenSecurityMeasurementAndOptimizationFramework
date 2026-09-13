from datetime import datetime
from typing import Optional, cast, Any, Dict
from typing import List
from dependency_injector.providers import Configuration
from gymnasium import spaces
from pydantic import BaseModel, ConfigDict, computed_field, Field
from stable_baselines3 import PPO
import torch.nn as nn
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy

from hadoop_optimizer.optimization_mode import OptimizationMode


class ModelInitializationConfig(BaseModel):
    pretrained_model_path: Optional[str]

    @computed_field
    @property
    def is_pretrained(self) -> bool:
        return self.pretrained_model_path is not None


class StateConfig(BaseModel):
    split_by: str
    leverage_telemetry_in_state: bool
    time_windows_seconds: List[int]


class UtilizationPolicyConfig(BaseModel):
    max_param_diff_percent: float
    min_required_similar_samples: int
    results_noise_scale: float
    similarity_temperature: float
    running_time_max_deviation_percent: float
    energy_max_deviation_percent: float


class CachedResultsConfig(BaseModel):
    search_since: datetime
    force_real_execution_probability: float
    utilization_policy: UtilizationPolicyConfig


class RewardConfig(BaseModel):
    truncated_penalty: float
    alpha: float
    beta: float
    lambda_: float
    epsilon: float
    tau: float = Field(alias="energy_importance")
    delta: float = Field(alias="running_time_importance")


class PPOAlgorithmConfig(BaseModel):
    algorithm_name: str

    observation_space: Dict[str, Any]
    action_space: Dict[str, Any]

    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    ent_coef: float
    vf_coef: float
    use_sde: bool
    sde_sample_freq: int
    gae_lambda: float

    n_envs: int
    device: str

    clip_range: Optional[float] # todo: save the schedule somehow when we start to use it
    clip_range_vf: Optional[float]
    max_grad_norm: float
    normalize_advantage: bool
    target_kl: Optional[float]

    @classmethod
    def from_model(cls, model) -> "PPOAlgorithmConfig":
        return cls(
            algorithm_name=type(model).__name__,
            observation_space=cls._serialize_space(cast(spaces.Box, model.observation_space)),
            action_space=cls._serialize_space(cast(spaces.Box, model.action_space)),
            learning_rate=model.lr_schedule(1.0),
            n_steps=model.n_steps,
            batch_size=model.batch_size,
            n_epochs=model.n_epochs,
            gamma=model.gamma,
            ent_coef=model.ent_coef,
            vf_coef=model.vf_coef,
            clip_range=cls._initial_schedule_value(model.clip_range),
            clip_range_vf=cls._initial_schedule_value(model.clip_range_vf),
            max_grad_norm=model.max_grad_norm,
            normalize_advantage=model.normalize_advantage,
            target_kl=getattr(model, "target_kl", None),
            use_sde=model.use_sde,
            sde_sample_freq=model.sde_sample_freq,
            gae_lambda=model.gae_lambda,
            n_envs=model.n_envs,
            device=str(model.device),
        )

    @staticmethod
    def _serialize_space(space: spaces.Box) -> dict[str, Any]:
        return {
            "type": type(space).__name__,
            "shape": space.shape,
            "dtype": str(space.dtype),
            "low": space.low.tolist(),
            "high": space.high.tolist(),
        }

    @staticmethod
    def _initial_schedule_value(schedule):
        if schedule is None:
            return None

        return float(schedule(1.0))


class PolicyConfig(BaseModel):
    policy_name: str
    policy_architecture: str
    actor_policy_network: str
    critic_value_network: str
    squash_output: bool
    log_std_init: float

    @classmethod
    def from_policy(cls, policy: ActorCriticPolicy) -> "PolicyConfig":
        return cls(
            policy_name=type(policy).__name__,
            policy_architecture=str(policy),
            actor_policy_network=cls._network_summary(policy.mlp_extractor.policy_net, policy.action_net),
            critic_value_network=cls._network_summary(policy.mlp_extractor.value_net, policy.value_net),
            squash_output=policy.squash_output,
            log_std_init=policy.log_std_init,
        )

    @staticmethod
    def _network_summary(hidden_module, output_module) -> str:
        layers = []

        for layer in hidden_module.children():
            if isinstance(layer, nn.Linear):
                layers.append(
                    f"Linear({layer.in_features}->{layer.out_features})"
                )
            else:
                layers.append(type(layer).__name__)

        if isinstance(output_module, nn.Linear):
            layers.append(
                f"Linear({output_module.in_features}->{output_module.out_features})"
            )
        else:
            layers.append(type(output_module).__name__)

        return " → ".join(layers)


class PPOModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: PPOAlgorithmConfig
    policy: PolicyConfig

    @classmethod
    def from_model(cls, model: PPO) -> "PPOModelConfig":

        return cls(
            model=PPOAlgorithmConfig.from_model(model),
            policy=PolicyConfig.from_policy(model.policy),
        )


class UserDefinedTrainingParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: OptimizationMode
    model_initialization: ModelInitializationConfig
    state: StateConfig
    cached_results: CachedResultsConfig
    learning_total_timestamps: int
    max_episode_steps: int
    reward: RewardConfig

    @classmethod
    def from_config(cls, config: Configuration) -> "UserDefinedTrainingParams":
        max_episode_steps = config.env.max_episode_steps()

        if config.mode() == OptimizationMode.CONTEXTUAL_BANDIT:
            max_episode_steps = 1

        return cls(
            mode=config.mode(),
            model_initialization=ModelInitializationConfig(
                pretrained_model_path=config.resume_from_path(),
            ),
            state=StateConfig(
                split_by=config.state.split_by(),
                leverage_telemetry_in_state=config.state.leverage_telemetry_in_state(),
                time_windows_seconds=config.state.time_windows_seconds(),
            ),
            cached_results=CachedResultsConfig(
                search_since=config.cached_results.search_since(),
                force_real_execution_probability=(
                    config.cached_results.force_real_execution_probability()
                ),
                utilization_policy=UtilizationPolicyConfig(
                    max_param_diff_percent=(
                        config.cached_results.utilization_policy.max_param_diff_percent()
                    ),
                    min_required_similar_samples=(
                        config.cached_results.utilization_policy.min_required_similar_samples()
                    ),
                    results_noise_scale=(
                        config.cached_results.utilization_policy.results_noise_scale()
                    ),
                    similarity_temperature=(
                        config.cached_results.utilization_policy.similarity_temperature()
                    ),
                    running_time_max_deviation_percent=(
                        config.cached_results.utilization_policy.running_time_max_deviation_percent()
                    ),
                    energy_max_deviation_percent=(
                        config.cached_results.utilization_policy.energy_max_deviation_percent()
                    ),
                ),
            ),
            learning_total_timestamps=config.learning_total_timestamps(),
            max_episode_steps=max_episode_steps,
            reward=RewardConfig(
                alpha=config.reward.alpha(),
                beta=config.reward.beta(),
                lambda_=config.reward.lambda_(),
                epsilon=config.reward.epsilon(),
                energy_importance=config.reward.tau(),
                running_time_importance=config.reward.delta(),
                truncated_penalty=config.env.truncated_penalty(),
            ),
        )


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    training_id: str
    drl_config: PPOModelConfig
    user_defined_params: UserDefinedTrainingParams
