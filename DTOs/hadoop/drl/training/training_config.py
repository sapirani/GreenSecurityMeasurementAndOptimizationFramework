from datetime import datetime
from typing import Optional

from typing import List

from dependency_injector.providers import Configuration
from pydantic import BaseModel, ConfigDict, computed_field

from hadoop_optimizer.optimization_mode import OptimizationMode


class ModelInitializationConfig(BaseModel):
    pretrained_model_path: Optional[str]

    @computed_field
    @property
    def is_pretrained(self) -> bool:
        return self.pretrained_model_path is not None


class EnvironmentConfig(BaseModel):
    max_episode_steps: int
    truncated_penalty: float


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
    alpha: float
    beta: float
    lambda_: float
    epsilon: float
    tau: float
    delta: float


class AlgorithmConfig(BaseModel):
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    ent_coef: float
    use_sde: bool
    sde_sample_freq: int


class PolicyConfig(BaseModel):
    net_arch: List[int]
    squash_output: bool
    log_std_init: float


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    train_id: str
    mode: str
    model_initialization: ModelInitializationConfig
    environment: EnvironmentConfig
    state: StateConfig
    cached_results: CachedResultsConfig
    learning_total_timestamps: int
    reward: RewardConfig
    algorithm: AlgorithmConfig
    policy: PolicyConfig

    @classmethod
    def from_config(cls, config: Configuration) -> "TrainingConfig":
        gamma = config.algorithm.hyperparameters.gamma()
        max_episode_steps = config.env.max_episode_steps()

        if config.mode() == OptimizationMode.CONTEXTUAL_BANDIT:
            gamma = 0
            max_episode_steps = 1

        return cls(
            train_id=config.train_id(),
            mode=config.mode(),
            model_initialization=ModelInitializationConfig(
                pretrained_model_path=config.resume_from_path(),
            ),
            environment=EnvironmentConfig(
                max_episode_steps=max_episode_steps,
                truncated_penalty=config.env.truncated_penalty(),
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
            reward=RewardConfig(
                alpha=config.reward.alpha(),
                beta=config.reward.beta(),
                lambda_=config.reward.lambda_(),
                epsilon=config.reward.epsilon(),
                tau=config.reward.tau(),
                delta=config.reward.delta(),
            ),
            algorithm=AlgorithmConfig(
                n_steps=config.algorithm.hyperparameters.n_steps(),
                batch_size=config.algorithm.hyperparameters.batch_size(),
                n_epochs=config.algorithm.hyperparameters.n_epochs(),
                gamma=gamma,
                ent_coef=config.algorithm.hyperparameters.ent_coef(),
                use_sde=config.algorithm.hyperparameters.use_sde(),
                sde_sample_freq=config.algorithm.hyperparameters.sde_sample_freq(),
            ),
            policy=PolicyConfig(
                net_arch=config.policy.hyperparameters.net_arch(),
                squash_output=config.policy.hyperparameters.squash_output(),
                log_std_init=config.policy.hyperparameters.log_std_init(),
            ),
        )
