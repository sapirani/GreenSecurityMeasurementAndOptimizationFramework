import os
from datetime import datetime
from logging import Handler
from unittest.mock import Mock
import gymnasium as gym
from dependency_injector import containers, providers
from dependency_injector.providers import Provider
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy

from DTOs.hadoop.drl.training.cached_results_utilization_policy import CachedResultsUtilizationPolicy
from DTOs.hadoop.drl.training.episode_context import EpisodeContext
from DTOs.logging.consts import LoggerName, IndexName
from application_logging.handlers.elastic_bulk_handler import get_elastic_bulk_handler
from application_logging.handlers.elastic_handler import get_elastic_logging_handler
from application_logging.logging_utils import get_measurement_logger
from hadoop_optimizer.common.env_composition_config.wrappers_config import get_training_wrappers
from hadoop_optimizer.common.utils import get_drl_model
from hadoop_optimizer.common.drl_telemetry.telemetry_aggregator import TelemetryAggregator
from elastic_reader.consts import TimePickerInputStrategy
from elastic_reader.elastic_consumers.elastic_aggregations_logger import ElasticAggregationsLogger
from hadoop_optimizer.common.env_composition_config.env_builder import build_env
from hadoop_optimizer.common.env_composition_config.env_wrapper_spec import EnvWrappersParams
from hadoop_optimizer.drl_envs.training.reward.reward_calculator import RewardCalculator
from hadoop_optimizer.drl_envs.training.training_env import OptimizerTrainingEnv
from hadoop_optimizer.drl_envs.training.training_progress_tracker import TrainingProgressTracker
from hadoop_optimizer.job_runner.clients.cached_job_performance_evaluator_client import CachedHadoopJobPerformanceEvaluatorClient
from training_loop.callbacks.drl_training_callback import PPODebugCallback
from user_input.elastic_reader_input.abstract_date_picker import TimePickerChosenInput, ReadingMode
from user_input.elastic_reader_input.time_picker_input_factory import get_time_picker_input

MODELS_DIR_NAME = "models"
INTERMEDIATE_MODELS_NAMES_DIR_NAME = "intermediate_models"
FINAL_MODEL_PREFIX = "trained_"


class TrainingContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    drl_time_picker_input: Provider[TimePickerChosenInput] = providers.Factory(
        get_time_picker_input,
        time_picker_input_strategy=TimePickerInputStrategy.FROM_CONFIGURATION,
        preconfigured_time_input=providers.Callable(lambda: TimePickerChosenInput(
            start=datetime.now(tz=datetime.now().astimezone().tzinfo),
            end=None,
            mode=ReadingMode.REALTIME
        ))
    )

    elastic_aggregations_logger: Provider[ElasticAggregationsLogger] = providers.Singleton(
        ElasticAggregationsLogger,
        reading_mode=ReadingMode.REALTIME,
        log_extra_fields=set(EpisodeContext.model_fields.keys())
    )

    training_elastic_handler: Provider[Handler] = providers.Singleton(
        get_elastic_logging_handler,
        elastic_username=config.elastic.username,
        elastic_password=config.elastic.password,
        elastic_url=config.elastic.url,
        index_name=IndexName.DRL_TRAINING,
        ignore_exceptions=False,
    )

    training_results_logger = providers.Singleton(
        get_measurement_logger,
        logger_name=LoggerName.DRL_TRAINING,
        logger_handler=training_elastic_handler
    )

    training_debugger_elastic_handler: Provider[Handler] = providers.Singleton(
        get_elastic_bulk_handler,
        elastic_username=config.elastic.username,
        elastic_password=config.elastic.password,
        elastic_url=config.elastic.url,
        index_name=IndexName.DRL_DEBUGGING,
        ignore_exceptions=False,
    )

    training_debugging_logger = providers.Singleton(
        get_measurement_logger,
        logger_name=LoggerName.DRL_DEBUGGING,
        logger_handler=training_elastic_handler
    )

    training_progress_tracker: Provider[TrainingProgressTracker] = providers.Singleton(
        TrainingProgressTracker
    )

    reward_calculator: Provider[RewardCalculator] = providers.Factory(
        RewardCalculator,
        alpha=config.drl.reward.alpha,
        beta=config.drl.reward.beta,
        lambda_=config.drl.reward.lambda_,
        epsilon=config.drl.reward.epsilon,
        tau=config.drl.reward.tau,
        delta=config.drl.reward.delta,
        truncated_penalty=config.drl.env.truncated_penalty
    )

    _telemetry_aggregator: Provider[TelemetryAggregator] = providers.Singleton(
        TelemetryAggregator,
        time_windows_seconds=config.drl.state.time_windows_seconds,
        split_by=config.drl.state.split_by,
    )

    telemetry_aggregator: Provider[TelemetryAggregator] = providers.Callable(
        lambda leverage_telemetry_in_state, telemetry_aggregator,
               mock: telemetry_aggregator() if leverage_telemetry_in_state else mock(),
        config.drl.state.leverage_telemetry_in_state,
        _telemetry_aggregator.provider,
        providers.Factory(Mock),
    )

    training_client: Provider[CachedHadoopJobPerformanceEvaluatorClient] = providers.Factory(
        CachedHadoopJobPerformanceEvaluatorClient,
        elastic_url=config.elastic.url,
        elastic_user=config.elastic.username,
        elastic_password=config.elastic.password,
        search_since=config.drl.cached_results.search_since,
        force_real_execution_probability=config.drl.cached_results.force_real_execution_probability,
    )

    cached_results_utilization_policy: Provider[CachedResultsUtilizationPolicy] = providers.Factory(
        CachedResultsUtilizationPolicy,
        max_param_diff_percent=config.drl.cached_results.utilization_policy.max_param_diff_percent,
        min_required_similar_samples=config.drl.cached_results.utilization_policy.min_required_similar_samples,
        results_noise_scale=config.drl.cached_results.utilization_policy.results_noise_scale,
        similarity_temperature=config.drl.cached_results.utilization_policy.similarity_temperature,
        running_time_max_deviation_percent=config.drl.cached_results.utilization_policy.running_time_max_deviation_percent,
        energy_max_deviation_percent=config.drl.cached_results.utilization_policy.energy_max_deviation_percent,
    )

    base_env: Provider[gym.Env] = providers.Factory(
        OptimizerTrainingEnv,
        telemetry_aggregator=telemetry_aggregator,
        training_client=training_client,
        reward_calculator=reward_calculator,
        train_id=config.drl.train_id,
        training_progress_tracker=training_progress_tracker,
        cached_results_utilization_policy=cached_results_utilization_policy,
    )

    env_wrappers_params: Provider[EnvWrappersParams] = providers.Factory(
        EnvWrappersParams.from_config,
        config.drl.env
    )

    training_env_wrappers_params: Provider[EnvWrappersParams] = providers.Factory(
        EnvWrappersParams.from_config,
        config.drl.env.training
    )

    training_env: Provider[gym.Env] = providers.Singleton(
        build_env,
        base_env=base_env,
        basic_wrappers_params=env_wrappers_params,
        extended_wrappers_params=training_env_wrappers_params,
        extended_env_wrappers=get_training_wrappers,
    )

    default_drl_model: Provider[BaseAlgorithm] = providers.Singleton(
        PPO,
        policy=ActorCriticPolicy,
        env=training_env,
        verbose=2,
        n_steps=config.drl.algorithm.hyperparameters.n_steps,
        batch_size=config.drl.algorithm.hyperparameters.batch_size,
        n_epochs=config.drl.algorithm.hyperparameters.n_epochs,
        gamma=config.drl.algorithm.hyperparameters.gamma,
        ent_coef=config.drl.algorithm.hyperparameters.ent_coef,
        use_sde=config.drl.algorithm.hyperparameters.use_sde,
        policy_kwargs=providers.Dict(
            net_arch=config.drl.policy.hyperparameters.net_arch,
            squash_output=config.drl.policy.hyperparameters.squash_output,
            log_std_init=config.drl.policy.hyperparameters.log_std_init,
        ),
    )

    drl_training_model = providers.Singleton(
        get_drl_model,
        resume_from_path=config.drl.resume_from_path,
        env=training_env,
        default_drl_model=default_drl_model,
    )

    training_base_dir = providers.Callable(
        lambda models_base_dir, train_id: os.path.join(
            models_base_dir,
            MODELS_DIR_NAME,
            f"{datetime.now():%Y-%m-%d}_{train_id}",
        ),
        models_base_dir=config.drl.storage.models_base_dir,
        train_id=config.drl.train_id,
    )

    checkpoint_callback = providers.Factory(
        CheckpointCallback,
        save_freq=config.drl.storage.save_freq,
        save_path=providers.Callable(
            lambda training_base_dir: os.path.join(training_base_dir, INTERMEDIATE_MODELS_NAMES_DIR_NAME),
            training_base_dir=training_base_dir,
        ),
        name_prefix=providers.Callable(
            lambda model: model.__class__.__name__,
            model=drl_training_model,
        )
    )

    debug_callback = providers.Factory(
        PPODebugCallback,
        logger=training_debugging_logger,
        train_id=config.drl.train_id,
    )

    training_callback = providers.Factory(
        CallbackList,
        callbacks=providers.List(checkpoint_callback, debug_callback)
    )

    final_model_saving_paths = providers.Callable(
        lambda training_base_dir, model_name: [
            os.path.join(training_base_dir, f"{FINAL_MODEL_PREFIX}{model_name}"),
            os.path.join(training_base_dir, MODELS_DIR_NAME, model_name),
        ],
        training_base_dir=training_base_dir,
        model_name=providers.Callable(
            lambda model: model.__class__.__name__,
            model=drl_training_model,
        ),
    )
