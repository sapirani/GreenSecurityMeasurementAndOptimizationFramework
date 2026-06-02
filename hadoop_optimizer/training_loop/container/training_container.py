from datetime import datetime
from logging import Handler
from unittest.mock import Mock

import gymnasium as gym
from dependency_injector import containers, providers
from dependency_injector.providers import Provider
from human_id import generate_id
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy

from DTOs.hadoop.drl.training.episode_context import EpisodeContext
from DTOs.logging.consts import LoggerName, IndexName
from application_logging.handlers.elastic_handler import get_elastic_logging_handler
from application_logging.logging_utils import get_measurement_logger
from elastic_reader.consts import TimePickerInputStrategy
from elastic_reader.elastic_consumers.elastic_aggregations_logger import ElasticAggregationsLogger
from hadoop_optimizer.common.env_composition_config.env_builder import build_env
from hadoop_optimizer.common.env_composition_config.env_wrapper_spec import EnvWrappersParams
from hadoop_optimizer.drl_envs.training.reward.reward_calculator import RewardCalculator
from hadoop_optimizer.drl_envs.training.training_env import OptimizerTrainingEnv
from hadoop_optimizer.drl_envs.training.training_progress_tracker import TrainingProgressTracker
from hadoop_optimizer.job_runner.clients.cached_job_performance_evaluator_client import CachedHadoopJobPerformanceEvaluatorClient
from user_input.elastic_reader_input.abstract_date_picker import TimePickerChosenInput, ReadingMode
from user_input.elastic_reader_input.time_picker_input_factory import get_time_picker_input


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
    )

    # todo: think about what to do with the telemetry aggregator, is it necessary?
    telemetry_aggregator = Mock()
    training_client: CachedHadoopJobPerformanceEvaluatorClient = providers.Factory(
        CachedHadoopJobPerformanceEvaluatorClient,
    )

    base_env: Provider[gym.Env] = providers.Factory(
        OptimizerTrainingEnv,
        telemetry_aggregator=telemetry_aggregator,
        training_client=training_client,
        reward_calculator=reward_calculator,
        train_id=generate_id(word_count=3),
        training_results_logger=training_results_logger,
        training_progress_tracker=training_progress_tracker,
        max_param_diff_percent=config.drl.env.max_param_diff_percent,
    )

    env_wrappers_params: Provider[EnvWrappersParams] = providers.Factory(
        EnvWrappersParams.from_config,
        config.drl.env
    )

    training_env: Provider[gym.Env] = providers.Factory(
        build_env,
        base_env=base_env,
        wrappers_params=env_wrappers_params,
    )

    training_drl_model: Provider[BaseAlgorithm] = providers.Singleton(
        PPO,
        policy=ActorCriticPolicy,
        env=training_env,
        verbose=2,
        # TODO: REFINE THE FOLLOWING PARAMETERS:
        n_steps=128,
        batch_size=32,
        n_epochs=10,
        gamma=1,
        ent_coef=0.01,  # encourage exploration
        policy_kwargs=dict(
            net_arch=[128, 128]
        ),
    )
