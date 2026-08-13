from datetime import datetime
from unittest.mock import Mock

import gymnasium as gym
from dependency_injector import containers, providers
from dependency_injector.providers import Provider
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy

from elastic_reader.consts import TimePickerInputStrategy, AggregationStrategy
from elastic_reader.elastic_reader_service import ElasticReaderService
from hadoop_optimizer.common.drl_telemetry.telemetry_aggregator import TelemetryAggregator
from hadoop_optimizer.common.env_composition_config.env_builder import build_env
from hadoop_optimizer.common.env_composition_config.env_wrapper_spec import EnvWrappersParams
from hadoop_optimizer.job_config_recommender.server.drl_deployment_manager import DRLDeploymentManager
from hadoop_optimizer.drl_envs.deployment.deployment_env import OptimizerDeploymentEnv
from hadoop_optimizer.common.utils import get_drl_model
from user_input.elastic_reader_input.abstract_date_picker import TimePickerChosenInput, ReadingMode
from user_input.elastic_reader_input.time_picker_input_factory import get_time_picker_input


class DeploymentContainer(containers.DeclarativeContainer):
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

    _telemetry_aggregator: Provider[TelemetryAggregator] = providers.Singleton(
        TelemetryAggregator,
        time_windows_seconds=config.drl.state.time_windows_seconds,
        split_by=config.drl.state.split_by,
    )

    telemetry_aggregator: Provider[TelemetryAggregator] = providers.Callable(
        lambda leverage_telemetry_in_state, telemetry_aggregator, mock: telemetry_aggregator() if leverage_telemetry_in_state else mock(),
        config.drl.state.leverage_telemetry_in_state,
        _telemetry_aggregator.provider,
        providers.Factory(Mock),
    )

    elastic_reader_service: Provider[ElasticReaderService] = providers.Singleton(
        ElasticReaderService,
        consumers=providers.List(telemetry_aggregator),
        aggregation_strategy=AggregationStrategy.CALCULATE,
        time_picker_input=drl_time_picker_input,
        indices_to_read_from=config.elastic.indices_to_read_from,
    )

    base_env: Provider[gym.Env] = providers.Factory(
        OptimizerDeploymentEnv,
        telemetry_aggregator=telemetry_aggregator
    )

    env_wrappers_params: Provider[EnvWrappersParams] = providers.Factory(
        EnvWrappersParams.from_config,
        config.drl.env
    )

    deployment_env: Provider[gym.Env] = providers.Factory(
        build_env,
        base_env=base_env,
        wrappers_params=env_wrappers_params,
    )

    deployment_drl_model = providers.Callable(
        get_drl_model,
        resume_from_path=config.drl.resume_from_path,
        env=deployment_env,
        default_drl_model=None,
    )

    drl_deployment_manager: Provider[DRLDeploymentManager] = providers.Factory(
        DRLDeploymentManager,
        deployment_drl_model=deployment_drl_model,
        deployment_env=deployment_env,
    )
