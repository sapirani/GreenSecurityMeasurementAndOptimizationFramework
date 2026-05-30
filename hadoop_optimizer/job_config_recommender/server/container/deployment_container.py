from datetime import datetime

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

    telemetry_aggregator: Provider[TelemetryAggregator] = providers.Singleton(
        TelemetryAggregator,
        time_windows_seconds=config.drl.state.time_windows_seconds,
        split_by=config.drl.state.split_by,
    )

    elastic_reader_service: Provider[ElasticReaderService] = providers.Singleton(
        aggregation_strategy=AggregationStrategy.CALCULATE,
        time_picker_input=drl_time_picker_input,
        consumers=[telemetry_aggregator],
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

    # TODO: LOAD THE BEST MODEL INSTEAD OF INITIALIZING A NEW MODEL HERE, FOR EXAMPLE: PPO.load(<path>)
    deployment_drl_model: Provider[BaseAlgorithm] = providers.Singleton(
        PPO,
        policy=ActorCriticPolicy,
        env=deployment_env,
        # TODO: REMOVE WHEN WE HAVE A REAL MODEL, IT INCREASES THE PREDICTION VARIANCE
        policy_kwargs=dict(log_std_init=0.8)
    )

    drl_deployment_manager: Provider[DRLDeploymentManager] = providers.Factory(
        DRLDeploymentManager,
        deployment_drl_model=deployment_drl_model,
        deployment_env=deployment_env,
    )
