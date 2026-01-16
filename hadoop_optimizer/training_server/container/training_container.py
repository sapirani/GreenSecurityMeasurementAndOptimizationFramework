from datetime import datetime
from unittest.mock import Mock

from dependency_injector import containers, providers
import gymnasium as gym
from dependency_injector.providers import Provider
from human_id import generate_id
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy

from elastic_consumers.elastic_aggregations_logger import ElasticAggregationsLogger
from elastic_reader.consts import TimePickerInputStrategy
from hadoop_optimizer.drl_envs.training_env import OptimizerTrainingEnv
from hadoop_optimizer.drl_telemetry.energy_tracker import EnergyTracker
from hadoop_optimizer.env_composition_config.env_builder import build_env
from hadoop_optimizer.env_composition_config.env_wrapper_spec import EnvWrappersParams
from hadoop_optimizer.reward.reward_calculator import RewardCalculator
from hadoop_optimizer.training_client.client import HadoopOptimizerTrainingClient
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
    )

    energy_tracker: Provider[EnergyTracker] = providers.Singleton(
        EnergyTracker
    )

    reward_calculator: Provider[EnergyTracker] = providers.Factory(
        RewardCalculator,
        runtime_importance_factor=config.runtime_importance_factor,
        energy_importance_factor=config.energy_importance_factor,
    )

    # todo: think about what to do with the telemetry aggregator, is it necessary?
    telemetry_aggregator = Mock()
    training_client: HadoopOptimizerTrainingClient = providers.Factory(
        HadoopOptimizerTrainingClient,
    )

    base_env: Provider[gym.Env] = providers.Factory(
        OptimizerTrainingEnv,
        telemetry_aggregator=telemetry_aggregator,
        training_client=training_client,
        energy_tracker=energy_tracker,
        reward_calculator=reward_calculator,
        training_session_id=generate_id(word_count=3)
    )

    env_wrappers_params: Provider[EnvWrappersParams] = providers.Factory(
        EnvWrappersParams.from_config,
        config
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
        n_steps=2,
        n_epochs=1,
        batch_size=2,
    )
