from unittest.mock import Mock

from dependency_injector import containers, providers
import gymnasium as gym
from dependency_injector.providers import Provider
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy

from hadoop_optimizer.drl_envs.training_env import OptimizerTrainingEnv
from hadoop_optimizer.env_composition_config.env_builder import build_env
from hadoop_optimizer.env_composition_config.env_wrapper_spec import EnvWrappersParams
from hadoop_optimizer.training_client.client import HadoopOptimizerTrainingClient


class TrainingContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    # TODO: IS IT NECESSARY IN TRAINING?
    # drl_time_picker_input: Provider[TimePickerChosenInput] = providers.Factory(
    #     get_time_picker_input,
    #     time_picker_input_strategy=TimePickerInputStrategy.FROM_CONFIGURATION,
    #     preconfigured_time_input=providers.Callable(lambda: TimePickerChosenInput(
    #         start=datetime.now(tz=datetime.now().astimezone().tzinfo),
    #         end=None,
    #         mode=ReadingMode.REALTIME
    #     ))
    # )

    # drl_telemetry_manager: Provider[DRLTelemetryManager] = providers.Singleton(
    #     DRLTelemetryManager,
    #     time_windows_seconds=config.drl_state.time_windows_seconds,
    #     split_by=config.drl_state.split_by,
    # )

    # todo: think about what to do with the drl telemetry manager
    drl_telemetry_manager = Mock()
    training_client: HadoopOptimizerTrainingClient = providers.Factory(
        HadoopOptimizerTrainingClient,
    )

    base_env: Provider[gym.Env] = providers.Factory(
        OptimizerTrainingEnv,
        telemetry_manager=drl_telemetry_manager,
        training_client=training_client
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
        # TODO: REMOVE WHEN WE HAVE A REAL MODEL, IT INCREASES THE PREDICTION VARIANCE
        policy_kwargs=dict(log_std_init=0.8)
    )

    # TODO: SHOULD WE USE THE DRL MANAGER?
    # drl_manager: Provider[DRLManager] = providers.Factory(
    #     DRLManager,
    #     deployment_drl_model=deployment_drl_model,
    #     deployment_env=deployment_env,
    # )
