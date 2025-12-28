from datetime import datetime
from dependency_injector import containers, providers
from dependency_injector.providers import Provider
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy

from elastic_reader.consts import TimePickerInputStrategy
from hadoop_optimizer.deployment_server.drl_model.drl_model import DRLModel
from hadoop_optimizer.deployment_server.drl_model.drl_state import DRLState
from hadoop_optimizer.drl_envs.deployment_env import OptimizerDeploymentEnv
from user_input.elastic_reader_input.abstract_date_picker import TimePickerChosenInput, ReadingMode
from user_input.elastic_reader_input.time_picker_input_factory import get_time_picker_input


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    drl_state: Provider[DRLState] = providers.Factory(
        DRLState,
        time_windows_seconds=config.drl_state.time_windows_seconds,
        split_by=config.drl_state.split_by,
    )

    # drl_model: Provider[DRLModel] = providers.Singleton(
    #     DRLModel,
    #     drl_state=drl_state
    # )

    deployment_env: Provider[OptimizerDeploymentEnv] = providers.Factory(
        OptimizerDeploymentEnv,
    )

    # TODO: LOAD THE BEST AGENT INSTEAD OF INITIALIZING A NEW MODEL HERE, FOR EXAMPLE: PPO.load(<path>)
    deployment_agent: Provider[BaseAlgorithm] = providers.Singleton(
        PPO,
        policy=ActorCriticPolicy,
        env=deployment_env,
    )

    drl_time_picker_input: Provider[TimePickerChosenInput] = providers.Factory(
        get_time_picker_input,
        time_picker_input_strategy=TimePickerInputStrategy.FROM_CONFIGURATION,
        preconfigured_time_input=providers.Callable(lambda: TimePickerChosenInput(
            start=datetime.now(tz=datetime.now().astimezone().tzinfo),
            end=None,
            mode=ReadingMode.REALTIME
        ))
    )