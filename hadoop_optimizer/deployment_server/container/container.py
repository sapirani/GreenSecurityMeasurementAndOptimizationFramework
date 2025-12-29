import gymnasium as gym
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
from hadoop_optimizer.gymnasium_wrappers.common.time_limit_wrapper import TimeLimitWrapper
from hadoop_optimizer.gymnasium_wrappers.common.reset_enforcer import ResetEnforcer
from user_input.elastic_reader_input.abstract_date_picker import TimePickerChosenInput, ReadingMode
from user_input.elastic_reader_input.time_picker_input_factory import get_time_picker_input


class Container(containers.DeclarativeContainer):
    # TODO: ENSURE THAT CONFIG HIERARCHY MAKES SENSE
    config = providers.Configuration()

    drl_state: Provider[DRLState] = providers.Factory(
        DRLState,
        time_windows_seconds=config.drl_state.time_windows_seconds,
        split_by=config.drl_state.split_by,
    )

    # TODO: TURN INTO A CLASS THAT MONITORS THE CLUSTER LOAD
    drl_model: Provider[DRLModel] = providers.Singleton(
        DRLModel,
        drl_state=drl_state
    )

    base_env: Provider[gym.Env] = providers.Factory(
        OptimizerDeploymentEnv,
    )

    # TODO: ALLOW MORE CONFIGURABLE DECORATORS BOOTSTRAPPING
    time_limit_env: Provider[gym.Env] = providers.Factory(
        TimeLimitWrapper,
        base_env,
        max_episode_steps=config.max_episode_steps,
    )

    deployment_env: Provider[gym.Env] = providers.Factory(
        ResetEnforcer,
        time_limit_env,
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
