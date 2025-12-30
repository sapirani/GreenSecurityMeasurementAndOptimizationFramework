import gymnasium as gym
from datetime import datetime
from dependency_injector import containers, providers
from dependency_injector.providers import Provider
from gymnasium.wrappers import OrderEnforcing, FlattenObservation, RescaleObservation, RescaleAction
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from elastic_reader.consts import TimePickerInputStrategy
from hadoop_optimizer.deployment_server.drl_manager import DRLManager
from hadoop_optimizer.drl_telemetry.telemetry_manager import DRLTelemetryManager
from hadoop_optimizer.drl_envs.deployment_env import OptimizerDeploymentEnv
from hadoop_optimizer.gymnasium_wrappers.action.action_types_decoder import ActionTypesDecoder
from hadoop_optimizer.gymnasium_wrappers.action.flatten_action import FlattenAction
from hadoop_optimizer.gymnasium_wrappers.state.dict_leafs_as_numpy import DictLeafsAsNumpy
from hadoop_optimizer.gymnasium_wrappers.state.time_limit_wrapper import TimeLimitWrapper
from hadoop_optimizer.gymnasium_wrappers.state.reset_enforcer import ResetEnforcer
from hadoop_optimizer.gymnasium_wrappers.state_validators.enforce_observation_bounds import EnforceObservationBounds
from user_input.elastic_reader_input.abstract_date_picker import TimePickerChosenInput, ReadingMode
from user_input.elastic_reader_input.time_picker_input_factory import get_time_picker_input


class Container(containers.DeclarativeContainer):
    # TODO: ENSURE THAT CONFIG HIERARCHY MAKES SENSE
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

    drl_telemetry_manager: Provider[DRLTelemetryManager] = providers.Singleton(
        DRLTelemetryManager,
        time_windows_seconds=config.drl_state.time_windows_seconds,
        split_by=config.drl_state.split_by,
    )

    base_env: Provider[gym.Env] = providers.Factory(
        OptimizerDeploymentEnv,
        telemetry_manager=drl_telemetry_manager
    )

    # TODO: ALLOW EASY CONFIGURATION OF DECORATORS IN BOOTSTRAPPING
    order_enforcer: Provider[gym.Env] = providers.Factory(
        OrderEnforcing,
        base_env,
    )

    time_limit_env: Provider[gym.Env] = providers.Factory(
        TimeLimitWrapper,
        order_enforcer,
        max_episode_steps=config.max_episode_steps,
    )

    reset_enforcer_env: Provider[gym.Env] = providers.Factory(
        ResetEnforcer,
        time_limit_env,
    )

    dict_leafs_as_numpy: Provider[gym.Env] = providers.Factory(
        DictLeafsAsNumpy,
        reset_enforcer_env,
    )

    flatten_observation_env: Provider[gym.Env] = providers.Factory(
        FlattenObservation,
        dict_leafs_as_numpy,
    )

    enforce_observation_bounds: Provider[gym.Env] = providers.Factory(
        EnforceObservationBounds,
        flatten_observation_env,
    )

    action_types_decoder_env: Provider[gym.Env] = providers.Factory(
        ActionTypesDecoder,
        enforce_observation_bounds,
    )

    flatten_action_env: Provider[gym.Env] = providers.Factory(
        FlattenAction,
        action_types_decoder_env,
    )

    rescale_action_env: Provider[gym.Env] = providers.Factory(
        RescaleAction,
        flatten_action_env,
        min_action=0,
        max_action=1,
    )

    deployment_env: Provider[gym.Env] = providers.Factory(
        RescaleObservation,     # TODO: CONSIDER USING NormalizeObservation
        rescale_action_env,
        min_obs=-1,
        max_obs=1,
    )

    # TODO: LOAD THE BEST AGENT INSTEAD OF INITIALIZING A NEW MODEL HERE, FOR EXAMPLE: PPO.load(<path>)
    deployment_agent: Provider[BaseAlgorithm] = providers.Singleton(
        PPO,
        policy=ActorCriticPolicy,
        env=deployment_env,
        # TODO: REMOVE WHEN WE HAVE A REAL MODEL, IT INCREASES THE PREDICTION VARIANCE
        policy_kwargs=dict(log_std_init=0.8)
    )

    drl_manager: Provider[DRLManager] = providers.Factory(
        DRLManager,
        deployment_agent=deployment_agent,
        deployment_env=deployment_env,
    )
