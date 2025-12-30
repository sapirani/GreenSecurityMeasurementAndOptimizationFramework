import gymnasium as gym
from datetime import datetime
from dependency_injector import containers, providers
from dependency_injector.providers import Provider
from gymnasium.wrappers import OrderEnforcing, FlattenObservation, RescaleObservation, RescaleAction
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from elastic_reader.consts import TimePickerInputStrategy
from hadoop_optimizer.deployment_server.drl_model.drl_model import DRLModel
from hadoop_optimizer.deployment_server.drl_model.drl_state import DRLState
from hadoop_optimizer.drl_envs.deployment_env import OptimizerDeploymentEnv
from hadoop_optimizer.drl_envs.dummy_env import DummyEnv
from hadoop_optimizer.gymnasium_wrappers.action.flatten_action import FlattenAction
from hadoop_optimizer.gymnasium_wrappers.common.time_limit_wrapper import TimeLimitWrapper
from hadoop_optimizer.gymnasium_wrappers.common.reset_enforcer import ResetEnforcer
from hadoop_optimizer.gymnasium_wrappers.state_validators.enforce_observation_bounds import EnforceObservationBounds
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

    flatten_observation_env: Provider[gym.Env] = providers.Factory(
        FlattenObservation,
        reset_enforcer_env,
    )

    enforce_observation_bounds: Provider[gym.Env] = providers.Factory(
        EnforceObservationBounds,
        flatten_observation_env,
    )

    flatten_action_env: Provider[gym.Env] = providers.Factory(
        FlattenAction,
        enforce_observation_bounds,
    )

    # TODO: UNDERSTAND HOW TO WORK WITH RESCALING ACTIONS
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

    drl_time_picker_input: Provider[TimePickerChosenInput] = providers.Factory(
        get_time_picker_input,
        time_picker_input_strategy=TimePickerInputStrategy.FROM_CONFIGURATION,
        preconfigured_time_input=providers.Callable(lambda: TimePickerChosenInput(
            start=datetime.now(tz=datetime.now().astimezone().tzinfo),
            end=None,
            mode=ReadingMode.REALTIME
        ))
    )
