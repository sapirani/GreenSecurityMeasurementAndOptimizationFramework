from typing import List
from gymnasium.wrappers import OrderEnforcing, FlattenObservation, RescaleAction, RescaleObservation

from custom_package_extensions.gymnasium_wrappers.action.action_types_decoder import ActionTypesDecoder
from custom_package_extensions.gymnasium_wrappers.action.flatten_action import FlattenAction
from custom_package_extensions.gymnasium_wrappers.loggers.performance_logger import PerformanceLoggerWrapper
from custom_package_extensions.gymnasium_wrappers.state.dict_leafs_as_numpy import DictLeafsAsNumpy
from custom_package_extensions.gymnasium_wrappers.state.reset_enforcer import ResetEnforcer
from custom_package_extensions.gymnasium_wrappers.state.time_limit_wrapper import TimeLimitWrapper
from custom_package_extensions.gymnasium_wrappers.state_validators.enforce_observation_bounds import \
    EnforceObservationBounds
from hadoop_optimizer.common.env_composition_config.env_wrapper_spec import EnvWrapperSpec, EnvWrappersParams


def get_basic_env_wrappers(wrappers_params: EnvWrappersParams) -> List[EnvWrapperSpec]:
    # NOTE: the first wrapper is applied last
    return [
        EnvWrapperSpec(OrderEnforcing),
        EnvWrapperSpec(TimeLimitWrapper, dict(
            truncated_penalty=wrappers_params.truncated_penalty,
            max_episode_steps=wrappers_params.max_episode_steps
        )),
        EnvWrapperSpec(ResetEnforcer),
        EnvWrapperSpec(DictLeafsAsNumpy),
        EnvWrapperSpec(FlattenObservation),
        EnvWrapperSpec(EnforceObservationBounds),
        EnvWrapperSpec(ActionTypesDecoder),
        EnvWrapperSpec(FlattenAction),
        EnvWrapperSpec(
            RescaleAction,
            dict(min_action=wrappers_params.min_action, max_action=wrappers_params.max_action)
        ),
        EnvWrapperSpec(
            RescaleObservation,
            dict(min_obs=wrappers_params.min_obs, max_obs=wrappers_params.max_obs)
        ),    # TODO: CONSIDER USING NormalizeObservation
    ]

def get_training_wrappers(wrappers_params: EnvWrappersParams) -> List[EnvWrapperSpec]:
    return [
        EnvWrapperSpec(
            PerformanceLoggerWrapper,
            dict(logger=wrappers_params.logger)
        )
    ]