from gymnasium.wrappers import OrderEnforcing, FlattenObservation, RescaleAction, RescaleObservation
from hadoop_optimizer.env_composition_config.env_wrapper_spec import EnvWrapperSpec, EnvWrappersParams
from hadoop_optimizer.gymnasium_wrappers.action.action_types_decoder import ActionTypesDecoder
from hadoop_optimizer.gymnasium_wrappers.action.flatten_action import FlattenAction
from hadoop_optimizer.gymnasium_wrappers.state.dict_leafs_as_numpy import DictLeafsAsNumpy
from hadoop_optimizer.gymnasium_wrappers.state.reset_enforcer import ResetEnforcer
from hadoop_optimizer.gymnasium_wrappers.state.time_limit_wrapper import TimeLimitWrapper
from hadoop_optimizer.gymnasium_wrappers.state_validators.enforce_observation_bounds import EnforceObservationBounds


def get_env_wrappers(wrappers_params: EnvWrappersParams):
    return [
        EnvWrapperSpec(OrderEnforcing),
        EnvWrapperSpec(TimeLimitWrapper, dict(max_episode_steps=wrappers_params.max_episode_steps)),
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
