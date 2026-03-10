"""State observation wrapper with pluggable interpreter strategy.

A single ``StateWrapper`` class replaces the former StateWrapper / SW3–SW7
hierarchy.  The observation-building logic is delegated to a
:class:`~wrappers.state_interpreters.StateInterpreter` strategy object so that
new observation semantics can be added without touching side-effect code.
"""
from __future__ import annotations

import logging
import sys

import numpy as np
from gymnasium.core import ActionWrapper, ObservationWrapper

sys.path.insert(1, '/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch/src')
from config import config
from wrappers.state_interpreters import (
    AccumulatedLogVolumeInterpreter,
    AlertAwareInterpreter,
    KLDivergenceInterpreter,
    SparseStepInterpreter,
    StateContext,
    StateInterpreter,
)

logger = logging.getLogger(__name__)

ALERT_NORMALIZE_FACTOR = config.get('state.alert_normalize_factor', 100)


class StateWrapper(ObservationWrapper):
    """Manages log-type distributions, normalization and observation building.

    Side-effects (Splunk queries, distribution bookkeeping) live here.
    Pure observation construction is delegated to *interpreter*.
    """

    def __init__(self, env, interpreter: StateInterpreter, hosts_percentage: int = 100):
        super().__init__(env)

        self.interpreter = interpreter
        self.action_wrapper = self.get_wrapper(ActionWrapper)
        self.hosts_percentage = hosts_percentage

        self.total_current_logs = 0
        self.total_episode_logs = 0
        self._max_trigger_exposure_episode = np.zeros(
            len(config.get('reward.rule_names', [])), dtype=np.float64
        )

        n_logtypes = len(self.unwrapped.top_logtypes)
        total_steps = self.unwrapped.total_steps
        self.observation_space = interpreter.get_observation_space(n_logtypes, total_steps)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_wrapper(self, wrapper_class):
        """Retrieve a specific wrapper instance from the wrapper stack."""
        env = self.env
        while env:
            if isinstance(env, wrapper_class):
                return env
            env = getattr(env, 'env', None)
        raise ValueError(f"Wrapper {wrapper_class} not found in the wrapper stack.")

    def get_step_info(self):
        return {
            'total_current_logs': self.total_current_logs,
            'real_relevant_distribution': self.unwrapped.real_relevant_distribution,
            'total_episode_logs': self.total_episode_logs,
        }

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info.update(self.get_step_info())
        obs = self.observation(observation)
        info.update(self.get_step_info())
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation (unified flow)
    # ------------------------------------------------------------------

    def observation(self, obs):
        """Build the observation vector.

        1. Query Splunk for real distribution (unless episode is done).
        2. Mirror real → fake accumulated distribution (unless done).
        3. Normalize all vectors via the interpreter.
        4. Delegate final state construction to the interpreter.
        """
        if not self.unwrapped.done:
            self.update_real_distribution(self.unwrapped.time_manager.action_window.to_epoch_tuple())
        if not self.unwrapped.done:
            self.update_fake_distribution_from_real()

        ctx = self._build_context()
        state = self.interpreter.build_state(ctx)

        # Ensure float32 dtype (gym expects float32, not float64)
        state = np.asarray(state, dtype=np.float32)

        logger.debug(f"State: {state}")
        self.unwrapped.obs = state
        return state

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context(self) -> StateContext:
        """Normalize vectors and return a frozen snapshot for the interpreter."""
        # Per-step vectors
        real_state = self._get_state_vector(self.unwrapped.real_distribution)
        self.unwrapped.real_state = self.interpreter.normalize(real_state)

        fake_state = self._get_state_vector(self.unwrapped.fake_distribution)
        self.unwrapped.fake_state = self.interpreter.normalize(fake_state)

        # Accumulated vectors
        ac_real_state = self._get_state_vector(self.unwrapped.ac_real_distribution)
        self.unwrapped.ac_real_state = self.interpreter.normalize(ac_real_state)

        ac_fake_state = self._get_state_vector(self.unwrapped.ac_fake_distribution)
        self.unwrapped.ac_fake_state = self.interpreter.normalize(ac_fake_state)

        # Relevant-distribution dicts (written to shared state for reward wrappers)
        cache = self.unwrapped.logtype_key_cache
        self.unwrapped.real_relevant_distribution = {
            cache[lt]: self.unwrapped.ac_real_state[self.unwrapped.top_logtypes_indices[lt]]
            for lt in self.unwrapped.top_logtypes
        }
        self.unwrapped.fake_relevant_distribution = {
            cache[lt]: self.unwrapped.ac_fake_state[self.unwrapped.top_logtypes_indices[lt]]
            for lt in self.unwrapped.top_logtypes
        }

        # Alert-aware features (only computed for AlertAwareInterpreter)
        expected_baseline = None
        max_trigger_exposure = None
        if isinstance(self.interpreter, AlertAwareInterpreter):
            expected_baseline, max_trigger_exposure = self._build_alert_features()

        return StateContext(
            real_state=self.unwrapped.real_state,
            fake_state=self.unwrapped.fake_state,
            ac_real_state=self.unwrapped.ac_real_state,
            ac_fake_state=self.unwrapped.ac_fake_state,
            total_episode_logs=self.total_episode_logs,
            episodic_inserted_logs=self.unwrapped.episodic_inserted_logs,
            current_real_quantity=self.action_wrapper.current_real_quantity,
            step_counter=self.unwrapped.step_counter,
            total_steps=self.unwrapped.total_steps,
            n_logtypes=len(self.unwrapped.top_logtypes),
            expected_baseline=expected_baseline,
            max_trigger_exposure=max_trigger_exposure,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_state_vector(self, distribution):
        """Convert distribution dict to numpy vector (top_logtypes order)."""
        return np.array([distribution[lt] for lt in self.unwrapped.top_logtypes])

    def _build_alert_features(self):
        """Build expected-baseline and max-trigger-exposure vectors for AlertAwareInterpreter.

        Returns:
            (expected_baseline, max_trigger_exposure): Both np.ndarray of shape (n_rules,).
        """
        rule_names = config.get('reward.rule_names', [])
        n_rules = len(rule_names)

        # Expected baseline: static prior from config (raw alert counts)
        expected_values = config.get('reward.expected_alerts_per_rule',
                                     [0.0] * n_rules)
        expected_baseline = np.array(expected_values, dtype=np.float64)

        # Max trigger exposure: running maximum across the episode of
        # diversity_episode_logs[rule_logtype_1].
        # We track the max so that a high-exposure step is not forgotten if
        # the agent switches to a lower-diversity action in a later step.
        if self.unwrapped.step_counter == 0:
            self._max_trigger_exposure_episode = np.zeros(n_rules, dtype=np.float64)
        else:
            section_lt = self.unwrapped.section_logtypes
            diversity_logs = self.action_wrapper.diversity_episode_logs
            for i, rule in enumerate(rule_names):
                relevant_lts = section_lt.get(rule, [])
                if relevant_lts:
                    key = "_".join(relevant_lts[0]) + "_1"
                    current = diversity_logs.get(key, 0)
                    self._max_trigger_exposure_episode[i] = max(
                        self._max_trigger_exposure_episode[i], current
                    )

        # Normalize both vectors by a single shared factor (the global max
        # across both) so they are on the same scale for the policy network.
        trigger_exposure = self._max_trigger_exposure_episode.copy()
        normalizer_factor = 40
        expected_baseline = expected_baseline / normalizer_factor
        trigger_exposure = trigger_exposure / normalizer_factor

        return expected_baseline, trigger_exposure

    def update_fake_distribution_from_real(self):
        """Mirror real counts into fake accumulated distribution."""
        self.unwrapped.fake_distribution = {lt: 0 for lt in self.unwrapped.top_logtypes}
        self.unwrapped.fake_distribution['other'] = 0
        for lt in self.unwrapped.top_logtypes:
            self.unwrapped.ac_fake_distribution[lt] += self.unwrapped.real_distribution[lt]
            self.unwrapped.fake_distribution[lt] = self.unwrapped.real_distribution[lt]

    def update_real_distribution(self, time_range):
        """Query Splunk for the real log distribution in *time_range*."""
        real_counts = self.unwrapped.splunk_tools.get_real_distribution(
            *time_range, hosts_percentage=self.hosts_percentage,
        )
        self.unwrapped.real_distribution = {lt: 0 for lt in self.unwrapped.top_logtypes}
        self.unwrapped.real_distribution['other'] = 0
        self.total_current_logs = 0

        for logtype, count in real_counts.items():
            if logtype in self.unwrapped.top_logtypes:
                self.unwrapped.real_distribution[logtype] = count
                self.unwrapped.ac_real_distribution[logtype] += count
                self.total_current_logs += count

        if self.hosts_percentage < 100:
            scale_factor = 100 / self.hosts_percentage
            self.total_current_logs = int(self.total_current_logs * scale_factor)

        self.total_episode_logs += self.total_current_logs
        self.action_wrapper.current_real_quantity = self.total_current_logs

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Reset wrapper state for a new episode."""
        logger.info("Resetting StateWrapper")
        self.total_episode_logs = 0
        self._max_trigger_exposure_episode = np.zeros(
            len(config.get('reward.rule_names', [])), dtype=np.float64
        )
        self.unwrapped.done = False
        self.unwrapped.step_counter = 0

        # advance_window reads should_delete & episodic_fake_logs_qnt BEFORE clearing
        self.unwrapped.time_manager.advance_window(
            global_step=self.unwrapped.all_steps_counter,
            violation=False,
            should_delete=self.unwrapped.should_delete,
            logs_qnt=self.unwrapped.episodic_fake_logs_qnt,
        )
        if self.unwrapped.time_manager.is_delete:
            self.unwrapped.should_delete = False

        # Clear per-episode shared state
        self.unwrapped.shared.reset(
            self.unwrapped.top_logtypes,
            self.unwrapped.relevant_logtypes,
            self.unwrapped.logtype_key_cache,
        )

        new_obs = self.observation(None)
        options = self.get_step_info()
        obs, info = self.env.reset(seed=seed, options=options)
        info.update(self.get_step_info())

        return new_obs, info


# ======================================================================
# Registry & factory
# ======================================================================

def _make_alert_aware():
    n_rules = len(config.get('reward.rule_names', []))
    return AlertAwareInterpreter(n_rules=n_rules, include_step_ratio=True)


STATE_INTERPRETER_REGISTRY = {
    # Backward-compatible class names
    'StateWrapper7': lambda: AccumulatedLogVolumeInterpreter(include_step_ratio=True),
    'StateWrapper6': lambda: AccumulatedLogVolumeInterpreter(include_step_ratio=False),
    'StateWrapper3': lambda: KLDivergenceInterpreter(),
    'SparseStep':    lambda: SparseStepInterpreter(),
    'StateWrapper8': _make_alert_aware,
    # Descriptive names
    'KLDivergence':          lambda: KLDivergenceInterpreter(),
    'AccumulatedLogVolume':  lambda: AccumulatedLogVolumeInterpreter(include_step_ratio=True),
    'AlertAware':            _make_alert_aware,
}


def create_state_wrapper(env, state_type: str, hosts_percentage: int = 100) -> StateWrapper:
    """Create a StateWrapper with the specified interpreter.

    Args:
        env: The gymnasium environment to wrap.
        state_type: Key into STATE_INTERPRETER_REGISTRY.
        hosts_percentage: Percentage of hosts to query (0-100).

    Returns:
        StateWrapper instance.
    """
    if state_type not in STATE_INTERPRETER_REGISTRY:
        raise ValueError(
            f"Unknown state type '{state_type}'. "
            f"Available: {list(STATE_INTERPRETER_REGISTRY.keys())}"
        )
    interpreter = STATE_INTERPRETER_REGISTRY[state_type]()
    return StateWrapper(env, interpreter, hosts_percentage)
