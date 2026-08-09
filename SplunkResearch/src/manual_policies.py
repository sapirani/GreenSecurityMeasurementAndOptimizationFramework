"""Manual (fixed) policies for baseline evaluation.

Provides a lightweight model-like wrapper (`ManualPolicyModel`) that returns a
hand-crafted action vector each step, plus factory helpers to build action
vectors for common baseline policies.

Usage flow:
    --mode eval_post_training --manual-policy only_4662
    --mode eval_post_training --manual-policy all_relevant --manual-diversity 0.5
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class ManualPolicyModel:
    """Returns a fixed action vector each call. Compatible with SB3 eval flow."""

    def __init__(self, action_vector, observation_space, action_space):
        self._action = np.asarray(action_vector, dtype=np.float32).copy()
        if self._action.shape != action_space.shape:
            raise ValueError(
                f"action vector shape {self._action.shape} != "
                f"action_space.shape {action_space.shape}"
            )
        self.observation_space = observation_space
        self.action_space = action_space
        self.policy = self  # alias for callers reaching `model.policy.*`
        self.num_timesteps = 0

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        obs = np.asarray(observation)
        if obs.ndim == len(self.observation_space.shape) + 1:
            n_envs = obs.shape[0]
            actions = np.tile(self._action, (n_envs, 1))
        else:
            actions = self._action.copy()
        return actions, state

    # No-op stubs used by ExperimentManager._run_evaluation
    def set_env(self, env):
        return None

    def set_logger(self, logger):
        return None

    def save(self, path):
        np.save(str(path) + ".npy", self._action)


# ---------------------------------------------------------------------------
# Action vector builders
# ---------------------------------------------------------------------------

LogType = Tuple[str, str]


def build_action_vector(
    top_logtypes: Sequence[LogType],
    relevant_logtypes: Sequence[LogType],
    distribution_logtypes: Sequence[LogType],
    diversity_value: float,
) -> np.ndarray:
    """Build a SoftmaxDistribution / SmoothTrigger action vector.

    Layout: first ``len(top_logtypes)`` dims = softmax distribution (high-temperature
    softmax concentrates probability on dims set to 1.0), last ``len(relevant_logtypes)``
    dims = per-relevant-logtype diversity in [0,1].
    """
    n_top = len(top_logtypes)
    n_rel = len(relevant_logtypes)
    vec = np.zeros(n_top + n_rel, dtype=np.float32)

    top_index = {lt: i for i, lt in enumerate(top_logtypes)}
    rel_index = {lt: j for j, lt in enumerate(relevant_logtypes)}

    for lt in distribution_logtypes:
        if lt in top_index:
            vec[top_index[lt]] = 1.0
        if lt in rel_index:
            vec[n_top + rel_index[lt]] = float(diversity_value)

    return vec


def policy_only_4662_max_diversity(
    top_logtypes: Sequence[LogType],
    relevant_logtypes: Sequence[LogType],
    diversity_value: float = 1.0,
) -> np.ndarray:
    """Concentrate all volume on EventCode 4662 (AD Replication) at the given diversity."""
    target = ("wineventlog:security", "4662")
    if target not in top_logtypes:
        raise ValueError(f"{target} not in top_logtypes; cannot build policy")
    return build_action_vector(
        top_logtypes, relevant_logtypes,
        distribution_logtypes=[target],
        diversity_value=diversity_value,
    )


def policy_all_relevant_at_diversity(
    top_logtypes: Sequence[LogType],
    relevant_logtypes: Sequence[LogType],
    diversity_value: float,
) -> np.ndarray:
    """Uniform softmax over all relevant logtypes; same diversity on all."""
    return build_action_vector(
        top_logtypes, relevant_logtypes,
        distribution_logtypes=list(relevant_logtypes),
        diversity_value=diversity_value,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MANUAL_POLICY_NAMES = ("only_4662", "all_relevant")


def build_manual_policy_action(
    name: str,
    top_logtypes: Sequence[LogType],
    relevant_logtypes: Sequence[LogType],
    diversity_value: float = 1.0,
) -> np.ndarray:
    if name == "only_4662":
        return policy_only_4662_max_diversity(top_logtypes, relevant_logtypes, diversity_value)
    if name == "all_relevant":
        return policy_all_relevant_at_diversity(top_logtypes, relevant_logtypes, diversity_value)
    raise ValueError(
        f"Unknown manual policy '{name}'. Available: {MANUAL_POLICY_NAMES}"
    )
