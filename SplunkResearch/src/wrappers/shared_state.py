"""Typed dataclass for episode-level shared state on SplunkEnv.

This centralises the 'blackboard' attributes that Action, State,
and Reward wrappers all read/write through ``env.unwrapped.*``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class EpisodeSharedState:
    """Mutable shared state for one episode.

    Created once in ``SplunkEnv.__init__`` and reset at the start
    of each episode via :meth:`reset`.
    """

    # --- counters ---
    episodic_inserted_logs: int = 0
    episodic_fake_logs_qnt: int = 0
    step_counter: int = 0
    done: bool = False

    # --- observation cache ---
    obs: np.ndarray = field(default_factory=lambda: np.array([]))
    should_delete: bool = False

    # --- distribution dicts (populated by reset()) ---
    fake_distribution: Dict = field(default_factory=dict)
    ac_fake_distribution: Dict = field(default_factory=dict)
    real_distribution: Dict = field(default_factory=dict)
    ac_real_distribution: Dict = field(default_factory=dict)
    fake_relevant_distribution: Dict = field(default_factory=dict)
    real_relevant_distribution: Dict = field(default_factory=dict)

    # --- normalised state vectors ---
    real_state: np.ndarray = field(default_factory=lambda: np.array([]))
    fake_state: np.ndarray = field(default_factory=lambda: np.array([]))
    ac_real_state: np.ndarray = field(default_factory=lambda: np.array([]))
    ac_fake_state: np.ndarray = field(default_factory=lambda: np.array([]))

    # --- per-rule tracking ---
    rules_rel_diff_alerts: Dict = field(default_factory=dict)

    def init_distributions(
        self,
        top_logtypes: List[Tuple[str, str]],
        relevant_logtypes: List[Tuple[str, str]],
        logtype_key_cache: Dict = None,
    ) -> None:
        """Populate distribution dicts from the environment's logtype lists.

        Called once during ``SplunkEnv.__init__``.
        """
        cache = logtype_key_cache or {lt: "_".join(lt) for lt in top_logtypes}
        self.fake_distribution = {lt: 0 for lt in top_logtypes}
        self.fake_distribution['other'] = 0
        self.fake_relevant_distribution = {cache[lt]: 0 for lt in top_logtypes}
        self.real_distribution = {lt: 0 for lt in top_logtypes}
        self.real_distribution['other'] = 0
        self.ac_real_distribution = {lt: 0 for lt in top_logtypes}
        self.ac_real_distribution['other'] = 0
        self.ac_fake_distribution = {lt: 0 for lt in top_logtypes}
        self.ac_fake_distribution['other'] = 0
        self.real_relevant_distribution = {cache[lt]: 0 for lt in top_logtypes}
        self.rules_rel_diff_alerts = {rule: 0 for rule in relevant_logtypes}

    def reset(
        self,
        top_logtypes: List[Tuple[str, str]],
        relevant_logtypes: List[Tuple[str, str]],
        logtype_key_cache: Dict = None,
    ) -> None:
        """Clear per-episode state.

        **Important:** ``StateWrapper`` reads ``episodic_fake_logs_qnt``
        *before* calling this method, so the ordering in the reset chain
        must be preserved.
        """
        self.episodic_inserted_logs = 0
        self.episodic_fake_logs_qnt = 0
        self.done = False
        self.obs = np.array([])

        self.real_state = np.array([])
        self.fake_state = np.array([])
        self.ac_real_state = np.array([])
        self.ac_fake_state = np.array([])

        self.init_distributions(top_logtypes, relevant_logtypes, logtype_key_cache)
