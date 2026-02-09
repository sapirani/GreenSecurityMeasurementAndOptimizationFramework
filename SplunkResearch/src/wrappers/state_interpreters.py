"""State interpreters: strategy objects that build observation vectors.

Each interpreter defines:
  - ``get_observation_space(...)`` -- the Gymnasium Box for this state semantics
  - ``normalize(state)``          -- normalization strategy for raw count vectors
  - ``build_state(ctx)``          -- pure function returning the observation array

The base ``StateWrapper`` owns side-effects (Splunk queries, distribution updates).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateContext:
    """Read-only snapshot passed to interpreters each step."""

    real_state: np.ndarray          # per-step normalized
    fake_state: np.ndarray          # per-step normalized
    ac_real_state: np.ndarray       # accumulated normalized
    ac_fake_state: np.ndarray       # accumulated normalized
    total_episode_logs: int
    episodic_inserted_logs: int
    current_real_quantity: int
    step_counter: int
    total_steps: int
    n_logtypes: int


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class StateInterpreter(ABC):
    """Strategy interface for state/observation construction."""

    @abstractmethod
    def get_observation_space(self, n_logtypes: int, total_steps: int) -> spaces.Space:
        ...

    @abstractmethod
    def normalize(self, state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def build_state(self, ctx: StateContext) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Concrete: SparseStepInterpreter  (== original base StateWrapper)
# ---------------------------------------------------------------------------

class SparseStepInterpreter(StateInterpreter):
    """Per-step distributions + one-hot step vector.

    Space: ``(n_logtypes*2 + total_steps,)``
    Normalize: proportional (state / sum).
    """

    def get_observation_space(self, n_logtypes, total_steps):
        return spaces.Box(
            low=0, high=1,
            shape=(n_logtypes * 2 + total_steps,),
            dtype=np.float64,
        )

    def normalize(self, state):
        total = np.sum(state)
        if total > 0:
            return state / total
        return np.ones_like(state) / len(state)

    def build_state(self, ctx: StateContext) -> np.ndarray:
        state = np.append(ctx.real_state, ctx.fake_state)
        sparse_vector = np.zeros(ctx.total_steps)
        sparse_vector[ctx.step_counter] = 1
        return np.append(state, sparse_vector)


# ---------------------------------------------------------------------------
# Concrete: KLDivergenceInterpreter  (== StateWrapper3)
# ---------------------------------------------------------------------------

class KLDivergenceInterpreter(StateInterpreter):
    """Accumulated distributions + KL divergence + step ratio + current qty.

    Space: ``(n_logtypes*2 + 3,)``
    Normalize: absolute ``(state + eps) / 100000``.
    """

    def get_observation_space(self, n_logtypes, total_steps):
        return spaces.Box(
            low=0, high=3,
            shape=(n_logtypes * 2 + 3,),
            dtype=np.float64,
        )

    def normalize(self, state):
        return (state + 1e-10) / 100000

    def build_state(self, ctx: StateContext) -> np.ndarray:
        # KL divergence between accumulated distributions
        p = ctx.ac_real_state / np.sum(ctx.ac_real_state)
        q = ctx.ac_fake_state / np.sum(ctx.ac_fake_state)
        kl_div = np.sum(p * np.log(p / q))

        state = np.append(ctx.ac_real_state, ctx.ac_fake_state)
        state = np.append(state, kl_div)
        state = np.append(state, ctx.step_counter / ctx.total_steps)
        state = np.append(state, ctx.current_real_quantity / 100000)
        return state


# ---------------------------------------------------------------------------
# Concrete: AccumulatedLogVolumeInterpreter  (== StateWrapper6 / StateWrapper7)
# ---------------------------------------------------------------------------

class AccumulatedLogVolumeInterpreter(StateInterpreter):
    """Accumulated distributions + log volume metrics.

    Args:
        include_step_ratio: If True, append ``step_counter / total_steps``
            (StateWrapper7 behaviour). If False, omit (StateWrapper6).

    Space: ``(n_logtypes*2 + 3,)`` or ``(n_logtypes*2 + 2,)``
    Normalize: sum-based ``state / (sum + eps)``.
    """

    def __init__(self, include_step_ratio: bool = True):
        self.include_step_ratio = include_step_ratio

    def get_observation_space(self, n_logtypes, total_steps):
        extra = 3 if self.include_step_ratio else 2
        return spaces.Box(
            low=0, high=1,
            shape=(n_logtypes * 2 + extra,),
            dtype=np.float64,
        )

    def normalize(self, state):
        return state / (np.sum(state) + 1e-10)

    def build_state(self, ctx: StateContext) -> np.ndarray:
        state = np.append(ctx.ac_real_state, ctx.ac_fake_state)
        state = np.append(state, ctx.total_episode_logs / 500000)
        state = np.append(
            state,
            (ctx.episodic_inserted_logs + ctx.total_episode_logs) / 500000,
        )
        if self.include_step_ratio:
            state = np.append(state, ctx.step_counter / ctx.total_steps)
        return state
