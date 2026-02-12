"""Action interpreters: strategy objects that convert raw NN output → InjectionPlan.

Each interpreter defines:
  - ``get_action_space(...)`` – the Gymnasium Box for this action semantics
  - ``interpret(raw_action, ctx)`` – pure function returning an InjectionPlan

The base ``Action`` wrapper owns side-effects (episode tracking, Splunk calls).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

import numpy as np
from gymnasium import spaces

import sys
sys.path.insert(1, '/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch/src')
from config import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ActionContext:
    """Read-only context passed to interpreters each step."""

    top_logtypes: List[Tuple[str, str]]
    relevant_logtypes: List[Tuple[str, str]]
    relevant_logtypes_index: Dict[Tuple[str, str], int]
    additional_percentage: float
    base_log_count: int
    diversity_factor: int
    softmax_temperature: float
    sigmoid_sharpness: float


@dataclass
class InjectionPlan:
    """Output of an interpreter – what to inject this step."""

    logs_to_inject: Dict[str, Dict]  # {key: {'count': N, 'diversity': D}}
    total_inserted: int
    per_logtype_counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ActionInterpreter(ABC):
    """Strategy interface for action interpretation."""

    @abstractmethod
    def get_action_space(
        self,
        top_logtypes: List[Tuple[str, str]],
        relevant_logtypes: List[Tuple[str, str]],
    ) -> spaces.Space:
        ...

    @abstractmethod
    def interpret(self, raw_action: np.ndarray, ctx: ActionContext) -> InjectionPlan:
        ...


# ---------------------------------------------------------------------------
# Concrete: SoftmaxDistributionInterpreter  (== old Action8)
# ---------------------------------------------------------------------------

class SoftmaxDistributionInterpreter(ActionInterpreter):
    """Exact replica of Action8 semantics.

    Space: Box([0,1]^(N+M))  where N = len(top), M = len(relevant)
    - First N dims → softmax(temperature * x) distribution over top_logtypes
    - Last M dims  → diversity per relevant logtype  (ceil trigger)
    - Fixed volume: additional_percentage * base_log_count
    """

    def get_action_space(self, top_logtypes, relevant_logtypes):
        n = len(top_logtypes) + len(relevant_logtypes)
        return spaces.Box(
            low=np.zeros(n, dtype=np.float32),
            high=np.ones(n, dtype=np.float32),
            shape=(n,),
            dtype=np.float32,
        )

    def interpret(self, raw_action: np.ndarray, ctx: ActionContext) -> InjectionPlan:
        n_top = len(ctx.top_logtypes)

        # Softmax distribution
        distribution = raw_action[:n_top]
        scaled = ctx.softmax_temperature * distribution
        exp_vals = np.exp(scaled - np.max(scaled))
        distribution = exp_vals / exp_vals.sum()

        diversity_list = raw_action[n_top:]
        num_logs = ctx.additional_percentage * ctx.base_log_count

        logs_to_inject: Dict[str, Dict] = {}
        total_inserted = 0

        for i, logtype in enumerate(ctx.top_logtypes):
            log_count = int(distribution[i] * num_logs)
            total_inserted += log_count

            diversity = 0
            is_trigger = 0

            if logtype in ctx.relevant_logtypes:
                idx = ctx.relevant_logtypes.index(logtype)
                diversity = float(diversity_list[idx])
                if log_count == 0:
                    diversity = 0
                is_trigger = int(np.ceil(diversity))
                if is_trigger:
                    diversity = int(diversity * ctx.diversity_factor)
                else:
                    diversity = 0  # Not triggering → no malicious variants

            diversity = max(1, min(diversity, log_count))

            key = f"{logtype[0]}_{logtype[1]}_{is_trigger}"
            logs_to_inject[key] = {'count': log_count, 'diversity': diversity}

            if logtype in ctx.relevant_logtypes:
                opp_trigger = 1 - is_trigger
                opp_key = f"{logtype[0]}_{logtype[1]}_{opp_trigger}"
                logs_to_inject[opp_key] = {'count': 0, 'diversity': 0}

        return InjectionPlan(
            logs_to_inject=logs_to_inject,
            total_inserted=total_inserted,
        )


# ---------------------------------------------------------------------------
# Concrete: SmoothTriggerInterpreter  (sigmoid trigger, fixed volume)
# ---------------------------------------------------------------------------

class SmoothTriggerInterpreter(ActionInterpreter):
    """Sigmoid trigger with fixed volume (same as SoftmaxDistribution).

    Space: Box([0,1]^(N+M))  where N = len(top), M = len(relevant)
    - First N dims → softmax(temperature * x) distribution over top_logtypes
    - Last M dims  → diversity per relevant logtype (sigmoid trigger)
    - Fixed volume: additional_percentage * base_log_count

    Trigger: sigmoid(sharpness * (diversity - 0.5)) > 0.5
    """

    def get_action_space(self, top_logtypes, relevant_logtypes):
        n = len(top_logtypes) + len(relevant_logtypes)
        return spaces.Box(
            low=np.zeros(n, dtype=np.float32),
            high=np.ones(n, dtype=np.float32),
            shape=(n,),
            dtype=np.float32,
        )

    def interpret(self, raw_action: np.ndarray, ctx: ActionContext) -> InjectionPlan:
        n_top = len(ctx.top_logtypes)

        # Softmax distribution
        distribution = raw_action[:n_top]
        scaled = ctx.softmax_temperature * distribution
        exp_vals = np.exp(scaled - np.max(scaled))
        distribution = exp_vals / exp_vals.sum()

        diversity_list = raw_action[n_top:]
        num_logs = ctx.additional_percentage * ctx.base_log_count

        logs_to_inject: Dict[str, Dict] = {}
        total_inserted = 0

        for i, logtype in enumerate(ctx.top_logtypes):
            log_count = int(distribution[i] * num_logs)
            total_inserted += log_count

            diversity = 0
            is_trigger = 0

            if logtype in ctx.relevant_logtypes:
                idx = ctx.relevant_logtypes.index(logtype)
                raw_div = float(diversity_list[idx])
                if log_count == 0:
                    raw_div = 0.0
                # Smooth sigmoid trigger
                trigger_prob = 1.0 / (1.0 + np.exp(-ctx.sigmoid_sharpness * (raw_div - 0.5)))
                is_trigger = int(trigger_prob > 0.5)
                if is_trigger:
                    diversity = int(raw_div * ctx.diversity_factor)
                else:
                    diversity = 0  # Not triggering → no malicious variants

            diversity = max(1, min(diversity, log_count))

            key = f"{logtype[0]}_{logtype[1]}_{is_trigger}"
            logs_to_inject[key] = {'count': log_count, 'diversity': diversity}

            if logtype in ctx.relevant_logtypes:
                opp_trigger = 1 - is_trigger
                opp_key = f"{logtype[0]}_{logtype[1]}_{opp_trigger}"
                logs_to_inject[opp_key] = {'count': 0, 'diversity': 0}

        return InjectionPlan(
            logs_to_inject=logs_to_inject,
            total_inserted=total_inserted,
        )


# ---------------------------------------------------------------------------
# Decorator: LearnableVolumeDecorator  (composable with any interpreter)
# ---------------------------------------------------------------------------

class LearnableVolumeDecorator(ActionInterpreter):
    """Wraps any interpreter, appending a learnable volume dimension.

    Adds +1 to the action space. The last dimension is a volume multiplier
    in [0, 1] that scales ``additional_percentage`` before delegating to the
    inner interpreter.

    This keeps volume control orthogonal to distribution/trigger semantics.
    """

    def __init__(self, inner: ActionInterpreter):
        self._inner = inner

    def get_action_space(self, top_logtypes, relevant_logtypes):
        inner_space = self._inner.get_action_space(top_logtypes, relevant_logtypes)
        n = inner_space.shape[0] + 1
        return spaces.Box(
            low=np.zeros(n, dtype=np.float32),
            high=np.ones(n, dtype=np.float32),
            shape=(n,),
            dtype=np.float32,
        )

    def interpret(self, raw_action: np.ndarray, ctx: ActionContext) -> InjectionPlan:
        volume_pct = float(raw_action[-1])
        inner_action = raw_action[:-1]

        scaled_ctx = replace(
            ctx,
            additional_percentage=ctx.additional_percentage * volume_pct,
        )
        return self._inner.interpret(inner_action, scaled_ctx)
