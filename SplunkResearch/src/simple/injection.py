"""Action decode: raw policy output -> an injection plan.

Ported verbatim from wrappers/action_interpreters.py:95-141
(SoftmaxDistributionInterpreter, == the old "Action8"). Fixed volume, softmax
distribution over log types, per-rule trigger via ceil(diversity), and a malicious
"diversity" (variant count) that is what actually drives extra alerts.

Kept as a single pure function so a fixed action always yields the same plan
(deterministic — testable). No Splunk contact here; rendering/injection lives in
measure.LiveMeasurer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .rules import LogType, RELEVANT_LOGTYPES, logtype_key


@dataclass
class InjectionPlan:
    # Total injected count per top log type (drives the fake distribution / KL).
    per_logtype_count: Dict[LogType, int]
    # Malicious variant count per rule-triggering log type (drives extra alerts).
    diversity: Dict[LogType, int]
    # Original "<source>_<eventcode>_<is_trigger>" plan for live rendering.
    logs_to_inject: Dict[str, Dict] = field(default_factory=dict)
    total_inserted: int = 0


def action_dim(top_logtypes: List[LogType]) -> int:
    """Box([0,1]^(N+M)); N = top log types, M = relevant (triggering) log types."""
    return len(top_logtypes) + len(RELEVANT_LOGTYPES)


def decode_action(raw_action: np.ndarray,
                  top_logtypes: List[LogType],
                  *,
                  softmax_temperature: float,
                  diversity_factor: int,
                  base_log_count: int,
                  additional_percentage: float) -> InjectionPlan:
    """Verbatim Action8 semantics. See action_interpreters.py:95-141."""
    raw_action = np.asarray(raw_action, dtype=np.float64).ravel()
    n_top = len(top_logtypes)

    dist = raw_action[:n_top]
    scaled = softmax_temperature * dist
    exp_vals = np.exp(scaled - np.max(scaled))
    dist = exp_vals / exp_vals.sum()

    diversity_list = raw_action[n_top:]
    num_logs = additional_percentage * base_log_count

    per_logtype_count: Dict[LogType, int] = {}
    diversity: Dict[LogType, int] = {}
    logs_to_inject: Dict[str, Dict] = {}
    total_inserted = 0

    for i, lt in enumerate(top_logtypes):
        log_count = int(dist[i] * num_logs)
        total_inserted += log_count
        per_logtype_count[lt] = log_count

        div = 0
        is_trigger = 0
        if lt in RELEVANT_LOGTYPES:
            idx = RELEVANT_LOGTYPES.index(lt)
            d = float(diversity_list[idx])
            if log_count == 0:
                d = 0.0
            is_trigger = int(np.ceil(d))
            div = int(d * diversity_factor) if is_trigger else 0
            div = max(1, min(div, log_count))
            if is_trigger:
                diversity[lt] = div

        key = logtype_key(lt, is_trigger)
        logs_to_inject[key] = {"count": log_count, "diversity": div}
        if lt in RELEVANT_LOGTYPES:
            opp = logtype_key(lt, 1 - is_trigger)
            logs_to_inject[opp] = {"count": 0, "diversity": 0}

    return InjectionPlan(
        per_logtype_count=per_logtype_count,
        diversity=diversity,
        logs_to_inject=logs_to_inject,
        total_inserted=total_inserted,
    )
