"""The reward: one formula, computed once at episode end.

    reward = alpha * log1p(energy)                        # maximize detection CPU cost
           - beta  * tanh_hinge(alert, tau_a, sens_a)     # stealth: don't spike alerts
           - gamma * tanh_hinge(kl,    tau_k, sens_k)     # stealth: keep the log mix plausible

The stealth penalties are tanh-bounded to [0, 1] so a large violation cannot swamp
the energy term (an unbounded relu makes the tradeoff unlearnable — verified).

No Lagrange multipliers, no curriculum, no quota, no reward-mode switch, no
normalizer registries. The scalar this returns is exactly what gets logged — there
is no separate "diagnostic" reward that silently disagrees with it
(the trap in wrappers/reward.py:839).

Energy signal ported from wrappers/reward.py:804; alert from :743-748; KL from :697-704.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .measure import Measurement
from .rules import RULE_NAMES


@dataclass
class RewardBreakdown:
    reward: float
    energy_raw: float
    energy_term: float
    alert_metric: float
    alert_penalty: float
    kl: float
    kl_penalty: float
    baseline_cpu: float
    current_cpu: float


def tanh_hinge(metric: float, threshold: float, sensitivity: float) -> float:
    """Bounded hinge penalty in [0, 1]: tanh(max(0, metric - threshold) / sensitivity).

    The bound is load-bearing: it caps each stealth penalty at 1.0 so a large
    alert/KL violation cannot swamp the energy term and make the tradeoff
    unlearnable. (Ported from wrappers/reward.py:711-720 — an unbounded relu here
    collapses training, as an early version of this file demonstrated.)
    """
    s = sensitivity if sensitivity > 0 else 1.0
    return float(np.tanh(max(metric - threshold, 0.0) / s))


def kl_divergence(real_state: np.ndarray, fake_state: np.ndarray, eps: float) -> float:
    """KL(real || fake) over the accumulated, ε-smoothed log-type distributions."""
    if real_state.size == 0 or fake_state.size == 0:
        return 0.0
    p = (real_state + eps) / np.sum(real_state + eps)
    q = (fake_state + eps) / np.sum(fake_state + eps)
    return float(np.sum(p * np.log(np.clip(p / q, 1e-10, 1e10))))


def compute_reward(m: Measurement,
                   ac_real_state: np.ndarray,
                   ac_fake_state: np.ndarray,
                   cfg) -> RewardBreakdown:
    # Energy: relative CPU increase from injection (>= 0), compressed by log1p.
    energy_raw = max((m.current_cpu - m.baseline_cpu) / (m.baseline_cpu + cfg.energy_epsilon), 0.0)
    energy_term = float(np.log1p(energy_raw))

    # Alert: mean per-rule relative alert increase, penalized above tau_alert.
    rel_increases = []
    for rule in RULE_NAMES:
        base = m.baseline_alert[rule]
        cur = m.current_alert[rule]
        rel_increases.append(max((cur - base) / max(base, 1.0), 0.0))
    alert_metric = float(np.mean(rel_increases)) if rel_increases else 0.0
    alert_penalty = tanh_hinge(alert_metric, cfg.tau_alert, cfg.alert_sensitivity)

    # Distribution: KL between accumulated real and real+injected mixes.
    kl = kl_divergence(ac_real_state, ac_fake_state, cfg.dist_epsilon)
    kl_penalty = tanh_hinge(kl, cfg.tau_kl, cfg.kl_sensitivity)

    reward = (cfg.alpha * energy_term
              - cfg.beta * alert_penalty
              - cfg.gamma * kl_penalty)

    return RewardBreakdown(
        reward=float(reward),
        energy_raw=energy_raw, energy_term=energy_term,
        alert_metric=alert_metric, alert_penalty=alert_penalty,
        kl=kl, kl_penalty=kl_penalty,
        baseline_cpu=m.baseline_cpu, current_cpu=m.current_cpu,
    )
