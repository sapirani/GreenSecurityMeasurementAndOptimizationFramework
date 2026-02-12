"""Reward interpreters: strategy objects for alert reward computation.

Two orthogonal axes:
  - **AlertSignal**: how to compute the raw alert reward (predicted vs relative-diff)
  - **AlertNormalizer**: how to normalize the raw reward (tanh, identity, scaled-tanh)

The base ``AlertRewardWrapper`` composes one of each, eliminating the need
for four separate numbered wrapper classes.

Following the pattern of ``action_interpreters.py`` / ``state_interpreters.py``.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from config import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert Signal ABCs
# ---------------------------------------------------------------------------

class AlertSignal(ABC):
    """How to compute the raw (un-normalized) alert reward."""

    @abstractmethod
    def compute(self, info: Dict, expected_alerts: Dict, epsilon: float) -> float:
        ...


class PredictedAlertSignal(AlertSignal):
    """Use the predicted alert reward from ``BaseRuleExecutionWrapperWithPrediction``.

    Corresponds to the original AlertRewardWrapper / AlertRewardWrapper1 logic.
    """

    def compute(self, info: Dict, expected_alerts: Dict, epsilon: float) -> float:
        return info.get('predicted_alert_reward', 0)


class RelativeDiffAlertSignal(AlertSignal):
    """Per-rule relative difference between current and baseline alerts.

    Corresponds to the original AlertRewardWrapper2 / AlertRewardWrapper3 logic.

    Uses measured baselines from ``info['raw_baseline_metrics']`` (set by
    ``BaseRuleExecutionWrapperWithPrediction`` before ``AlertRewardWrapper``
    runs).  Falls back to ``fallback_baseline_alerts`` (config-derived) when
    the measured data is unavailable.
    """

    def __init__(self, fallback_baseline_alerts: Dict[str, float]):
        self.fallback_baseline_alerts = fallback_baseline_alerts

    def compute(self, info: Dict, expected_alerts: Dict, epsilon: float) -> float:
        measured_baselines = info.get('raw_baseline_metrics', {})
        current_rules_alerts = {
            rule: info['raw_metrics'][rule]['alert'] for rule in expected_alerts
        }
        alert_diffs = []
        for rule in expected_alerts:
            measured = measured_baselines.get(rule, {}).get('alert')
            baseline_alert = measured if measured is not None else self.fallback_baseline_alerts.get(rule, 0)
            alert_diff = (current_rules_alerts[rule] - baseline_alert) / (baseline_alert + epsilon)
            alert_diffs.append(alert_diff)
        if alert_diffs:
            return min(0, -np.mean(alert_diffs))
        return 0


class ExpectedDiffAlertSignal(AlertSignal):
    """Per-rule relative difference between current alerts and expected (config) values.

    Similar to ``RelativeDiffAlertSignal`` but always compares against the
    expected alert counts from config rather than real measured baselines.
    This removes dependence on measured baseline quality and provides a
    stable reference point across episodes.
    """

    def compute(self, info: Dict, expected_alerts: Dict, epsilon: float) -> float:
        current_rules_alerts = {
            rule: info['raw_metrics'][rule]['alert'] for rule in expected_alerts
        }
        alert_diffs = []
        for rule in expected_alerts:
            expected_alert = expected_alerts[rule]['alert']
            alert_diff = (current_rules_alerts[rule] - expected_alert) / (expected_alert + epsilon)
            alert_diffs.append(alert_diff)
        if alert_diffs:
            return min(0, -np.mean(alert_diffs))
        return 0


# ---------------------------------------------------------------------------
# Alert Normalizer ABCs
# ---------------------------------------------------------------------------

class AlertNormalizer(ABC):
    """How to normalize a raw alert reward into a bounded signal."""

    @abstractmethod
    def normalize(self, raw_reward: float, normalizer_factor: float) -> float:
        ...


class TanhNormalizer(AlertNormalizer):
    """``scale * tanh(-r / factor)``.

    With default ``scale=-1`` this gives ``-tanh(-r / factor)`` (original AlertRewardWrapper).
    With a custom scale (e.g. ``-2`` from config) it reproduces the original AlertRewardWrapper2.

    Args:
        tanh_scale: Multiplicative scale applied to tanh (default ``-1``).
    """

    def __init__(self, tanh_scale: float = -1):
        self.tanh_scale = tanh_scale

    def normalize(self, raw_reward: float, normalizer_factor: float) -> float:
        return self.tanh_scale * np.tanh(-raw_reward / normalizer_factor)


def ScaledTanhNormalizer(tanh_scale: float = None) -> TanhNormalizer:
    """Factory for backward compatibility: TanhNormalizer with config-driven scale."""
    if tanh_scale is None:
        tanh_scale = config.get('reward.tanh_scale_factor', -2)
    return TanhNormalizer(tanh_scale=tanh_scale)


class IdentityNormalizer(AlertNormalizer):
    """Pass-through  (original AlertRewardWrapper1 / AlertRewardWrapper3)."""

    def normalize(self, raw_reward: float, normalizer_factor: float) -> float:
        return raw_reward


# ---------------------------------------------------------------------------
# Energy Normalizer ABCs
# ---------------------------------------------------------------------------

class EnergyNormalizer(ABC):
    """How to normalize a raw energy reward into a bounded signal."""

    @abstractmethod
    def normalize(self, raw: float) -> float:
        ...


class TanhEnergyNormalizer(EnergyNormalizer):
    """Original: tanh(raw * scale). Saturates quickly at large values."""

    def __init__(self, scale: float = 1.5):
        self.scale = scale

    def normalize(self, raw: float) -> float:
        return np.tanh(raw * self.scale)


class Log1pEnergyNormalizer(EnergyNormalizer):
    """log(1+x)-based: maintains gradient at large values.

    Normalizes to [0, 1] by dividing by log1p(ceiling * scale).
    """

    def __init__(self, scale: float = 1.0, ceiling: float = 10.0):
        self.scale = scale
        self._denom = np.log1p(ceiling * scale)

    def normalize(self, raw: float) -> float:
        return np.log1p(raw * self.scale) / self._denom


ENERGY_NORMALIZER_REGISTRY = {
    'tanh': TanhEnergyNormalizer,
    'log1p': Log1pEnergyNormalizer,
}


# ---------------------------------------------------------------------------
# Alert Reward Registry
# ---------------------------------------------------------------------------

def _build_expected_baseline() -> Dict[str, float]:
    """Helper: flatten expected_alerts to {rule: alert_value}."""
    from wrappers.reward import expected_alerts
    return {rule: expected_alerts[rule]['alert'] for rule in expected_alerts}


ALERT_REWARD_REGISTRY: Dict[str, tuple] = {
    # Backward-compatible numbered names
    'AlertRewardWrapper':  (PredictedAlertSignal,    TanhNormalizer),
    'AlertRewardWrapper1': (PredictedAlertSignal,    IdentityNormalizer),
    'AlertRewardWrapper2': (RelativeDiffAlertSignal, ScaledTanhNormalizer),
    'AlertRewardWrapper3': (RelativeDiffAlertSignal, IdentityNormalizer),
    # Descriptive aliases
    'PredictedTanh':             (PredictedAlertSignal,    TanhNormalizer),
    'PredictedIdentity':         (PredictedAlertSignal,    IdentityNormalizer),
    'RelativeDiffScaledTanh':    (RelativeDiffAlertSignal, ScaledTanhNormalizer),
    'RelativeDiffIdentity':      (RelativeDiffAlertSignal, IdentityNormalizer),
    'ExpectedDiffScaledTanh':    (ExpectedDiffAlertSignal, ScaledTanhNormalizer),
    'ExpectedDiffIdentity':      (ExpectedDiffAlertSignal, IdentityNormalizer),
}
