"""Measurement: turn accumulated episode state into per-rule CPU and alert counts.

Two implementations behind one tiny interface:
  - MockMeasurer  : per-rule sklearn CPU regressors + predicted alerts. No Splunk.
  - LiveMeasurer  : render + inject logs, run the saved searches, read real CPU.

Both compare a *baseline* (no injection) against the *current* (with injection)
using the SAME estimator, so the energy signal is internally consistent — unlike
the legacy mock path, which compared a model estimate against a real measurement
(wrappers/reward.py:793-804).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Protocol

import joblib
import numpy as np

from .rules import (LogType, RULES, RULE_NAMES, RULE_LOGTYPE, EXPECTED_ALERTS,
                    DIVERSITY_INSENSITIVE_RULES)

logger = logging.getLogger(__name__)


@dataclass
class Measurement:
    baseline_cpu: float
    current_cpu: float
    baseline_alert: Dict[str, float]
    current_alert: Dict[str, float]


class Measurer(Protocol):
    def measure(self, env) -> Measurement:
        """Given the env at episode end, return baseline vs current CPU + alerts.

        The env exposes everything a backend needs: `ac_real`, `ac_fake`,
        `episode_diversity`, `top_logtypes`, and `episode_plans`
        (list of (logs_to_inject, w_start, w_end) for the live renderer).
        """
        ...


def _rule_diversity(diversity: Dict[LogType, int], rule: str) -> int:
    return int(diversity.get(RULE_LOGTYPE[rule], 0))


def predicted_alerts(diversity: Dict[LogType, int]) -> Dict[str, float]:
    """current_alert = expected_baseline + injected_diversity (per AlertPredictor).

    Diversity-insensitive rules (e.g. Rapid Auth) do not gain alerts from variants.
    """
    out = {}
    for rule in RULE_NAMES:
        base = EXPECTED_ALERTS[rule]
        if rule in DIVERSITY_INSENSITIVE_RULES:
            out[rule] = base
        else:
            out[rule] = base + _rule_diversity(diversity, rule)
    return out


class MockMeasurer:
    """Per-rule CPU regressors (src/cpu_model_<rule>.joblib), predicted alerts."""

    def __init__(self, cpu_models_dir: str,
                 fake_dist_normalizer: float, alert_normalizer: float):
        self.fake_dist_normalizer = fake_dist_normalizer
        self.alert_normalizer = alert_normalizer
        self.models = {}
        for rule in RULE_NAMES:
            path = os.path.join(cpu_models_dir, f"cpu_model_{rule}.joblib")
            self.models[rule] = joblib.load(path)
        n_expected = getattr(next(iter(self.models.values())), "n_features_in_", None)
        logger.info("Loaded %d CPU regressors (expect %s features = N_logtypes + 1)",
                    len(self.models), n_expected)
        self._n_features = n_expected

    def _dist_vector(self, ac: Dict[LogType, int], top_logtypes: List[LogType]) -> np.ndarray:
        return np.array([ac.get(lt, 0) for lt in top_logtypes], dtype=np.float64)

    def _estimate_cpu(self, dist_vec: np.ndarray, alerts: Dict[str, float]) -> float:
        dist = (dist_vec / self.fake_dist_normalizer).reshape(1, -1)
        total = 0.0
        for rule in RULE_NAMES:
            a = (alerts[rule] - 1) / self.alert_normalizer
            x = np.concatenate([dist, np.array([[a]])], axis=1)
            total += float(self.models[rule].predict(x)[0])
        return total

    def measure(self, env) -> Measurement:
        baseline_alert = dict(EXPECTED_ALERTS)
        current_alert = predicted_alerts(env.episode_diversity)
        real_vec = self._dist_vector(env.ac_real, env.top_logtypes)
        fake_vec = self._dist_vector(env.ac_fake, env.top_logtypes)
        baseline_cpu = self._estimate_cpu(real_vec, baseline_alert)
        current_cpu = self._estimate_cpu(fake_vec, current_alert)
        return Measurement(baseline_cpu, current_cpu, baseline_alert, current_alert)


def make_measurer(cfg) -> "Measurer":
    """Pick the measurement backend from cfg.live — the one and only mock/live switch.

    The live backend lives in simple/live.py and is imported lazily so the mock
    path never pulls in the old Splunk-coupled stack.
    """
    if cfg.live:
        from .live import LiveMeasurer
        return LiveMeasurer(cfg)
    return MockMeasurer(cfg.cpu_models_dir, cfg.fake_dist_normalizer, cfg.alert_normalizer)
