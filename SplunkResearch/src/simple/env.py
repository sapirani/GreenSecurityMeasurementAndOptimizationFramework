"""The environment — one flat gym.Env, no wrapper stack.

An episode is `total_steps` steps (12). Each step the agent picks a distribution
+ per-rule trigger/diversity; the plan is accumulated. Real historical log counts
for the step's time window are added to the running distributions. At the final
step we measure (mock or live) and compute the reward.

Shared state is plain instance attributes — no EpisodeSharedState, no property
aliases, no stack-walking. What you read in a debugger is the whole state.
"""
from __future__ import annotations

import datetime
import logging
import random
from typing import Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import SimpleConfig
from .injection import action_dim, decode_action
from .measure import make_measurer, Measurer
from .realdist import RealDistributionProvider
from .reward import compute_reward
from .rules import LogType, RULE_LOGTYPE, load_top_logtypes

logger = logging.getLogger(__name__)

_FMT = "%m/%d/%Y:%H:%M:%S"
_VOLUME_NORM = 500000.0  # matches state_interpreters.py volume normalization


class SimpleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: SimpleConfig, eval: bool = False):
        super().__init__()
        self.cfg = cfg
        self.eval = eval

        self.top_logtypes: List[LogType] = load_top_logtypes(
            cfg.top_logtypes_csv, max_logtypes=cfg.max_logtypes)
        self.N = len(self.top_logtypes)
        self.total_steps = cfg.total_steps

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(action_dim(self.top_logtypes),), dtype=np.float32)
        obs_dim = 2 * self.N + 3
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.realdist = RealDistributionProvider(cfg.dist_csv)
        self.measurer: Measurer = make_measurer(cfg)

        self._starts = self._generate_starts()
        self._rng = random.Random(cfg.seed)
        if not eval:
            self._rng.shuffle(self._starts)
        self._start_ptr = 0

        self._episode_start: datetime.datetime = self._starts[0]
        self._reset_accumulators()

    # ------------------------------------------------------------------
    def _generate_starts(self) -> List[datetime.datetime]:
        start = datetime.datetime.strptime(self.cfg.fake_start_datetime, _FMT)
        end = datetime.datetime.strptime(self.cfg.end_time, _FMT)
        window = datetime.timedelta(minutes=self.cfg.window_size_min)
        stride = datetime.timedelta(minutes=self.cfg.rule_frequency_min)
        out, cur = [], start
        while cur + window <= end:
            out.append(cur)
            cur += stride
        if not out:
            out = [start]
        return out

    def _reset_accumulators(self):
        self.step_counter = 0
        self.ac_real: Dict[LogType, int] = {lt: 0 for lt in self.top_logtypes}
        self.ac_fake: Dict[LogType, int] = {lt: 0 for lt in self.top_logtypes}
        self.episode_diversity: Dict[LogType, int] = {}
        self.episode_plans = []  # list of (logs_to_inject, w_start, w_end) for live injection
        self.total_episode_logs = 0
        self.episodic_inserted_logs = 0
        self.ac_real_state = np.zeros(self.N, dtype=np.float64)
        self.ac_fake_state = np.zeros(self.N, dtype=np.float64)

    def _action_window(self, step_idx: int):
        dur = datetime.timedelta(seconds=self.cfg.action_duration_s)
        s = self._episode_start + step_idx * dur
        return s, s + dur

    def _norm_state(self, ac: Dict[LogType, int]) -> np.ndarray:
        vec = np.array([ac[lt] for lt in self.top_logtypes], dtype=np.float64)
        total = vec.sum()
        return vec / total if total > 0 else vec

    def _build_obs(self) -> np.ndarray:
        volume = self.total_episode_logs / _VOLUME_NORM
        volume_with_injected = (self.episodic_inserted_logs + self.total_episode_logs) / _VOLUME_NORM
        step_ratio = self.step_counter / self.total_steps
        obs = np.concatenate([
            self.ac_real_state,
            self.ac_fake_state,
            np.array([volume, volume_with_injected, step_ratio], dtype=np.float64),
        ])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        self._episode_start = self._starts[self._start_ptr % len(self._starts)]
        self._start_ptr += 1
        self._reset_accumulators()
        return self._build_obs(), {"episode_start": self._episode_start.strftime(_FMT)}

    def step(self, action):
        cfg = self.cfg
        plan = decode_action(
            action, self.top_logtypes,
            softmax_temperature=cfg.softmax_temperature,
            diversity_factor=cfg.diversity_factor,
            base_log_count=cfg.base_log_count,
            additional_percentage=cfg.additional_percentage,
        )

        w_start, w_end = self._action_window(self.step_counter)
        real_counts = self.realdist.window_counts(w_start, w_end, self.top_logtypes)

        for lt in self.top_logtypes:
            r = real_counts.get(lt, 0)
            inj = plan.per_logtype_count.get(lt, 0)
            self.ac_real[lt] += r
            self.ac_fake[lt] += r + inj
        self.total_episode_logs += sum(real_counts.values())
        self.episodic_inserted_logs += plan.total_inserted
        for lt, div in plan.diversity.items():
            self.episode_diversity[lt] = max(self.episode_diversity.get(lt, 0), div)
        self.episode_plans.append((plan.logs_to_inject, w_start, w_end))

        self.step_counter += 1
        self.ac_real_state = self._norm_state(self.ac_real)
        self.ac_fake_state = self._norm_state(self.ac_fake)

        done = self.step_counter >= self.total_steps
        reward = 0.0
        info = {"episode_start": self._episode_start.strftime(_FMT),
                "step": self.step_counter}

        if done:
            m = self.measurer.measure(self)
            rb = compute_reward(m, self.ac_real_state, self.ac_fake_state, cfg)
            reward = rb.reward
            info.update({
                "reward": rb.reward,
                "energy_raw": rb.energy_raw, "energy_term": rb.energy_term,
                "alert_metric": rb.alert_metric, "alert_penalty": rb.alert_penalty,
                "kl": rb.kl, "kl_penalty": rb.kl_penalty,
                "baseline_cpu": rb.baseline_cpu, "current_cpu": rb.current_cpu,
                "injected_logs": self.episodic_inserted_logs,
                "n_triggered_rules": len(self.episode_diversity),
            })

        return self._build_obs(), reward, done, False, info

    def close(self):
        m = getattr(self, "measurer", None)
        if m is not None and hasattr(m, "close"):
            m.close()
