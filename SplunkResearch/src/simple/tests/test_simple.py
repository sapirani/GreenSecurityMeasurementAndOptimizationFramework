"""Verification tests for the simple package. Run:

    python -m pytest SplunkResearch/src/simple/tests/test_simple.py -q
    # or, without pytest:
    python SplunkResearch/src/simple/tests/test_simple.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simple.config import SimpleConfig
from simple.injection import decode_action, action_dim
from simple.rules import load_top_logtypes, RELEVANT_LOGTYPES
from simple.env import SimpleEnv


def _top():
    cfg = SimpleConfig()
    return load_top_logtypes(cfg.top_logtypes_csv, max_logtypes=cfg.max_logtypes)


def test_decode_is_deterministic():
    top = _top()
    a = np.linspace(0, 1, action_dim(top)).astype(np.float32)
    kw = dict(softmax_temperature=20.0, diversity_factor=30,
              base_log_count=20000, additional_percentage=1.0)
    p1 = decode_action(a, top, **kw)
    p2 = decode_action(a, top, **kw)
    assert p1.per_logtype_count == p2.per_logtype_count
    assert p1.diversity == p2.diversity
    assert p1.total_inserted == p2.total_inserted


def test_decode_volume_and_diversity_bounds():
    top = _top()
    a = np.ones(action_dim(top), dtype=np.float32)
    plan = decode_action(a, top, softmax_temperature=20.0, diversity_factor=30,
                         base_log_count=20000, additional_percentage=1.0)
    # Total injected never exceeds the fixed budget.
    assert plan.total_inserted <= 20000
    # Diversity only for relevant logtypes, clamped to [1, log_count].
    for lt, div in plan.diversity.items():
        assert lt in RELEVANT_LOGTYPES
        assert 1 <= div <= plan.per_logtype_count[lt]


def test_reward_integrity_and_determinism():
    """The reward returned at episode end is exactly what info['reward'] reports,
    and a fixed env+action reproduces the same reward (no RNG in the reward path)."""
    cfg = SimpleConfig()
    env = SimpleEnv(cfg, eval=True)
    a = np.full(env.action_space.shape[0], 0.5, dtype=np.float32)

    def run():
        env.reset()
        done, last_r, last_info = False, None, {}
        while not done:
            _, r, done, _, info = env.step(a)
            last_r, last_info = r, info
        return last_r, last_info

    r1, info1 = run()
    assert info1["reward"] == r1                       # logged == returned
    env2 = SimpleEnv(cfg, eval=True)
    a2 = a
    env2.reset()
    done, r2 = False, None
    while not done:
        _, r2, done, _, _ = env2.step(a2)
    assert r1 == r2                                     # reproducible


if __name__ == "__main__":
    test_decode_is_deterministic()
    test_decode_volume_and_diversity_bounds()
    test_reward_integrity_and_determinism()
    print("All simple/ tests passed.")
