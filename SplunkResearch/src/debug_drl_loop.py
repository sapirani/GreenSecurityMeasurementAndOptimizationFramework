#!/usr/bin/env python3
"""
Debug script: manually step through the DRL environment and print
state → action → new_state transitions with time windows.

Usage:
    cd SplunkResearch/src
    python debug_drl_loop.py [--steps N] [--episodes E]

This uses mock mode (no live Splunk) and the same wrapper stack
as ExperimentManager.create_environment().
"""

import argparse
import logging
import sys
import os
import textwrap

import numpy as np
import pandas as pd

# Ensure our src/, custom_splunk, and project root are on the path
_src_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_src_dir))
sys.path.insert(0, _src_dir)
sys.path.insert(0, os.path.join(_src_dir, 'custom_splunk'))
sys.path.insert(0, _project_root)

from gymnasium import make
from config import config
from custom_splunk.envs.custom_splunk_env import SplunkConfig
from wrappers.action import create_action_wrapper
from wrappers.state import create_state_wrapper
from wrappers.reward import (
    BaseRuleExecutionWrapperWithPrediction,
    EnergyRewardWrapper,
    create_alert_reward_wrapper,
    ENUM_DISTRIBUTION_REWARD_METHODS,
)
from time_manager import TimeWrapper

# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def fmt_array(arr, max_items=10):
    """Format a numpy array compactly."""
    if arr is None:
        return "None"
    if len(arr) <= max_items:
        return np.array2string(arr, precision=6, suppress_small=True, separator=', ')
    head = np.array2string(arr[:max_items // 2], precision=6, suppress_small=True, separator=', ')
    tail = np.array2string(arr[-max_items // 2:], precision=6, suppress_small=True, separator=', ')
    return f"{head[:-1]}, ..., {tail[1:]}  (len={len(arr)})"


def print_state(label, obs, env):
    """Print observation + underlying distribution state."""
    base = env.unwrapped
    tw = base.time_manager

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"  Episode window : {tw.current_window.start} → {tw.current_window.end}")
    print(f"  Action window  : {tw.action_window.start} → {tw.action_window.end}")
    print(f"  Step counter   : {base.step_counter} / {base.total_steps}")
    print(f"  Done           : {base.done}")
    print(f"  Obs shape      : {obs.shape}")
    print(f"  Obs            : {fmt_array(obs)}")

    # Underlying distribution vectors
    print(f"\n  --- Accumulated distributions (normalized) ---")
    print(f"  ac_real_state  : {fmt_array(base.ac_real_state)}")
    print(f"  ac_fake_state  : {fmt_array(base.ac_fake_state)}")

    # Relevant distribution dicts (subset)
    if base.real_relevant_distribution:
        items = list(base.real_relevant_distribution.items())[:5]
        print(f"  real_relevant  : {dict(items)} ...")
    if base.fake_relevant_distribution:
        items = list(base.fake_relevant_distribution.items())[:5]
        print(f"  fake_relevant  : {dict(items)} ...")

    print(f"  episodic_inserted_logs: {base.episodic_inserted_logs}")

    # Raw distribution dicts (pre-normalization)
    print(f"\n  --- Raw accumulated distributions ---")
    ac_real_vals = [base.ac_real_distribution[lt] for lt in base.top_logtypes]
    ac_fake_vals = [base.ac_fake_distribution[lt] for lt in base.top_logtypes]
    print(f"  ac_real_dist sum : {sum(ac_real_vals)}")
    print(f"  ac_fake_dist sum : {sum(ac_fake_vals)}")
    # Show top-5 non-zero entries
    nonzero_real = [(base.logtype_key_cache[lt], base.ac_real_distribution[lt])
                    for lt in base.top_logtypes if base.ac_real_distribution[lt] > 0]
    nonzero_fake = [(base.logtype_key_cache[lt], base.ac_fake_distribution[lt])
                    for lt in base.top_logtypes if base.ac_fake_distribution[lt] > 0]
    print(f"  ac_real non-zero : {nonzero_real[:5]} ({'...' if len(nonzero_real) > 5 else ''})")
    print(f"  ac_fake non-zero : {nonzero_fake[:5]} ({'...' if len(nonzero_fake) > 5 else ''})")
    # Check if fake == real
    if ac_real_vals == ac_fake_vals:
        print(f"  ac_fake == ac_real : YES")
    else:
        diffs = [(base.logtype_key_cache[lt], base.ac_real_distribution[lt], base.ac_fake_distribution[lt])
                 for lt in base.top_logtypes
                 if base.ac_real_distribution[lt] != base.ac_fake_distribution[lt]]
        print(f"  ac_fake == ac_real : NO ({len(diffs)} diffs, first 5: {diffs[:5]})")
    print()


def print_action(raw_action, logs_to_inject, env):
    """Print the raw NN action and interpreted injection plan."""
    print(f"  --- Action ---")
    print(f"  Raw action     : {fmt_array(raw_action)}")

    # Show the injection plan (top 5 entries by count)
    if logs_to_inject:
        sorted_logs = sorted(logs_to_inject.items(), key=lambda x: -x[1].get('count', 0))
        print(f"  Injection plan ({len(sorted_logs)} entries, top 5):")
        for key, val in sorted_logs[:5]:
            print(f"    {key:40s} count={val['count']:6d}  diversity={val['diversity']}")
        total = sum(v['count'] for v in logs_to_inject.values())
        print(f"    {'TOTAL':40s} count={total:6d}")
    print()


def print_info_summary(info):
    """Print key info dict entries."""
    keys_of_interest = [
        'done', 'step', 'total_steps',
        'ac_distribution_value', 'ac_distribution_reward',
        'predicted_alert_reward', 'alert_reward', 'norm_alert_reward',
        'energy_reward', 'norm_energy_reward',
        'inserted_logs', 'episodic_inserted_logs',
        'distribution_reward',
    ]
    print(f"  --- Info (selected) ---")
    for k in keys_of_interest:
        if k in info:
            v = info[k]
            if isinstance(v, float):
                print(f"    {k:30s}: {v:.6f}")
            else:
                print(f"    {k:30s}: {v}")
    print()


# ---------------------------------------------------------------------------
# Environment creation (mirrors ExperimentManager.create_environment)
# ---------------------------------------------------------------------------

def create_debug_env():
    """Build the full wrapper stack in mock mode."""
    # Load top logtypes
    top_logtypes_df = pd.read_csv(config.get('paths.top_logtypes'))
    log_types = config.get('environment.log_types', ['wineventlog:security', 'wineventlog:system'])
    max_lt = config.get('environment.max_logtypes', 20)
    top_logtypes_df = top_logtypes_df[top_logtypes_df['source'].str.lower().isin(log_types)]
    top_logtypes = top_logtypes_df.sort_values(by='count', ascending=False)[['source', 'EventCode']].values.tolist()[:max_lt]
    top_logtypes = [(x[0].lower(), str(x[1])) for x in top_logtypes]

    env_config = SplunkConfig(
        rule_frequency=config.get('splunk.rule_frequency', 120),
        search_window=config.get('splunk.search_window', 2880),
        logs_per_minute=config.get('splunk.logs_per_minute', 150),
        additional_percentage=config.get('environment.additional_percentage', 1.0),
        action_duration=config.get('splunk.action_duration', 14400),
        num_of_measurements=config.get('splunk.num_measurements', 1),
        baseline_num_of_measurements=config.get('splunk.baseline_num_measurements', 1),
        env_id=config.get('splunk.train_env_id', 'splunk_train-v32'),
        end_time=config.get('splunk.end_time', '08/01/2025:00:00:00'),
        is_test=True,
        ip=1,
    )

    base_dir = config.get('paths.splunk_research_dir')
    baseline_dir = os.path.join(base_dir, 'host_debug_experiments', 'baseline')
    os.makedirs(baseline_dir, exist_ok=True)

    env = make(
        id=env_config.env_id,
        config=env_config,
        top_logtypes=top_logtypes,
        baseline_dir=baseline_dir,
    )

    action_type = config.get('environment.action_type', 'Action8')
    env = create_action_wrapper(env, action_type, test_random=False)

    dist_method = config.get('reward.distribution_method', 'DistributionRewardWrapper')
    env = ENUM_DISTRIBUTION_REWARD_METHODS[dist_method](
        env,
        gamma=config.get('reward.gamma', 0.2),
        epsilon=config.get('reward.distribution_epsilon',
                           config.get('reward.epsilon', 1e-8)),
        distribution_freq=config.get('reward.distribution_freq', 1),
        distribution_threshold=config.get('reward.distribution_threshold', 0.22),
    )

    env = BaseRuleExecutionWrapperWithPrediction(
        env,
        is_mock=True,
        use_energy=config.get('reward.use_energy_reward', True),
        use_alert=config.get('reward.use_alert_reward', True),
        is_train=True,
        is_eval=False,
    )

    use_stationary_scaling = config.get('reward.use_stationary_scaling', False)
    env = EnergyRewardWrapper(env, alpha=config.get('reward.alpha', 0.5), is_mock=True,
                              use_stationary_scaling=use_stationary_scaling)

    alert_method = config.get('reward.alert_method', 'AlertRewardWrapper')
    env = create_alert_reward_wrapper(
        env,
        method_name=alert_method,
        beta=config.get('reward.beta', 0.5),
        epsilon=config.get('reward.alert_epsilon',
                           config.get('reward.epsilon', 1e-8)),
        normalizer_factor=config.get('reward.normalizer_factor', 10),
        use_stationary_scaling=use_stationary_scaling,
    )

    env = TimeWrapper(env)

    state_type = config.get('environment.state_type', 'StateWrapper7')
    hosts_pct = config.get('environment.hosts_percentage', 100)
    env = create_state_wrapper(env, state_type, hosts_pct)

    return env


# ---------------------------------------------------------------------------
# Main debug loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Debug DRL state-action-state transitions')
    parser.add_argument('--steps', type=int, default=3,
                        help='Number of steps per episode to trace (default: 3)')
    parser.add_argument('--episodes', type=int, default=2,
                        help='Number of episodes to trace (default: 2)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
        stream=sys.stderr,
    )

    print("Creating environment (mock mode) ...")
    env = create_debug_env()
    base = env.unwrapped

    # Load pre-fetched distribution CSV (same as experiment_manager_new.py line 579)
    import datetime as _dt
    base.splunk_tools.load_real_logs_distribution_bucket(
        _dt.datetime.strptime(base.time_manager.first_start_datetime, '%m/%d/%Y:%H:%M:%S'),
        _dt.datetime.strptime(base.time_manager.end_time, '%m/%d/%Y:%H:%M:%S')
    )

    print(f"  top_logtypes     : {len(base.top_logtypes)} types")
    print(f"  relevant_logtypes: {len(base.relevant_logtypes)} types")
    print(f"  total_steps/ep   : {base.total_steps}")
    print(f"  action_space     : {env.action_space}")
    print(f"  obs_space        : {env.observation_space}")
    print(f"  action_type      : {config.get('environment.action_type', 'Action8')}")
    print(f"  state_type       : {config.get('environment.state_type', 'StateWrapper7')}")

    # Access the Action wrapper to intercept its plan
    from wrappers.action import Action
    action_wrapper = env
    while action_wrapper is not None:
        if isinstance(action_wrapper, Action):
            break
        action_wrapper = getattr(action_wrapper, 'env', None)

    for ep in range(args.episodes):
        print(f"\n{'#'*72}")
        print(f"  EPISODE {ep + 1}")
        print(f"{'#'*72}")

        obs, info = env.reset()
        print_state(f"INITIAL STATE (after reset)", obs, env)

        steps_to_trace = min(args.steps, base.total_steps)

        for step_i in range(steps_to_trace):
            # Sample a random action
            raw_action = env.action_space.sample()

            # Print the action plan (use the Action wrapper's interpret method)
            ctx = action_wrapper._build_context()
            plan = action_wrapper.interpreter.interpret(raw_action, ctx)
            print_action(raw_action, plan.logs_to_inject, env)

            # Step
            new_obs, reward, terminated, truncated, info = env.step(raw_action)

            print(f"  Reward: {reward:.6f}  terminated={terminated}  truncated={truncated}")
            print_info_summary(info)
            print_state(f"STATE after step {step_i + 1}", new_obs, env)

            if terminated or truncated:
                print(f"  >>> Episode ended at step {step_i + 1}")
                break

            obs = new_obs

    env.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
