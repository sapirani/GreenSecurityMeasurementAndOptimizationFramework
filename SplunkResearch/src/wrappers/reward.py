import asyncio
import datetime
import json
from os import path
import random
from time import sleep
import gymnasium as gym
from gymnasium.core import RewardWrapper
import numpy as np
from typing import Dict, Tuple, Optional

import pandas as pd
import joblib

from env_utils import *
import logging
logger = logging.getLogger(__name__)
from time_manager import TimeWindow
from config import config

# Load configuration values
std = config.get('reward.std', 13)


def _load_expected_alerts() -> Dict[str, Dict[str, float]]:
    """Build the expected_alerts dict from config keys."""
    rule_names = config.get('reward.rule_names', [])
    alert_values = config.get('reward.expected_alerts_per_rule', [4, 0.7, 0, 4, 0, 0, 0, 0, 0])
    return {rule: {'alert': alert} for rule, alert in zip(rule_names, alert_values)}


expected_alerts = _load_expected_alerts()
class AlertPredictor:
    """Separate class to handle alert prediction logic"""

    def __init__(self, epsilon: float = None):
        self.epsilon = epsilon if epsilon is not None else config.get('reward.alert_epsilon',
                                                                       config.get('reward.epsilon', 1e-8))
        self.current_alerts = {}
        
    def predict_alert_reward(self, rule_name: str, baseline_alert: float,
                           diversity_logs: Dict, section_logtypes: Dict,
                           is_mock: bool = False) -> None:
        """Predict current alert count for a specific rule and store in self.current_alerts."""
        relevant_log = section_logtypes.get(rule_name, None)
        if relevant_log:
            relevant_log = "_".join(relevant_log[0]) + "_1"
            diversity = diversity_logs.get(relevant_log, 0)

            if rule_name in ['ESCU Windows Rapid Authentication On Multiple Hosts Rule']:
                predicted_current = baseline_alert
            else:
                predicted_current = baseline_alert + diversity
        else:
            predicted_current = baseline_alert

        self.current_alerts[rule_name] = predicted_current
    
    def predict_overall_alert_reward(self, baseline_metrics: Dict, diversity_logs: Dict,
                                   section_logtypes: Dict, is_mock: bool = False) -> float:
        """Predict overall alert reward for all rules"""
        for rule_name, expected in expected_alerts.items():
            baseline_alert = baseline_metrics.get(rule_name, {}).get('alert', 0)
            self.predict_alert_reward(
                rule_name, baseline_alert, diversity_logs,
                section_logtypes, is_mock
            )

        baseline_alerts = {rule: expected_alerts.get(rule, {}).get('alert', 0) for rule in expected_alerts}
        normal_alert_rate = sum(baseline_alerts.values())
        return min(-(sum(self.current_alerts.values()) - normal_alert_rate + self.epsilon) / (normal_alert_rate + self.epsilon), 0)


class BaseRuleExecutionWrapperWithPrediction(RewardWrapper):
    """Enhanced base wrapper with alert prediction capability"""

    def __init__(self, env, is_mock: bool = False,
                 use_energy: bool = True, use_alert: bool = True,
                 is_eval: bool = False, is_train: bool = False):
        super().__init__(env)

        self.is_mock = is_mock
        self.use_energy = use_energy
        self.use_alert = use_alert
        self.energy_models = {}
        self.injection_id = 0
        self._action_wrapper = self._find_action_wrapper()
        self.baseline_measured = False
        self.measuring = False
        self.alert_predictor = AlertPredictor()
        self.is_eval = is_eval
        self.is_train = is_train
        self._episode_counter = 0
        self._measurement_frequency = config.get('reward.measurement_frequency', 10000)

        # Load joblib models for energy consumption for each rule
        base_dir = config.get('paths.base_dir', '/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch')
        model_path = path.join(base_dir, 'src/models_all_rules_cpu.joblib')

        if path.exists(model_path):
            self.energy_models['all'] = joblib.load(model_path)

        for rule in expected_alerts:
            rule_model_path = path.join(base_dir, f'src/cpu_model_{rule}.joblib')
            self.energy_models[rule] = joblib.load(rule_model_path)

        self.distributions = []
        self.alerts = []

        self.epsilon = config.get('reward.alert_epsilon', config.get('reward.epsilon', 1e-8))
        self._baseline_cache = {}  # {(start_time, end_time): DataFrame}

    def _find_action_wrapper(self):
        """Traverse the wrapper stack to find the Action wrapper."""
        env = self.env
        while env is not None:
            if hasattr(env, 'inject_episodic_logs'):
                return env
            env = getattr(env, 'env', None)
        return None

    def _get_cached_baseline(self, time_range):
        """Return baseline rows for *time_range*, using a dict cache to avoid O(n) filter."""
        key = (time_range[0], time_range[1])
        if key not in self._baseline_cache:
            mask = ((self.unwrapped.baseline_df['start_time'] == key[0]) &
                    (self.unwrapped.baseline_df['end_time'] == key[1]))
            self._baseline_cache[key] = self.unwrapped.baseline_df[mask]
        return self._baseline_cache[key]

    def get_baseline_data(self, time_range: TimeWindow, rerun=False) -> Dict:
        """Get baseline data - execute after cleaning if needed"""
        num_of_measurements = self.unwrapped.config.baseline_num_of_measurements
        relevant_rows = self._get_cached_baseline(time_range)
        actual_num_of_measurements = relevant_rows.groupby(['start_time', 'end_time', 'search_name']).size().values[0] if not relevant_rows.empty else 0
        needed_measurements = num_of_measurements - actual_num_of_measurements
        
        # check if some search is missing
        running_dict = {}
        existing_searches = set()
        if len(relevant_rows) != 0:
            existing_searches = set(relevant_rows.reset_index()['search_name'].values)
            running_dict.update({search: needed_measurements for search in existing_searches})
        missing_searches = set([search for search in self.unwrapped.splunk_tools.active_saved_searches]) - existing_searches
        running_dict.update({search: num_of_measurements for search in missing_searches})
        if rerun:
            running_dict = {search: num_of_measurements for search in self.unwrapped.splunk_tools.active_saved_searches}

        if self.use_energy or self.use_alert :
            if sum(running_dict.values()) > 0:
                logger.info(f"Need to run baseline for {running_dict}")
                empty_monitored_files(get_system_monitor_path(self.unwrapped.splunk_tools.splunk_host))
                empty_monitored_files(get_security_monitor_path(self.unwrapped.splunk_tools.splunk_host))
                logger.info('Cleaning the environment')
                clean_env(self.unwrapped.splunk_tools, time_range, logs_qnt=None, host=self.unwrapped.splunk_tools.splunk_host)
                logger.info('Measure no agent reward values')
                logger.info('wait for the environment to be cleaned')
                sleep(3)
                                
                logger.info(f"Running {running_dict}")
                rules_metrics = asyncio.run(self.unwrapped.splunk_tools.run_saved_searches(time_range))
                new_lines = self.convert_metrics(time_range, rules_metrics)
                if len(new_lines) != 0:
                    if rerun:
                        # remove existing rows for the time range
                        self.unwrapped.baseline_df = self.unwrapped.baseline_df[~((self.unwrapped.baseline_df['start_time'] == time_range[0]) & (self.unwrapped.baseline_df['end_time'] == time_range[1]))]
                    self.unwrapped.baseline_df = pd.concat([self.unwrapped.baseline_df, pd.DataFrame(new_lines)])
                    # Invalidate cache for this time range since data changed
                    self._baseline_cache.pop((time_range[0], time_range[1]), None)
                self.baseline_measured = True

        random_val = np.random.randint(0, 10)
        if random_val % 3 == 0 and not self.is_mock:
            self.unwrapped.baseline_df.to_csv(self.unwrapped.baseline_path, index=False)

        return self._get_cached_baseline(time_range)
    
    def compute_baseline_and_prediction(self, time_range: TimeWindow, diversity_logs: Dict) -> Tuple[float, Dict, pd.DataFrame]:
        """Fetch baseline data and compute predicted alert reward.

        Returns:
            (predicted_reward, raw_baseline_metrics, baseline_data)
        """
        baseline_data = self.get_baseline_data(time_range)

        if baseline_data.empty:
            return None, {}, baseline_data

        grouped_baseline = baseline_data.groupby('search_name')
        raw_baseline_metrics, _ = self.process_metrics(grouped_baseline)

        predicted_reward = self.alert_predictor.predict_overall_alert_reward(
            raw_baseline_metrics,
            diversity_logs,
            self.unwrapped.section_logtypes,
            self.is_mock,
        )

        logger.info(f"Alert prediction: reward={predicted_reward:.3f}")

        return predicted_reward, raw_baseline_metrics, baseline_data
    
    def mock_rules_metrics(self, time_range: TimeWindow) -> Dict:
        """Mock rules metrics for the given time range"""
        rules_metrics = self.unwrapped.splunk_tools.mock_run_saved_searches(time_range)
        return rules_metrics
    
    def get_current_reward_values(self, time_range: TimeWindow) -> Tuple[pd.DataFrame, Dict]:
        """Get current reward values"""
        if (self.is_mock and not self.measuring) or not self.use_energy or not self.use_alert:
            rules_metrics = self.mock_rules_metrics(time_range)
        else:
            rules_metrics = asyncio.run(self.unwrapped.splunk_tools.run_saved_searches(
                time_range))
        relevant_rows = self.convert_metrics(time_range, rules_metrics)
        relevant_rows = pd.DataFrame(relevant_rows)
        grouped = relevant_rows.groupby('search_name')
        return self.process_metrics(grouped)
    
    def get_baseline_reward_values(self, time_range: TimeWindow) -> Tuple[pd.DataFrame, Dict]:
        """Get baseline reward values"""
        relevant_rows = self.get_baseline_data(time_range)
        grouped = relevant_rows.groupby('search_name')
        return self.process_metrics(grouped)
    
    def convert_metrics(self, time_range, rules_metrics):
        return [{
            'search_name': metric.search_name,
            'alert': metric.results_count,
            'duration': metric.execution_time,
            'cpu': metric.cpu,
            'start_time': metric.start_time,
            'end_time': metric.end_time,
            'read_count': metric.io_metrics['read_count'],
            'write_count': metric.io_metrics['write_count'],
            'read_bytes': metric.io_metrics['read_bytes'],
            'write_bytes': metric.io_metrics['write_bytes'],
            'memory_mb': metric.memory_mb
        } for metric in rules_metrics]

    def process_metrics(self, grouped):
        raw_metrics = {}
        for search_name, group in grouped:
            raw_metrics[search_name] = {
                'duration': group['duration'].mean(),
                'cpu': group['cpu'].mean(),
                'read_count': group['read_count'].mean(),
                'write_count': group['write_count'].mean(),
                'read_bytes': group['read_bytes'].mean(),
                'write_bytes': group['write_bytes'].mean(),
                'memory_mb': group['memory_mb'].mean(),
                'alert': group['alert'].mean()}
                
            if raw_metrics[search_name]['alert'] != round(raw_metrics[search_name]['alert']):
                logger.info(f"Alert value is not an integer: {search_name}, {raw_metrics[search_name]['alert']}, {group['alert']}")
                max_idx = group['alert'].idxmax()
                for col in ['alert', 'duration', 'cpu', 'read_count', 'write_count', 'read_bytes', 'write_bytes']:
                    raw_metrics[search_name][col] = group[col].loc[max_idx]

        combined_metrics = {
            'duration': sum([metric['duration'] for metric in raw_metrics.values()]),
            'cpu': sum([metric['cpu'] for metric in raw_metrics.values()]),
            'read_count': sum([metric['read_count'] for metric in raw_metrics.values()]),
            'write_count': sum([metric['write_count'] for metric in raw_metrics.values()]),
            'read_bytes': sum([metric['read_bytes'] for metric in raw_metrics.values()]),
            'write_bytes': sum([metric['write_bytes'] for metric in raw_metrics.values()]),
            'memory_mb': sum([metric['memory_mb'] for metric in raw_metrics.values()]),
            'alert': sum([metric['alert'] for metric in raw_metrics.values()])
        }

        
        return raw_metrics, combined_metrics
       
    def reward(self, reward: float) -> float:
        return reward

    def _validate_alert_consistency(self, alerts_diff, diversity_logs, time_range_start_epoch, time_range_end_epoch):
        """Check compatibility of alerts_diff with diversity info."""
        for rule in expected_alerts:
            if rule == 'ESCU Windows Rapid Authentication On Multiple Hosts Rule':
                continue
            relevant_log = self.unwrapped.section_logtypes.get(rule, None)
            if relevant_log:
                log_type = "_".join(relevant_log[0]) + "_1"
                diversity_value = diversity_logs.get(log_type, 0)
                if alerts_diff[rule] != diversity_value:
                    logger.warning(f"Alert difference mismatch for {rule}: alerts_diff={alerts_diff[rule]}, diversity_value={diversity_value}")
                    self._log_event_times(rule, relevant_log, time_range_start_epoch, time_range_end_epoch)
                else:
                    logger.info(f"Alert difference match for {rule}: alerts_diff={alerts_diff[rule]}, diversity_value={diversity_value}")

    def _log_event_times(self, rule_name, relevant_log, start_epoch, end_epoch):
        """Query Splunk for event times to diagnose alert mismatches."""
        default_host = config.get('splunk.default_host', 'dt-splunk')
        secondary_host = config.get('hosts.secondary', '132.72.81.150')
        query = f'index={self.unwrapped.splunk_tools.index_name} host IN ("{default_host}", {secondary_host}) EventCode={relevant_log[0][1]}  | stats count by real_ts var_id'
        results = self.unwrapped.splunk_tools.run_search(query, start_epoch, end_epoch)
        formatted_log = "\n".join([json.dumps(record) for record in results])
        logger.info(f"Event times for {rule_name} in ({start_epoch}, {end_epoch}): {formatted_log}")

    def _persist_baseline_snapshot(self, info, raw_baseline_metrics):
        """Append baseline metrics once per time-range, flush when buffer is full."""
        if len(self.unwrapped.all_baseline_data) < len(self.unwrapped.baseline_df.time_range.unique()):
            self.unwrapped.all_baseline_data.append({
                'time_range': info['current_window'],
                'ac_real_distribution': self.unwrapped.ac_real_distribution,
                'raw_metrics': raw_baseline_metrics,
            })
        else:
            all_baseline_data_df = pd.json_normalize(self.unwrapped.all_baseline_data, sep='_')
            all_baseline_data_df.to_csv(self.unwrapped.all_baseline_data_path, index=False, mode='w')
            self.unwrapped.all_baseline_data = []

    def _update_measurement_state(self):
        """Advance episode counter and decide whether to take real measurements."""
        self._episode_counter += 1
        if (self._episode_counter % self._measurement_frequency == 0 or self.is_eval) and not self.is_mock:
            logger.info("Measuring")
            self.measuring = True
        else:
            self.measuring = False
        self.unwrapped.should_delete = not self.is_mock or self.measuring

    def _inject_logs_if_needed(self, time_range_start_epoch, time_range_end_epoch):
        """Inject episodic logs into Splunk when running in live mode."""
        if (not self.is_mock or self.measuring) and self.use_energy and self.use_alert:
            if self.baseline_measured:
                logger.info("Empty deletion dict of log generator and relevant fake_splunk_state")
                self.unwrapped.log_generator.logs_to_delete = {}
                for t_r in self.unwrapped.log_generator.fake_splunk_state:
                    t_r_datetime = (datetime.datetime.strptime(t_r[0], '%m/%d/%Y:%H:%M:%S'), datetime.datetime.strptime(t_r[1], '%m/%d/%Y:%H:%M:%S'))
                    if t_r_datetime[0].timestamp() >= time_range_start_epoch and t_r_datetime[1].timestamp() <= time_range_end_epoch:
                        self.unwrapped.log_generator.fake_splunk_state[t_r] = {}
            self._action_wrapper.inject_episodic_logs(self.injection_id)
            self.injection_id += 1

    def _apply_mock_overrides(self, info, raw_metrics, combined_metrics):
        """In mock mode, substitute predicted alerts for measured ones."""
        if self.is_mock and self.measuring:
            combined_metrics['real_cpu'] = combined_metrics['cpu']

        if self.is_mock and (self.use_alert or self.use_energy):
            current_alerts = self.alert_predictor.current_alerts
            for rule in expected_alerts:
                raw_metrics[rule]['alert'] = current_alerts[rule]
            info['combined_metrics']['alert'] = sum(current_alerts.values())

    def _persist_experiment_data(self, info, raw_metrics):
        """Buffer and periodically flush experiment data to CSV."""
        if not self.is_mock and self.use_energy and self.use_alert:
            self.unwrapped.all_data.append({
                'time_range': info['current_window'],
                'ac_fake_distribution': self.unwrapped.ac_fake_distribution,
                'raw_metrics': raw_metrics,
            })
            if random.randint(0, 1000) % 10 == 0:
                if self.unwrapped.all_data:
                    all_data_df = pd.json_normalize(self.unwrapped.all_data, sep='_')
                    all_data_df.to_csv(self.unwrapped.all_data_path, index=False, mode='a', header=not path.exists(self.unwrapped.all_data_path))
                self.unwrapped.all_data = []

    def step(self, action):
        """Orchestrate episode-end reward computation."""
        obs, reward, terminated, truncated, info = super().step(action)

        if info.get('done', True):
            self.baseline_measured = False
            diversity_logs = info.get('diversity_episode_logs', {})

            predicted_alert_reward, raw_baseline_metrics, cached_baseline_data = self.compute_baseline_and_prediction(
                info['current_window'], diversity_logs
            )

            self._persist_baseline_snapshot(info, raw_baseline_metrics)

            info['predicted_alert_reward'] = predicted_alert_reward
            self._update_measurement_state()

            time_window = info['current_window']
            self._inject_logs_if_needed(time_window.start_epoch, time_window.end_epoch)

            raw_metrics, combined_metrics = self.get_current_reward_values(info['current_window'])
            baseline_raw_metrics, combined_baseline_metrics = self.process_metrics(
                cached_baseline_data.groupby('search_name')
            )

            alerts_diff = {rule: raw_metrics[rule]['alert'] - baseline_raw_metrics.get(rule, {}).get('alert', 0) for rule in expected_alerts}
            if (not self.is_mock or self.measuring) and self.use_energy and self.use_alert:
                self._validate_alert_consistency(alerts_diff, diversity_logs, time_window.start_epoch, time_window.end_epoch)

            info['combined_metrics'] = combined_metrics
            info['combined_baseline_metrics'] = combined_baseline_metrics
            self._apply_mock_overrides(info, raw_metrics, combined_metrics)
            self._persist_experiment_data(info, raw_metrics)

            if not self.use_alert:
                predicted_alert_reward = 0

            info['raw_metrics'] = raw_metrics
            info['raw_baseline_metrics'] = raw_baseline_metrics

        return obs, reward, terminated, truncated, info




from wrappers.reward_interpreters import (
    AlertSignal, AlertNormalizer,
    PredictedAlertSignal, RelativeDiffAlertSignal, ExpectedDiffAlertSignal,
    TanhNormalizer, IdentityNormalizer, ScaledTanhNormalizer, PowerAlertNormalizer,
    ALERT_REWARD_REGISTRY,
    ENERGY_NORMALIZER_REGISTRY, TanhEnergyNormalizer, PowerEnergyNormalizer,
)


class AlertRewardWrapper(RewardWrapper):
    """Unified alert reward wrapper composing a signal strategy and a normalizer."""

    def __init__(self, env: gym.Env, signal: AlertSignal, normalizer: AlertNormalizer,
                 beta: float = None, epsilon: float = None, normalizer_factor: float = None,
                 use_stationary_scaling: bool = False):
        super().__init__(env)
        self.signal = signal
        self.normalizer = normalizer
        self.beta = beta if beta is not None else config.get('reward.beta', 0.5)
        self.epsilon = epsilon if epsilon is not None else config.get('reward.alert_epsilon',
                                                                       config.get('reward.epsilon', 1e-8))
        self.normalizer_factor = normalizer_factor if normalizer_factor is not None else config.get('reward.normalizer_factor', 10)
        self.use_stationary_scaling = use_stationary_scaling

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get('done', True):
            raw_alert = self.signal.compute(info, expected_alerts, self.epsilon)
            info['alert_reward'] = raw_alert
            info['norm_alert_reward'] = self.normalizer.normalize(raw_alert, self.normalizer_factor)
            scale = 1.0 if self.use_stationary_scaling else self.unwrapped.total_steps
            reward += scale * self.beta * info['norm_alert_reward']
            logger.info(f"Alert reward: {raw_alert:.3f}, norm_alert_reward: {info['norm_alert_reward']:.3f}")
        return obs, reward, terminated, truncated, info


def create_alert_reward_wrapper(env, method_name: str, beta: float = None,
                                epsilon: float = None, normalizer_factor: float = None,
                                use_stationary_scaling: bool = False) -> AlertRewardWrapper:
    """Factory: build an AlertRewardWrapper from a registry name.

    Args:
        method_name: Key in ``ALERT_REWARD_REGISTRY`` (e.g. ``'AlertRewardWrapper2'``).
    """
    if method_name not in ALERT_REWARD_REGISTRY:
        raise ValueError(f"Unknown alert reward method: {method_name!r}. "
                         f"Available: {list(ALERT_REWARD_REGISTRY.keys())}")

    signal_cls, normalizer_cls = ALERT_REWARD_REGISTRY[method_name]

    # RelativeDiffAlertSignal needs fallback_baseline_alerts
    if signal_cls is RelativeDiffAlertSignal:
        fallback = {rule: expected_alerts[rule]['alert'] for rule in expected_alerts}
        signal = signal_cls(fallback)
    else:
        signal = signal_cls()

    normalizer = normalizer_cls()

    return AlertRewardWrapper(
        env, signal=signal, normalizer=normalizer,
        beta=beta, epsilon=epsilon, normalizer_factor=normalizer_factor,
        use_stationary_scaling=use_stationary_scaling,
    )


# Backward-compatible alias (maps names → lambdas that call the factory)
ENUM_ALERT_REWARD_METHODS = {
    name: lambda env, beta=None, epsilon=None, normalizer_factor=None, _n=name: (
        create_alert_reward_wrapper(env, _n, beta=beta, epsilon=epsilon,
                                    normalizer_factor=normalizer_factor)
    )
    for name in ['AlertRewardWrapper', 'AlertRewardWrapper1',
                 'AlertRewardWrapper2', 'AlertRewardWrapper3']
}         

class EnergyRewardWrapper(RewardWrapper):
    """Wrapper for energy consumption rewards"""
    def __init__(self, env: gym.Env, alpha: float = None, is_mock: bool = False,
                 use_stationary_scaling: bool = False, energy_normalizer=None):
        super().__init__(env)
        self.alpha = alpha if alpha is not None else config.get('reward.alpha', 0.5)
        self.is_mock = is_mock
        self.use_stationary_scaling = use_stationary_scaling
        self.energy_normalizer = energy_normalizer or self._build_normalizer()
        self.energies = []
        self.epsilon = config.get('reward.energy_epsilon', config.get('reward.epsilon', 1e-8))
        self.ENERGY_CHANGE_TARGET = config.get('reward.energy_change_target', 1)

    def _build_normalizer(self):
        """Build energy normalizer from config."""
        name = config.get('reward.energy_normalizer', 'tanh')
        cls = ENERGY_NORMALIZER_REGISTRY.get(name, TanhEnergyNormalizer)
        if name == 'tanh':
            return cls(scale=config.get('reward.tanh_energy_scale', 1.5))
        elif name == 'log1p':
            return cls(
                scale=config.get('reward.energy_normalizer_scale', 1.0),
                ceiling=config.get('reward.energy_normalizer_ceiling', 10.0),
            )
        elif name == 'clip_squared':
            return cls(ceiling=config.get('reward.energy_normalizer_ceiling', 2.0))
        elif name == 'power':
            return cls(
                threshold=config.get('reward.energy_power_threshold', 1.0),
                exponent=config.get('reward.power_energy_exponent', 2.0),
            )
        return cls()
    
    def estimate_energy_consumption(self, fake_dist, info):
        """Estimate energy consumption using per-rule regression models."""
        rules_alerts = {rule: info['raw_metrics'][rule]['alert'] for rule in self.unwrapped.splunk_tools.active_saved_searches}

        if not isinstance(fake_dist, np.ndarray):
            fake_dist = np.array(list(fake_dist)[:-1])
        rules_alerts_array = np.array(list(rules_alerts.values()))
        if fake_dist.ndim == 1:
            fake_dist = fake_dist.reshape(1, -1)
        if rules_alerts_array.ndim == 1:
            rules_alerts_array = rules_alerts_array.reshape(1, -1)
        fake_dist_normalizer = config.get('reward.fake_dist_normalizer', 475796)
        alert_normalizer = config.get('reward.alert_normalizer', 203)
        fake_dist = fake_dist / fake_dist_normalizer
        rules_alerts_array = (rules_alerts_array - 1) / alert_normalizer

        estimated_energy = 0
        for i, rule in enumerate(expected_alerts):
            rule_model = self.env.energy_models[rule]
            rule_alert = rules_alerts_array[:, i].reshape(1, -1)
            rule_X = np.concatenate((fake_dist, rule_alert), axis=1)
            rule_cpu = rule_model.predict(rule_X)[0]
            info['raw_metrics'][rule]['cpu'] = rule_cpu
            estimated_energy += rule_cpu
        return estimated_energy

    
    def step(self, action):
        """Override step to properly handle info updates"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get('done', True):
            if self.is_mock:
                estimated_energy = self.estimate_energy_consumption(
                    self.unwrapped.ac_fake_distribution.values(),
                    info
                )

                info['combined_metrics']['cpu'] = estimated_energy
            current = info['combined_metrics']['cpu']
            baseline = info['combined_baseline_metrics']['cpu']
            energy_reward = max((current  - baseline) / (baseline + self.epsilon), 0)
            self.energies.append(energy_reward)
            info['energy_reward'] = energy_reward
            info['norm_energy_reward'] = self.energy_normalizer.normalize(energy_reward)
            logger.info(f"Energy reward: {energy_reward:.3f}, current: {current:.3f}, baseline: {baseline:.3f}")
            scale = 1.0 if self.use_stationary_scaling else self.unwrapped.total_steps
            reward += scale * self.alpha * info['norm_energy_reward']
        return obs, reward, terminated, truncated, info

class DistributionRewardWrapper(RewardWrapper):
    """Wrapper for distribution similarity rewards.

    Unlike energy and alert rewards (applied once at episode end when
    ``info['done']`` is True), distribution reward is applied on every step.
    This is intentional: continuous per-step feedback steers the agent toward
    realistic log distributions throughout the episode, rather than only
    penalizing the final divergence.
    """
    def __init__(self, env: gym.Env, gamma: float = None, epsilon: float = None,
                 distribution_freq: int = None, distribution_threshold: float = None,
                 normalize_to_episode: bool = None):
        super().__init__(env)
        self.gamma = gamma if gamma is not None else config.get('reward.distribution_gamma', 0.2)
        self.epsilon = epsilon if epsilon is not None else config.get('reward.distribution_epsilon',
                                                                       config.get('reward.epsilon', 1e-8))
        self.distribution_reward_freq = distribution_freq if distribution_freq is not None else config.get('reward.distribution_freq', 3)
        self.distribution_threshold = distribution_threshold if distribution_threshold is not None else config.get('reward.distribution_threshold', 0.22)
        self.normalize_to_episode = normalize_to_episode if normalize_to_episode is not None \
            else config.get('reward.normalize_distribution_to_episode', True)
        
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        dist_value = self._calculate_distribution_value(
            self.unwrapped.ac_real_state,
            self.unwrapped.ac_fake_state
        )
        info['ac_distribution_value'] = dist_value
        info['full_ac_distribution_value'] = dist_value
        info['ac_distribution_reward'] = self._calculate_distribution_reward(dist_value)
        dist_reward = self.gamma * info['ac_distribution_reward']
        if self.normalize_to_episode:
            dist_reward /= self.unwrapped.total_steps
        reward += dist_reward
        return obs, reward, terminated, truncated, info
    
    def _calculate_distribution_reward(self, distribution_value: float) -> float:
        kl_scale = config.get('reward.kl_divergence_scale', 4)
        return -np.tanh(distribution_value * kl_scale)

    def _calculate_distribution_value(self, real_dist, fake_dist):
        # Add epsilon and normalize
        real_dist = (real_dist + self.epsilon) / np.sum(real_dist + self.epsilon)
        fake_dist = (fake_dist + self.epsilon) / np.sum(fake_dist + self.epsilon)
        
        return self._kl_divergence(real_dist, fake_dist)
        
    def _kl_divergence(self, p, q):
        return np.sum(p * np.log(np.clip(p / q, 1e-10, 1e10)))
    
    def chi_square(self, p, q):
        return np.sum((p - q) ** 2 / (p + q + self.epsilon))

class DistributionRewardWrapper1(DistributionRewardWrapper):
    """Wrapper for distribution similarity rewards"""
    def __init__(self, env: gym.Env, gamma: float = None, epsilon: float = None,
                 distribution_freq: int = None, distribution_threshold: float = None,
                 normalize_to_episode: bool = None):
        super().__init__(env, gamma, epsilon, distribution_freq, distribution_threshold,
                         normalize_to_episode)

    def _calculate_distribution_reward(self, distribution_value: float) -> float:
        return -distribution_value


class DistributionRewardWrapper2(DistributionRewardWrapper):
    """Convex distribution penalty: ``-(KL / T)^p``.

    Reuses ``distribution_threshold`` (default 0.22) as T.
    Config keys: ``reward.distribution_power_threshold``, ``reward.power_distribution_exponent``.
    """

    def _calculate_distribution_reward(self, distribution_value: float) -> float:
        T = config.get('reward.distribution_power_threshold',
                       config.get('reward.distribution_threshold', 0.22))
        p = config.get('reward.power_distribution_exponent', 2.0)
        T = T if T != 0 else 1e-8
        return -(distribution_value / T) ** p


ENUM_DISTRIBUTION_REWARD_METHODS = {
    'DistributionRewardWrapper': DistributionRewardWrapper,
    'DistributionRewardWrapper1': DistributionRewardWrapper1,
    'DistributionRewardWrapper2': DistributionRewardWrapper2,
    }