from ast import Tuple
import asyncio
from curses import raw
from os import path
import pickle
import random
from time import sleep
import gymnasium as gym
from gymnasium.core import RewardWrapper
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import sys

from traitlets import Bool
sys.path.insert(1, '/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch/src')

from env_utils import *
import logging
logger = logging.getLogger(__name__)
from time_manager import TimeWindow
from config import config

# Load configuration values
std = config.get('reward.std', 13)

# Convert expected_alerts from config array to dictionary format
_expected_alert_values = config.get('reward.expected_alerts_per_rule', [4, 0.7, 0, 4, 0, 0, 0, 0, 0])
_rule_names = [
    'Windows Event For Service Disabled',
    'Detect New Local Admin Account',
    'ESCU Network Share Discovery Via Dir Command Rule',
    'Known Services Killed by Ransomware',
    'Non Chrome Process Accessing Chrome Default Dir',
    'Kerberoasting SPN Request With RC4 Encryption',
    'Clop Ransomware Known Service Name',
    'Windows AD Replication Request Initiated from Unsanctioned Location',
    'ESCU Windows Rapid Authentication On Multiple Hosts Rule'
]
expected_alerts = {rule: {'alert': alert} for rule, alert in zip(_rule_names, _expected_alert_values)}
class AlertPredictor:
    """Separate class to handle alert prediction logic"""
    
    def __init__(self, expected_alerts: Dict, epsilon: float = 1):
        expected_alerts = expected_alerts
        self.epsilon = epsilon
        self.current_alerts = {}
        
    def predict_alert_reward(self, rule_name: str, baseline_alert: float, 
                           diversity_logs: Dict, section_logtypes: Dict, 
                           is_mock: bool = False) -> float:
        """Predict alert reward for a specific rule"""
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
        expected = expected_alerts.get(rule_name, 0).get('alert', 0)
        gap = max(0, predicted_current - expected)
        reward = -(gap) / (expected + self.epsilon)
        
        return reward
    
    def predict_overall_alert_reward(self, baseline_metrics: Dict, diversity_logs: Dict,
                                   section_logtypes: Dict, is_mock: bool = False, normalized_distribution: Optional[np.ndarray] = np.array([0.0])) -> float:
        """Predict overall alert reward for all rules"""
        rewards = []
        for rule_name, expected in expected_alerts.items():
            baseline_alert = baseline_metrics.get(rule_name, {}).get('alert', 0)
            reward = self.predict_alert_reward(
                rule_name, baseline_alert, diversity_logs, 
                section_logtypes, is_mock
            )
            rewards.append(reward)
            
        # return np.mean(rewards) if rewards else 0
        # return sum(self.current_alerts.values())
        baseline_alerts = {rule: expected_alerts.get(rule, {}).get('alert', 0) for rule in expected_alerts}
        
        normal_alert_rate = sum(baseline_alerts.values())
        # normal_alert_rate = sum(baseline_alerts.values())
        return min(-(sum(self.current_alerts.values()) - normal_alert_rate  + self.epsilon)/ (normal_alert_rate + self.epsilon), 0)
        # return min(-(sum(self.current_alerts.values()) - 7 )/ (7 + self.epsilon), 0)
        # return min(-(sum(self.current_alerts.values()) - sum(expected_alerts.values()))/ (sum(expected_alerts.values()) + self.epsilon), 0)


class BaseRuleExecutionWrapperWithPrediction(RewardWrapper):
    """Enhanced base wrapper with alert prediction capability"""

    def __init__(self, env, is_mock: bool = False,
                 enable_prediction: bool = True, alert_threshold: float = None,
                 skip_on_low_alert: bool = True, use_energy: bool = True,use_alert: bool = True, is_eval: bool = False, is_train: bool = False, beta: float = None, gamma: float = None):
        super().__init__(env)

        # Load default values from config if not provided
        if alert_threshold is None:
            alert_threshold = config.get('reward.alert_threshold', -0.5)
        if beta is None:
            beta = config.get('reward.beta', 0.33)
        if gamma is None:
            gamma = config.get('reward.gamma', 0.33)

        self.is_mock = is_mock
        
        # Prediction configuration
        self.enable_prediction = enable_prediction
        self.alert_threshold = alert_threshold
        self.skip_on_low_alert = skip_on_low_alert
        

        self.use_energy = use_energy
        self.use_alert = use_alert
        self.energy_models = {}
        self.beta = beta
        self.gamma = gamma
        self.injection_id = 0
        self.baseline_measured = False
        # if is_mock:
        #     for rule in self.unwrapped.splunk_tools.active_saved_searches:
        #         self.energy_models[rule] = pickle.load(open(f"/home/shouei/GreenSecurity-FirstExperiment/baseline_splunk_train-v32_2880_cpu_regressor_results/RandomForestRegressor_{rule}_with alert = 0.pkl", "rb"))
        
        # Initialize alert predictor

        self.measuring = False
        self.alert_predictor = AlertPredictor(expected_alerts)
        self.execution_decisions = []
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
        self.epsilon = config.get('reward.epsilon', 0.00000001)
        self._baseline_cache = {}  # {(start_time, end_time): DataFrame}


  
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
    
    def predict_and_decide_execution(self, time_range: TimeWindow, diversity_logs: Dict, distribution_value: float) -> Tuple[bool, float, Dict, pd.DataFrame]:
        """Predict alert reward and decide whether to execute rules.

        Returns:
            (should_execute, predicted_reward, raw_baseline_metrics, baseline_data)
            The caller can reuse *baseline_data* instead of querying again.
        """
        # First get baseline data (this might trigger cleaning and baseline execution)
        baseline_data = self.get_baseline_data(time_range)

        if baseline_data.empty:
            # No baseline data, must execute
            return True, None, {}, baseline_data

        # Process baseline metrics
        grouped_baseline = baseline_data.groupby('search_name')
        raw_baseline_metrics, _ = self.process_metrics(grouped_baseline)

        # Predict alert reward
        predicted_reward = self.alert_predictor.predict_overall_alert_reward(
            raw_baseline_metrics,
            diversity_logs,
            self.unwrapped.section_logtypes,
            self.is_mock,
        )

        # Decide whether to execute
        should_execute = True
        if self.enable_prediction and self.skip_on_low_alert:
            # should_execute =  ((predicted_reward <= (sum(expected_alerts.values()) + (self.alert_threshold*std))) and (distribution_value < self.env.distribution_threshold) and self.use_energy) or self.is_eval
            should_execute = ((predicted_reward >= self.alert_threshold) and (distribution_value < self.env.distribution_threshold) and self.use_energy) or self.is_eval or True # TRY!!!!!!!

            # self.unwrapped.should_delete = should_execute and not self.is_mock and not self.measuring

        # Log decision
        self.execution_decisions.append({
            'time_range': time_range,
            'predicted_reward': predicted_reward,
            'should_execute': should_execute,
            'threshold': self.alert_threshold
        })

        logger.info(f"Alert prediction: reward={predicted_reward:.3f}, " f"threshold={self.alert_threshold}, execute={should_execute}")

        return should_execute, predicted_reward, raw_baseline_metrics, baseline_data
    
    def mock_rules_metrics(self, time_range: TimeWindow) -> Dict:
        """Mock rules metrics for the given time range"""
        rules_metrics = self.unwrapped.splunk_tools.mock_run_saved_searches(time_range)
        return rules_metrics
    
    def get_current_reward_values(self, time_range: TimeWindow, should_execute: Bool) -> Tuple[pd.DataFrame, Dict]:
        """Get current reward values"""
        if (self.is_mock and not self.measuring) or not should_execute or not self.use_energy or not self.use_alert:
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
        # logger.info(f"rules_metrics: {rules_metrics}")
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
        # print(grouped)
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
                # choose the measurement with the highest alert value
                max_alert = group['alert'].max()
                max_alert_index = group['alert'].idxmax()
                raw_metrics[search_name]['alert'] = max_alert
                raw_metrics[search_name]['duration'] = group['duration'].loc[max_alert_index]
                raw_metrics[search_name]['cpu'] = group['cpu'].loc[max_alert_index]
                raw_metrics[search_name]['read_count'] = group['read_count'].loc[max_alert_index]
                raw_metrics[search_name]['write_count'] = group['write_count'].loc[max_alert_index]
                raw_metrics[search_name]['read_bytes'] = group['read_bytes'].loc[max_alert_index]
                raw_metrics[search_name]['write_bytes'] = group['write_bytes'].loc[max_alert_index]

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
      
    def step(self, action):
        """Override step to properly handle info updates with prediction"""
        obs, reward, terminated, truncated, info = super().step(action)
        if info.get('done', True) and info.get('distribution_reward') == 0:
            reward = 0

        if info.get('done', True):
            self.baseline_measured = False
            inserted_logs = info.get('inserted_logs', 0)
            diversity_logs = info.get('diversity_episode_logs', {})
            
            # Predict and decide whether to execute
            should_execute, predicted_alert_reward, raw_baseline_metrics, cached_baseline_data = self.predict_and_decide_execution(
                info['current_window'],
                diversity_logs,
                info.get('ac_distribution_value', 0)
            )
            # dump baseline metrics to all data (once)
            if len(self.unwrapped.all_baseline_data) < len(self.unwrapped.baseline_df.time_range.unique()):
                self.unwrapped.all_baseline_data.append({
                    'time_range': info['current_window'],
                    'ac_real_distribution': self.unwrapped.ac_real_distribution,
                    'raw_metrics':raw_baseline_metrics,
                })
            else:
                # dump and empty the all_data to csv in path if exists
                all_baseline_data_df = pd.json_normalize(self.unwrapped.all_baseline_data, sep='_')
                all_baseline_data_df.to_csv(self.unwrapped.all_baseline_data_path, index=False, mode='w')          
                self.unwrapped.all_baseline_data = []
            
            # Store prediction info
            info['predicted_alert_reward'] = predicted_alert_reward
            # info['execution_skipped'] = not should_execute and False # TRY!!!!!!!
            self._episode_counter += 1
            if  ((self._episode_counter % self._measurement_frequency == 0 or self.is_eval) and should_execute and not self.is_mock):
                logger.info(f"Measuring")
                self.measuring = True
            else:
                self.measuring = False
                
            self.unwrapped.should_delete = should_execute and (not self.is_mock or self.measuring)
                
            # self.unwrapped.is_mock = (not should_execute or not self.use_energy or not self.use_alert) 
            time_window = info['current_window']
            time_range_start_epoch = time_window.start_epoch
            time_range_end_epoch = time_window.end_epoch
            # inject logs if not is_mock
            if (not self.is_mock  or self.measuring) and should_execute and self.use_energy and self.use_alert:
                if self.baseline_measured:
                    logger.info(f"Empty deletion dict of log generator and relvant fake_splunk_state")
                    self.unwrapped.log_generator.logs_to_delete = {}
                    for t_r in self.unwrapped.log_generator.fake_splunk_state:
                        t_r_datetime = (datetime.datetime.strptime(t_r[0], '%m/%d/%Y:%H:%M:%S'), datetime.datetime.strptime(t_r[1], '%m/%d/%Y:%H:%M:%S'))
                        if t_r_datetime[0].timestamp() >= time_range_start_epoch and t_r_datetime[1].timestamp() <= time_range_end_epoch:
                            self.unwrapped.log_generator.fake_splunk_state[t_r] = {}
                self.env.env.inject_episodic_logs(self.injection_id) # access to action wrapper0
                self.injection_id += 1
                # wait for the logs to be injected
                # sleep(4)
            should_run = True
            attempt = 0
            stop_loop = False
            # while should_run and not stop_loop:
            #     rerun = False
            #     if attempt > 4:
            #         logger.info(f"Re-running due to mismatch in alerts difference")
            rerun = False
                    
            #     attempt += 1
                # Normal execution flow
                
            raw_metrics, combined_metrics = self.get_current_reward_values(info['current_window'], should_execute)
            # logger.info(f"Raw metrics: {raw_metrics}")
            should_run = False
            baseline_raw_metrics, combined_baseline_metrics = self.process_metrics(
                cached_baseline_data.groupby('search_name')
            )
                # if rerun or self.is_mock:
                #     stop_loop = True

            # find the difference of alerts between raw_metrics and baseline_raw_metrics
            alerts_diff = {rule: raw_metrics[rule]['alert'] - baseline_raw_metrics.get(rule, {}).get('alert', 0) for rule in expected_alerts}
            
            if (not self.is_mock  or self.measuring) and should_execute and self.use_energy and self.use_alert:
                # check compatibility of alerts_diff with diversity info
                for rule in expected_alerts:
                    if rule == 'ESCU Windows Rapid Authentication On Multiple Hosts Rule':
                        continue
                    relevant_log = self.unwrapped.section_logtypes.get(rule, None)
                    if relevant_log:
                        log_type = "_".join(relevant_log[0]) + "_1"
                        diversity_value = diversity_logs.get(log_type, 0)
                        if alerts_diff[rule] != diversity_value:
                            
                            logger.warning(f"Alert difference mismatch for {rule}: alerts_diff={alerts_diff[rule]}, diversity_value={diversity_value}")
                            # get the field real_ts of the events in the results of the query to find the mismatch
                            def get_event_times(rule_name, time_range):
                                # query splunk for the events in the time range
                                default_host = config.get('splunk.default_host', 'dt-splunk')
                                secondary_host = config.get('hosts.secondary', '132.72.81.150')
                                query = f'index={self.unwrapped.splunk_tools.index_name} host IN ("{default_host}", {secondary_host}) EventCode={relevant_log[0][1]}  | stats count by real_ts var_id'
                                # print(time_range_date)
                                results = self.unwrapped.splunk_tools.run_search(query, *time_range)
                                formatted_log = "\n".join([json.dumps(record) for record in results])
                                logger.info(f"Event times for {rule_name} in {time_range}: {formatted_log}")
                            get_event_times(rule, (time_range_start_epoch, time_range_end_epoch))
                        else:
                            logger.info(f"Alert difference match for {rule}: alerts_diff={alerts_diff[rule]}, diversity_value={diversity_value}")
                        
            if self.is_mock and self.measuring and should_execute:  
                combined_metrics['real_cpu'] = combined_metrics['cpu'] 

            info['combined_metrics'] = combined_metrics
            info['combined_baseline_metrics'] = combined_baseline_metrics
            if self.is_mock and (self.use_alert or self.use_energy):
                current_alerts = self.alert_predictor.current_alerts
                # baseline_alerts = {rule: raw_baseline_metrics[rule]['alert'] for rule in expected_alerts}
                for rule in expected_alerts:
                    raw_metrics[rule]['alert'] = current_alerts[rule]
                    
                    # raw_baseline_metrics[rule]['cpu'] = 0
                    # raw_baseline_metrics[rule]['duration'] = 0
                info['combined_metrics']['alert'] = sum(current_alerts.values())# + sum(baseline_alerts.values())
                # sleep(2)

                
            if should_execute and not self.is_mock and self.use_energy and self.use_alert:
                self.unwrapped.all_data.append({
                    'time_range': info['current_window'],
                    'ac_fake_distribution': self.unwrapped.ac_fake_distribution,
                    'raw_metrics':raw_metrics,
                })
                if random.randint(0, 1000) % 10 == 0:
                    # dump and empty the all_data to csv in path if exists
                    if self.unwrapped.all_data:
                        all_data_df = pd.json_normalize(self.unwrapped.all_data, sep='_')
                        all_data_df.to_csv(self.unwrapped.all_data_path, index=False, mode='a', header=not path.exists(self.unwrapped.all_data_path))
                    
                    self.unwrapped.all_data = []
            # Set a penalty reward for skipping
            # info['alert_reward'] = predicted_alert_reward
            if not self.use_alert:
                predicted_alert_reward = 0

            info['raw_metrics'] = raw_metrics
            info['raw_baseline_metrics'] = raw_baseline_metrics
            
        return obs, reward, terminated, truncated, info




class AlertRewardWrapper(RewardWrapper):
    """Wrapper for alert-based rewards"""
    def __init__(self, env: gym.Env, beta: float = None, epsilon: float = None, normalizer_factor: float = None):
        super().__init__(env)
        self.beta = beta if beta is not None else config.get('reward.beta', 0.5)
        self.epsilon = epsilon if epsilon is not None else config.get('reward.epsilon', 1e-8)
        self.normalizer_factor = normalizer_factor if normalizer_factor is not None else config.get('reward.normalizer_factor', 10)

    def calculate_alert_reward(self, predicted_alert_reward: float) -> float:
        """Calculate normalized alert reward"""
        norm_alert_reward = -np.tanh(-predicted_alert_reward/self.normalizer_factor)
        return norm_alert_reward
    
    def step(self, action):
        """Override step to properly handle info updates"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get('done', True):
            predicted_alert_reward = info.get('predicted_alert_reward', 0)
            info['alert_reward'] = predicted_alert_reward
            info['norm_alert_reward'] = self.calculate_alert_reward(predicted_alert_reward)
            ############### Total reward ##############
            reward += self.unwrapped.total_steps*self.beta*info['norm_alert_reward']
            logger.info(f"Alert reward: {predicted_alert_reward:.3f}, norm_alert_reward: {info['norm_alert_reward']:.3f}")
        return obs, reward, terminated, truncated, info
    
class AlertRewardWrapper1(AlertRewardWrapper):
    """Wrapper for alert-based rewards"""
    def __init__(self, env: gym.Env, beta: float = None, epsilon: float = None, normalizer_factor: float = None):
        super().__init__(env, beta, epsilon, normalizer_factor)

    def calculate_alert_reward(self, predicted_alert_reward: float) -> float:
        """Calculate normalized alert reward"""
        norm_alert_reward = predicted_alert_reward
        return norm_alert_reward

class AlertRewardWrapper2(AlertRewardWrapper):
    """Wrapper for alert-based rewards"""
    def __init__(self, env: gym.Env, beta: float = None, epsilon: float = None, normalizer_factor: float = None):
        super().__init__(env, beta, epsilon, normalizer_factor)
        self.baseline_rules_alerts = {rule: expected_alerts[rule]['alert'] for rule in expected_alerts}

    def calculate_alert_reward(self, predicted_alert_reward: float) -> float:
        """Calculate normalized alert reward"""
        # norm_alert_reward = - self.normalizer_factor*((-predicted_alert_reward)**3)
        tanh_scale = config.get('reward.tanh_scale_factor', -2)
        norm_alert_reward = tanh_scale*np.tanh(-predicted_alert_reward/self.normalizer_factor)
        return norm_alert_reward
    
    def step(self, action):
        """Override step to properly handle info updates"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get('done', True):
            current_rules_alerts = {rule: info['raw_metrics'][rule]['alert'] for rule in expected_alerts}
            # print(f"Current rules alerts: {current_rules_alerts}")
            
            # print(f"Baseline rules alerts: {baseline_rules_alerts}")
            # average relative diff of alerts by rule
            alert_diffs = []
            for rule in expected_alerts:
                baseline_alert = self.baseline_rules_alerts.get(rule, 0)
                alert_diff = (current_rules_alerts[rule] - baseline_alert) / (baseline_alert + self.epsilon)
                alert_diffs.append(alert_diff)
            if alert_diffs:
                # print(f"Alert diffs: {alert_diffs}")
                reward_alert = min(0, -np.mean(alert_diffs))
                
            else:
                reward_alert = 0
            info['alert_reward'] = reward_alert
            info['norm_alert_reward'] = self.calculate_alert_reward(reward_alert)
            # reward += info['norm_alert_reward']
            reward += self.unwrapped.total_steps*self.beta*info['norm_alert_reward']
        return obs, reward, terminated, truncated, info

class AlertRewardWrapper3(AlertRewardWrapper2):
    def __init__(self, env: gym.Env, beta: float = None):
        super().__init__(env, beta=beta)
          
    def calculate_alert_reward(self, predicted_alert_reward: float) -> float:
        """Calculate normalized alert reward"""
        norm_alert_reward = predicted_alert_reward
        return norm_alert_reward  
    
# enum for different alert reward calculation methods     
ENUM_ALERT_REWARD_METHODS = {
    'AlertRewardWrapper': AlertRewardWrapper,
    'AlertRewardWrapper1': AlertRewardWrapper1,
    'AlertRewardWrapper2': AlertRewardWrapper2,
    'AlertRewardWrapper3': AlertRewardWrapper3,
    }         

class EnergyRewardWrapper(RewardWrapper):
    """Wrapper for energy consumption rewards"""
    def __init__(self, env: gym.Env, alpha: float = None, is_mock: bool = False):
        super().__init__(env)
        self.alpha = alpha if alpha is not None else config.get('reward.beta', 0.5)
        self.is_mock = is_mock
        self.energies = []
        self.epsilon = config.get('reward.epsilon', 1e-8)
        self.ENERGY_CHANGE_TARGET = config.get('reward.energy_change_target', 1)
    
    def estimate_energy_consumption(self, fake_dist, info):
        # Placeholder for energy consumption estimation logic
        # This should return the estimated energy consumption for the current state
        # Use the energy model to estimate energy consumption
        rules_alerts = {rule: info['raw_metrics'][rule]['alert'] for rule in self.unwrapped.splunk_tools.active_saved_searches} 
        
        model = self.env.energy_models["all"]
        # normalize the fake distribution and rule alert
        if not isinstance(fake_dist, np.ndarray):
            fake_dist = np.array(list(fake_dist)[:-1])
        rules_alerts_array = np.array(list(rules_alerts.values()))
        if fake_dist.ndim == 1:
            fake_dist = fake_dist.reshape(1, -1)
        if rules_alerts_array.ndim == 1:
            rules_alerts_array = rules_alerts_array.reshape(1, -1)
        # print(fake_dist, rules_alerts[rule])
        # print(fake_dist)
        # print(rules_alerts_array)
        fake_dist_normalizer = config.get('reward.fake_dist_normalizer', 475796)
        alert_normalizer = config.get('reward.alert_normalizer', 203)
        fake_dist = (fake_dist) / fake_dist_normalizer
        rules_alerts_array = (rules_alerts_array - 1) / alert_normalizer
        X = np.concatenate((fake_dist, rules_alerts_array), axis=1)
        # print(X)

        # print(X)
        # estimated_energy_all_model = model.predict(X)[0]
        
        # calculate estimated energy consumption using individual models and sum them
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
                # print(rules_alerts)
                # print(self.unwrapped.ac_fake_distribution)                                        
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

            # info['norm_energy_reward'] = np.clip(energy_reward, 0, 1)
            tanh_energy_scale = config.get('reward.tanh_energy_scale', 1.5)
            info['norm_energy_reward'] = np.tanh(energy_reward*tanh_energy_scale)
            # info['norm_energy_reward'] = energy_reward*100
            # info['norm_energy_reward'] = np.clip(energy_reward/self.ENERGY_CHANGE_TARGET, 0, 1)
            logger.info(f"Energy reward: {energy_reward:.3f}, current: {current:.3f}, baseline: {baseline:.3f}")
            reward += info['norm_energy_reward']
            reward += self.unwrapped.total_steps*self.alpha*info['norm_energy_reward']
        return obs, reward, terminated, truncated, info

class DistributionRewardWrapper(RewardWrapper):
    """Wrapper for distribution similarity rewards"""
    def __init__(self, env: gym.Env, gamma: float = None, epsilon: float = None, distribution_freq: int = None, distribution_threshold: float = None):
        super().__init__(env)
        self.gamma = gamma if gamma is not None else config.get('reward.distribution_gamma', 0.2)
        # print(f"Gamma: {self.gamma}")
        self.epsilon = epsilon if epsilon is not None else config.get('reward.epsilon', 1e-8)
        self.distribution_reward_freq = distribution_freq if distribution_freq is not None else config.get('reward.distribution_freq', 3)
        self.distribution_threshold = distribution_threshold if distribution_threshold is not None else config.get('reward.distribution_threshold', 0.22)
        
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)    
        # if terminated:
        dist_value = self._calculate_distribution_value(
            self.unwrapped.ac_real_state,
            self.unwrapped.ac_fake_state
        )
        info['ac_distribution_value'] = dist_value
        info['full_ac_distribution_value'] = dist_value
        info['ac_distribution_reward'] = self._calculate_distribution_reward(dist_value)
        reward = self.gamma*info['ac_distribution_reward']
        return obs, reward, terminated, truncated, info
    
    def _calculate_distribution_reward(self, distribution_value: float) -> float:
        # return -distribution_value*100
        kl_scale = config.get('reward.kl_divergence_scale', 4)
        return -np.tanh(distribution_value*kl_scale)

    def _calculate_distribution_value(self, real_dist, fake_dist):
        # Add epsilon and normalize
        real_dist = (real_dist + self.epsilon) / np.sum(real_dist + self.epsilon)
        fake_dist = (fake_dist + self.epsilon) / np.sum(fake_dist + self.epsilon)
        
        # Calculate JSD
        # m = (real_dist + fake_dist) / 2
        # jsd = (self._kl_divergence(real_dist, m) + 
        #        self._kl_divergence(fake_dist, m)) / 2
        # return jsd
        # Calculate KL divergence
        return self._kl_divergence(real_dist , fake_dist)
        # return self.chi_square(fake_dist, real_dist)
        
    def _kl_divergence(self, p, q):
        return np.sum(p * np.log(p / q))
    
    def chi_square(self, p, q):
        return np.sum((p - q) ** 2 / (p + q + self.epsilon))

class DistributionRewardWrapper1(DistributionRewardWrapper):
    """Wrapper for distribution similarity rewards"""
    def __init__(self, env: gym.Env, gamma: float = None, epsilon: float = None, distribution_freq: int = None, distribution_threshold: float = None):
        super().__init__(env, gamma, epsilon, distribution_freq, distribution_threshold)
        
    def calculate_distribution_reward(self, distribution_value: float) -> float:
        return -distribution_value
    
ENUM_DISTRIBUTION_REWARD_METHODS = {
    'DistributionRewardWrapper': DistributionRewardWrapper,
    'DistributionRewardWrapper1': DistributionRewardWrapper1,
    }