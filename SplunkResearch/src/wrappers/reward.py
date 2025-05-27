from ast import Tuple
import asyncio
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
import sys
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src')

from env_utils import *
import logging
logger = logging.getLogger(__name__)
from time_manager import TimeWindow

class AlertPredictor:
    """Separate class to handle alert prediction logic"""
    
    def __init__(self, expected_alerts: Dict, epsilon: float = 1):
        self.expected_alerts = expected_alerts
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
        expected = self.expected_alerts.get(rule_name, 0)
        gap = max(0, predicted_current - expected)
        reward = -(gap) / (expected + self.epsilon)
        
        return reward
    
    def predict_overall_alert_reward(self, baseline_metrics: Dict, diversity_logs: Dict,
                                   section_logtypes: Dict, is_mock: bool = False) -> float:
        """Predict overall alert reward for all rules"""
        rewards = []
        
        for rule_name, expected in self.expected_alerts.items():
            baseline_alert = baseline_metrics.get(rule_name, {}).get('alert', 0)
            reward = self.predict_alert_reward(
                rule_name, baseline_alert, diversity_logs, 
                section_logtypes, is_mock
            )
            rewards.append(reward)
            
        return np.mean(rewards) if rewards else 0
        # return -(sum(self.current_alerts.values()) - sum(self.expected_alerts.values())/ (sum(self.expected_alerts.values()) + self.epsilon))


class BaseRuleExecutionWrapperWithPrediction(RewardWrapper):
    """Enhanced base wrapper with alert prediction capability"""
    
    def __init__(self, env, baseline_dir: str = "baselines", is_mock: bool = False,
                 enable_prediction: bool = True, alert_threshold: float = -0.5,
                 skip_on_low_alert: bool = True, use_energy: bool = True):
        super().__init__(env)
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.is_mock = is_mock
        
        # Prediction configuration
        self.enable_prediction = enable_prediction
        self.alert_threshold = alert_threshold
        self.skip_on_low_alert = skip_on_low_alert
        
        # Load or create baseline table
        self.baseline_path = self._get_baseline_path()
        self.baseline_df = self._load_baseline_table()
        self.use_energy = use_energy
        self.energy_models = {}
        # if is_mock:
        #     for rule in self.splunk_tools.active_saved_searches:
        #         self.energy_models[rule] = pickle.load(open(f"/home/shouei/GreenSecurity-FirstExperiment/baseline_splunk_train-v32_2880_cpu_regressor_results/RandomForestRegressor_{rule}_with alert = 0.pkl", "rb"))
        
        # Initialize alert predictor
        self.expected_alerts = {
            'ESCU Windows Rapid Authentication On Multiple Hosts Rule': 0,
            'Windows AD Replication Request Initiated from Unsanctioned Location': 0,
            'Windows Event For Service Disabled': 2.5,
            'Detect New Local Admin account': 0.7,
            'ESCU Network Share Discovery Via Dir Command Rule': 0,
            'Known Services Killed by Ransomware': 6,
            'Non Chrome Process Accessing Chrome Default Dir': 0,
            'Kerberoasting spn request with RC4 encryption': 0,
            'Clop Ransomware Known Service Name': 0
        }
        
        self.alert_predictor = AlertPredictor(self.expected_alerts)
        self.execution_decisions = []
        
    def _get_baseline_path(self) -> Path:
        """Get path for baseline data based on environment config"""
        env_id = self.env.config.env_id
        search_window = self.env.config.search_window
        return self.baseline_dir / f"baseline_{env_id}_{search_window}.csv"
        
    def _load_baseline_table(self) -> pd.DataFrame:
        """Load existing baseline table or create new one"""
        if self.baseline_path.exists():
            return pd.read_csv(self.baseline_path)
        return pd.DataFrame(columns=['start_time', 'end_time', 'alert_values', 'duration_values'])

    def get_baseline_data(self, time_range: TimeWindow) -> Dict:
        """Get baseline data - execute after cleaning if needed"""
        num_of_measurements = self.env.config.baseline_num_of_measurements
        relevant_rows = self.baseline_df[(self.baseline_df['start_time'] == time_range[0]) & (self.baseline_df['end_time'] == time_range[1])]
        actual_num_of_measurements = relevant_rows.groupby(['start_time', 'end_time', 'search_name']).size().values[0] if not relevant_rows.empty else 0
        needed_measurements = num_of_measurements - actual_num_of_measurements
        
        # check if some search is missing
        running_dict = {}
        existing_searches = set()
        if len(relevant_rows) != 0:
            existing_searches = set(relevant_rows.reset_index()['search_name'].values)
            running_dict.update({search: needed_measurements for search in existing_searches})
        missing_searches = set([search for search in self.splunk_tools.active_saved_searches]) - existing_searches
        running_dict.update({search: num_of_measurements for search in missing_searches})
        
        empty_monitored_files(SYSTEM_MONITOR_FILE_PATH)
        empty_monitored_files(SECURITY_MONITOR_FILE_PATH)
        
        if sum(running_dict.values()) > 0 and not self.is_mock:
            logger.info('Cleaning the environment')
            clean_env(self.splunk_tools, time_range)
            logger.info('Measure no agent reward values')
            
        if sum(running_dict.values()) > 0:
            logger.info(f"Running {running_dict}")
            # Execute rules and get metrics
            if self.is_mock:
                rules_metrics = self.mock_rules_metrics(time_range)
            else:
                rules_metrics, total_cpu = asyncio.run(self.splunk_tools.run_saved_searches(time_range, running_dict))
            new_lines = self.convert_metrics(time_range, rules_metrics)
            if len(new_lines) != 0:
                self.baseline_df = pd.concat([self.baseline_df, pd.DataFrame(new_lines)])
                
        random_val = np.random.randint(0, 10)
        if random_val % 3 == 0 and not self.is_mock:
            self.baseline_df.to_csv(self.baseline_path, index=False)
            
        relevant_rows = self.baseline_df[(self.baseline_df['start_time'] == time_range[0]) & (self.baseline_df['end_time'] == time_range[1])]
        return relevant_rows
    
    def predict_and_decide_execution(self, time_range: TimeWindow, diversity_logs: Dict, distribution_value: float) -> Tuple[bool, float, Dict]:
        """Predict alert reward and decide whether to execute rules"""
        # First get baseline data (this might trigger cleaning and baseline execution)
        baseline_data = self.get_baseline_data(time_range)
        
        if baseline_data.empty:
            # No baseline data, must execute
            return True, None, {}
            
        # Process baseline metrics
        grouped_baseline = baseline_data.groupby('search_name')
        raw_baseline_metrics, _ = self.process_metrics(grouped_baseline)
        
        # Predict alert reward
        predicted_reward = self.alert_predictor.predict_overall_alert_reward(
            raw_baseline_metrics, 
            diversity_logs,
            self.section_logtypes,
            self.is_mock
        )
        
        # Decide whether to execute
        should_execute = True
        if self.enable_prediction and self.skip_on_low_alert:
            should_execute = (predicted_reward >= self.alert_threshold) and (distribution_value < self.env.distribution_threshold) and self.use_energy
            
        # Log decision
        self.execution_decisions.append({
            'time_range': time_range,
            'predicted_reward': predicted_reward,
            'should_execute': should_execute,
            'threshold': self.alert_threshold
        })
        
        logger.info(f"Alert prediction: reward={predicted_reward:.3f}, "
                   f"threshold={self.alert_threshold}, execute={should_execute}")
        
        return should_execute, predicted_reward, raw_baseline_metrics
    
    def mock_rules_metrics(self, time_range: TimeWindow) -> Dict:
        """Mock rules metrics for the given time range"""
        rules_metrics = self.splunk_tools.mock_run_saved_searches(time_range)
        return rules_metrics
    
    def get_current_reward_values(self, time_range: TimeWindow) -> Tuple[pd.DataFrame, Dict]:
        """Get current reward values"""
        if self.unwrapped.is_mock:
            rules_metrics = self.mock_rules_metrics(time_range)
        else:
            rules_metrics, total_cpu = asyncio.run(self.splunk_tools.run_saved_searches(
                time_range, None, self.env.config.num_of_measurements))
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
        logger.info(f"rules_metrics: {rules_metrics}")
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
            inserted_logs = info.get('inserted_logs', 0)
            diversity_logs = info.get('diversity_episode_logs', {})
            
            # Predict and decide whether to execute
            should_execute, predicted_reward, raw_baseline_metrics = self.predict_and_decide_execution(
                info['current_window'], 
                diversity_logs,
                info.get('ac_distribution_value', 0)
            )
            
            # Store prediction info
            info['predicted_alert_reward'] = predicted_reward
            info['execution_skipped'] = not should_execute
            
            self.unwrapped.is_mock = not should_execute
            # inject logs if not is_mock
            if not self.unwrapped.is_mock:
                self.env.inject_episodic_logs()
                # wait for the logs to be injected
                sleep(5)
            # Normal execution flow
            raw_metrics, combined_metrics = self.get_current_reward_values(info['current_window'])
            _, combined_baseline_metrics = self.process_metrics(
                self.get_baseline_data(info['current_window']).groupby('search_name')
            )
            info['combined_metrics'] = combined_metrics
            info['combined_baseline_metrics'] = combined_baseline_metrics
            if self.unwrapped.is_mock:
                current_alerts = self.alert_predictor.current_alerts
                baseline_alerts = {rule: raw_baseline_metrics[rule]['alert'] for rule in self.expected_alerts}
                for rule in self.expected_alerts:
                    raw_metrics[rule]['alert'] = current_alerts[rule]
                info['combined_metrics']['alert'] = sum(current_alerts.values()) + sum(baseline_alerts.values())
            # Set a penalty reward for skipping
            info['alert_reward'] = predicted_reward
            reward = 1
            if info.get('ac_distribution_value', 0) > self.env.distribution_threshold:
                    reward = -np.exp(info['ac_distribution_value'] ** 2)
                    if predicted_reward < self.alert_threshold:
                        reward += predicted_reward
            else:
                if predicted_reward < self.alert_threshold:
                    reward = predicted_reward
                else:
                    reward = 1

            
            # Store in info for other wrappers to use
            info['raw_metrics'] = raw_metrics
            info['raw_baseline_metrics'] = raw_baseline_metrics
            
        return obs, reward, terminated, truncated, info
    

    
class EnergyRewardWrapper(RewardWrapper):
    """Wrapper for energy consumption rewards"""
    def __init__(self, env: gym.Env, alpha: float = 0.5,is_mock: bool = False):
        super().__init__(env)
        self.alpha = alpha
        # self.is_mock = is_mock
    
    def estimate_energy_consumption(self):
        # Placeholder for energy consumption estimation logic
        # This should return the estimated energy consumption for the current state
        cpu_dict = {}
        for rule in self.splunk_tools.active_saved_searches:
            # Use the energy model to estimate energy consumption
            model = self.energy_models[rule]
            # Assuming the model has a predict method
            # Replace with actual prediction logic
            estimated_energy = model.predict(self.ac_fake_state.reshape(1, -1))[0]
            cpu_dict[rule] = estimated_energy
        return cpu_dict
    
    def step(self, action):
        """Override step to properly handle info updates"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        step = info.get('step', 0)
        if info.get('done', True) and info.get('distribution_reward') == 0:
            reward = 0

        if info.get('done', True):
            # if self.unwrapped.is_mock:                                           
            #     cpu_dict = self.estimate_energy_consumption()
            #     for rule in self.splunk_tools.active_saved_searches:
            #         info['raw_metrics'][rule]['cpu'] = cpu_dict[rule]
            #     info['combined_metrics']['cpu'] = sum(cpu_dict.values())
            current = info['combined_metrics']['cpu']

            baseline = info['combined_baseline_metrics']['cpu']

            
            energy_reward = (current  - baseline) / baseline
            # if energy_reward <= 0.1:
            #     energy_reward = 0
            energy_reward = np.clip(energy_reward, 0, 1) # Normalize to [0, 1]
            info['energy_reward'] = energy_reward
            # reward +=  energy_reward
            # reward += self.alpha*energy_reward
            if reward != 1:
                return obs, reward, terminated, truncated, info
            reward = energy_reward * 10
            # reward += self.unwrapped.total_steps*self.alpha*energy_reward
            # reward = energy_reward/(reward + self.epsilon)
            # reward += self.alpha * energy_reward
            
        return obs, reward, terminated, truncated, info

class AlertRewardWrapper(RewardWrapper):
    """Wrapper for alert rate rewards"""
    def __init__(self, env: gym.Env, beta: float = 0.3, epsilon: float = 1e-5, is_mock: bool = False):
        super().__init__(env)
        self.beta = beta
        self.epsilon = epsilon
        self.expected_alerts = {'ESCU Windows Rapid Authentication On Multiple Hosts Rule': 0.2,
                                'Windows AD Replication Request Initiated from Unsanctioned Location': 0,
                                'Windows Event For Service Disabled':4.3,
                                'Detect New Local Admin account':0.3,
                                'ESCU Network Share Discovery Via Dir Command Rule':0,
                                'Known Services Killed by Ransomware':7.3,
                                'Non Chrome Process Accessing Chrome Default Dir':0,
                                'Kerberoasting spn request with RC4 encryption':0,
                                'Clop Ransomware Known Service Name':0}
        self.is_mock = is_mock
    
    def get_wrapper(self, wrapper_class):
        """
        Utility function to retrieve a specific wrapper instance from the wrapper stack.
        """
        env = self.env
        while env:
            if isinstance(env, wrapper_class):
                return env
            env = getattr(env, 'env', None)
        raise ValueError(f"Wrapper {wrapper_class} not found in the wrapper stack.")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        step = info.get('step', 0)
        
        if info.get('done', True) and info.get('distribution_reward') == 0:
            reward = 0

        if info.get('done', True):
            diversity_episode_logs = info['diversity_episode_logs']
            
            # Calculate alert reward
            if not self.is_mock:
                current_alerts = {rule:info['raw_metrics'][rule]['alert'] for rule in self.expected_alerts}
                baseline_alerts = {rule:info['raw_baseline_metrics'][rule]['alert'] for rule in self.expected_alerts}
            else:
                baseline_alerts = {rule:info['raw_baseline_metrics'][rule]['alert'] for rule in self.expected_alerts}
                current_alerts = self._calculate_alert_by_diversity(baseline_alerts, diversity_episode_logs)
                # update info with the current alerts
                for rule in self.expected_alerts:
                    info['raw_metrics'][rule]['alert'] = current_alerts[rule]
                    info['raw_baseline_metrics'][rule]['alert'] = baseline_alerts[rule]
                info['combined_metrics']['alert'] = sum(current_alerts.values()) + sum(baseline_alerts.values())
            
            self._sanity_check(current_alerts, baseline_alerts, diversity_episode_logs)
            
            alert_reward = self._calculate_alert_reward(current_alerts)
            
            info['alert_reward'] = alert_reward
            # reward += alert_reward
            ac_distribution_value = info.get('ac_distribution_value', 0)
            # if ac_distribution_value > self.distribution_threshold:
            #     reward = -ac_distribution_value*10
            # else:
            # reward =  (reward + self.beta *alert_reward)
            # reward += self.beta * alert_reward
            # reward +=self.unwrapped.total_steps*self.beta * alert_reward

            # reward /= (alert_reward + self.epsilon)
            if alert_reward < -1:
                reward = alert_reward
            else:
                reard = 0
        return obs, reward, terminated, truncated, info
    
    def _calculate_alert_by_diversity(self, baseline_alerts:Dict, diversity_episodes_logs: Dict):
        mock_alerts = {}
        for rule, expected in self.expected_alerts.items():
            baseline = baseline_alerts.get(rule, 0)
            relevant_log = self.section_logtypes.get(rule, None)
            if relevant_log is None:
                continue
            relevant_log = "_".join(relevant_log[0])
            relevant_log = "_".join((relevant_log, "1"))
            diversity = diversity_episodes_logs.get(relevant_log, 0)
            action_wrapper = self.get_wrapper(gym.ActionWrapper)
            trigger_log_q = action_wrapper.episode_logs[relevant_log]
            current = diversity
            if rule in ['ESCU Windows Rapid Authentication On Multiple Hosts Rule']:
                current = 0
            mock_alerts[rule] = current + baseline

        return mock_alerts
            
            
    def _sanity_check(self, current_alerts: Dict, baseline_alerts:Dict, diversity_episodes_logs: Dict):
        for rule, expected in self.expected_alerts.items():
            current = current_alerts.get(rule, 0)
            baseline = baseline_alerts.get(rule, 0)
            relevant_log = self.section_logtypes.get(rule, None)
            relevant_log = "_".join(relevant_log[0])
            relevant_log = "_".join((relevant_log, "1"))
            diversity = diversity_episodes_logs.get(relevant_log, 0)
            gap = current - baseline
            action_wrapper = self.get_wrapper(gym.ActionWrapper)
            trigger_log_q = action_wrapper.episode_logs[relevant_log]
            if (gap - diversity != 0) and (rule not in ['ESCU Windows Rapid Authentication On Multiple Hosts Rule']):
                logger.error(f"Gap is less than diversity: {rule}, current: {current}, baseline: {baseline}, diversity: {diversity}, gap: {gap}, trigger_log_q: {trigger_log_q}")

                    
    def _calculate_alert_reward(self, current_alerts: Dict) -> float:
        rewards = []
        max_rewards = []
        for rule, expected in self.expected_alerts.items():
            current = current_alerts.get(rule, 0)
            gap = max(0, current - expected)
            reward = (gap) / (expected + self.epsilon)
            # max_reward = (self.env.diversity_factor+1) / (expected + self.epsilon)
            # max_reward = (self.env.diversity_factor+1+expected) / (expected + self.epsilon)
            rewards.append(reward)
            # max_rewards.append(max_reward)
            # self.unwrapped.rules_rel_diff_alerts[rule] = reward/max_reward
        if not rewards:
            return 0
            
        return -np.mean(rewards) #/ (np.mean(max_rewards) + self.epsilon)
        # return -np.mean(rewards) #/ ((self.env.diversity_factor+1)/self.epsilon)


class DistributionRewardWrapper(RewardWrapper):
    """Wrapper for distribution similarity rewards"""
    def __init__(self, env: gym.Env, gamma: float = 0.2, epsilon: float = 1e-8, distribution_freq: int = 3):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.distribution_reward_freq = distribution_freq
        self.distribution_threshold = 0.22
        
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)    
        # self.update_fake_distribution(self.episode_logs)

        step_counter = info.get('step', 0)
        # if  random.randint(0, 10) % self.distribution_reward_freq == 0:
        #     dist_value = self._calculate_distribution_value(
        #         self.unwrapped.real_state,
        #         self.unwrapped.fake_state
        #     )
        #     info['distribution_value'] = dist_value
        #     dist_reward = self._calculate_distribution_reward(dist_value)
        #     # dist_reward /= 0.6 # NOrmalize the reward
        #     info['distribution_reward'] = dist_reward
        #     # reward += dist_reward
        #     reward += dist_reward

                # reward += self.gamma * dist_reward
        if info.get('done', True):
            dist_value = self._calculate_distribution_value(
                self.unwrapped.ac_real_state,
                self.unwrapped.ac_fake_state
            )
            info['ac_distribution_value'] = dist_value
            dist_reward = self._calculate_distribution_reward(dist_value)
            # dist_reward /= 0.6 # NOrmalize the reward
            info['ac_distribution_reward'] = dist_reward
            # reward += self.gamma*dist_reward
            if reward < -1:
                return obs, reward, terminated, truncated, info
            if dist_value > self.distribution_threshold:
                reward = -dist_value*100
                reward = dist_reward
            else:
                reward = dist_reward

        # since this is the last wrapper, we can consider it as final reward
        logger.info(f"Reward: {reward}")  
        return obs, reward, terminated, truncated, info
    
    def _calculate_distribution_reward(self, distribution_value: float) -> float:
        # return 0.1 * np.log(distribution_value+self.epsilon) - distribution_value**2
        # d_target = 0.2
        # if distribution_value > 1.5*d_target:
        #     self.gamma *= 1.5
        # elif distribution_value < 0.5*d_target:
        #     self.gamma *= 0.5
        # return -self.gamma * distribution_value
        return -distribution_value
        # return -distribution_value/0.26
        # return -(distribution_value*100)*3
        # return np.log(1-distribution_value)
     
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


class ClipRewardWrapper(RewardWrapper):
    """Clip reward values to a given range"""
    def __init__(self, env: gym.Env, low: float = -1.0, high: float = 1.0):
        super().__init__(env)
        self.low = low
        self.high = high
    
    def reward(self, reward: float) -> float:
        return np.clip(reward, self.low, self.high)
   
# class QuotaViolationWrapper(RewardWrapper):
#     """Wrapper for quota violation penalties"""
#     def __init__(self, env: gym.Env, penalty: float = 1.0):
#         super().__init__(env)
#         self.penalty = penalty
        
#     def reward(self, reward: float) -> float:
#         info = self.get_step_info()
#         remaining_quota = info.get('remaining_quota', 0)
        
#         if remaining_quota >= 1.5:
#             violation_reward = -remaining_quota**2 * self.penalty
#             info['violation_reward'] = violation_reward
#             return violation_reward
            
#         return reward









# Example usage:
if __name__ == "__main__":
    # Create your base environment
    env = gym.make('splunk_train-v32')
    
    # Add reward wrappers as needed
    env = DistributionRewardWrapper(env, gamma=0.2)
    env = EnergyRewardWrapper(env, alpha=0.5)
    env = AlertRewardWrapper(env, beta=0.3)
    
    # Now the environment will automatically combine all rewards
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            break
        
        