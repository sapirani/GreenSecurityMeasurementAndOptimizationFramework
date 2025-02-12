from ast import Tuple
import random
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

class DistributionRewardWrapper(RewardWrapper):
    """Wrapper for distribution similarity rewards"""
    def __init__(self, env: gym.Env, gamma: float = 0.2, epsilon: float = 1e-8):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        
    def reward(self, reward: float) -> float:
        """Modify reward based on distribution similarity"""
        info = self.get_step_info()  # Get current step info
        
        real_dist = info.get('real_distribution')
        fake_dist = info.get('fake_distribution')
        
        if real_dist is None or fake_dist is None:
            return reward
            
        dist_reward = self._calculate_distribution_reward(real_dist, fake_dist)
        info['distribution_reward'] = dist_reward
        
        return reward + self.gamma * dist_reward
        
    def _calculate_distribution_reward(self, real_dist, fake_dist):
        # Add epsilon and normalize
        real_dist = (real_dist + self.epsilon) / np.sum(real_dist + self.epsilon)
        fake_dist = (fake_dist + self.epsilon) / np.sum(fake_dist + self.epsilon)
        
        # Calculate JSD
        m = (real_dist + fake_dist) / 2
        jsd = (self._kl_divergence(real_dist, m) + 
               self._kl_divergence(fake_dist, m)) / 2
               
        return -jsd
        
    def _kl_divergence(self, p, q):
        return np.sum(p * np.log(p / q))
    
class BaseRuleExecutionWrapper(RewardWrapper):
    """Base wrapper that handles rule execution and baseline management"""
    
    def __init__(self, env, baseline_dir: str = "baselines"):
        super().__init__(env)
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create baseline table
        self.baseline_path = self._get_baseline_path()
        self.baseline_df = self._load_baseline_table()
        
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

    
    def run_saved_searches(self, time_range):
        alert_vals, duration_vals, std_duration_vals, saved_searches, mean_cpu_integrals, std_cpu_integrals, read_count, write_count, read_bytes, write_bytes, total_cpu_usage = self.splunk_tools.run_saved_searches_parallel(time_range)
        return {"alert":alert_vals, "duration":duration_vals, "std_duration":std_duration_vals, "saved_searches":saved_searches, "cpu":mean_cpu_integrals, "std_cpu":std_cpu_integrals, "read_count":read_count, "write_count":write_count, "read_bytes":read_bytes, "write_bytes":write_bytes, "total_cpu_usage":total_cpu_usage}
    
    def get_no_agent_reward(self, time_range: TimeWindow) -> Dict:
        relevant_row = self.baseline_df[(self.baseline_df['start_time'] == time_range[0]) & (self.baseline_df['end_time'] == time_range[1])]
        if not relevant_row.empty:
            combined_rules_metrics = self.rules_metrics_combiner(alert=relevant_row['alert'].values[0], duration=relevant_row['duration'].values[0], std_duration=relevant_row['std_duration'].values[0], cpu=relevant_row['cpu'].values[0], std_cpu=relevant_row['std_cpu'].values[0], read_count=relevant_row['read_count'].values[0], write_count=relevant_row['write_count'].values[0], read_bytes=relevant_row['read_bytes'].values[0], write_bytes=relevant_row['write_bytes'].values[0], total_cpu_usage=relevant_row['total_cpu_usage'].values[0])
        else:
            logger.info('Cleaning the environment')
            clean_env(self.splunk_tools, time_range)
            
            logger.info('Measure no agent reward values')
            new_line, combined_rules_metrics = self.get_rules_metrics(time_range)
            self.baseline_df = pd.concat([self.baseline_df, pd.DataFrame(
                new_line
            )])
            random_val = np.random.randint(0, 10)
            if random_val % 3 == 0:
                self.baseline_df.to_csv(self.baseline_path, index=False)
            relevant_row = self.baseline_df[(self.baseline_df['start_time'] == time_range[0]) & (self.baseline_df['end_time'] == time_range[1])]
            
        
        # periodly dunp no_agent table
        if random.random() < 0.3:
            self.baseline_df.to_csv(self.baseline_path, index=False)
        return combined_rules_metrics
    
    def get_duration_reward_values(self, time_range):
        new_line, combined_rules_metrics = self.get_rules_metrics(time_range)
        return combined_rules_metrics
    
    def rules_metrics_combiner(self, **rules_metrics):
        result = {}
        for rule_metric in rules_metrics:
            result[rule_metric] = np.sum(rules_metrics[rule_metric])
        return result
    
    def post_process_metrics(self, time_range, saved_searches, combined_rules_metrics, rules_metrics):
        logger.info(f"rules_metrics: {rules_metrics}")
        return {'start_time':[time_range[0]],
                'end_time':[time_range[1]],
                **{f"rule_{rule_metric}_{saved_search}": rules_metrics[rule_metric][i] 
                for rule_metric in rules_metrics  if rule_metric != "total_cpu_usage"
                for i, saved_search in enumerate(saved_searches)},
                **{rule_metric: combined_rules_metrics[rule_metric] for rule_metric in combined_rules_metrics}}
        
    def get_rules_metrics(self, time_range: TimeWindow):
        rules_metrics = self.run_saved_searches(time_range)
        saved_searches = rules_metrics['saved_searches']
        del rules_metrics['saved_searches']
        combined_rules_metrics = self.rules_metrics_combiner(**rules_metrics)
        new_line = self.post_process_metrics(time_range, saved_searches, combined_rules_metrics, rules_metrics)
        return new_line, combined_rules_metrics
    
    def reward(self, reward: float) -> float:
        info = self.env.get_step_info()
        
        if info.get('done', False):
            # Execute rules and get metrics
            current_metrics = self.get_duration_reward_values(info['current_window'])
            baseline_metrics = self.get_no_agent_reward(info['current_window'])
            
            # Store in info for other wrappers to use
            info['current_metrics'] = current_metrics
            info['baseline_metrics'] = baseline_metrics
            
        return reward
    
class EnergyRewardWrapper(RewardWrapper):
    """Wrapper for energy consumption rewards"""
    def __init__(self, env: gym.Env, alpha: float = 0.5):
        super().__init__(env)
        self.alpha = alpha
        
    def reward(self, reward: float) -> float:
        info = self.get_step_info()
        
        if not info.get('done', False):
            return reward
            
        current_cpu = info.get('current_cpu', 0)
        baseline_cpu = info.get('baseline_cpu', 0)
        
        if baseline_cpu > 0:
            energy_reward = max(0, (current_cpu - baseline_cpu) / baseline_cpu) / 2
            info['energy_reward'] = energy_reward
            return reward + self.alpha * energy_reward
            
        return reward

class AlertRewardWrapper(RewardWrapper):
    """Wrapper for alert rate rewards"""
    def __init__(self, env: gym.Env, beta: float = 0.3, epsilon: float = 1e-3):
        super().__init__(env)
        self.beta = beta
        self.epsilon = epsilon
        
    def reward(self, reward: float) -> float:
        info = self.get_step_info()
        
        if not info.get('done', False):
            return reward
            
        current_alerts = info.get('alert_counts', {})
        expected_alerts = info.get('expected_alerts', {})
        
        if not current_alerts or not expected_alerts:
            return reward
            
        alert_reward = self._calculate_alert_reward(current_alerts, expected_alerts)
        info['alert_reward'] = alert_reward
        
        return reward + self.beta * alert_reward
        
    def _calculate_alert_reward(self, current_alerts: Dict, expected_alerts: Dict) -> float:
        rewards = []
        for rule, expected in expected_alerts.items():
            current = current_alerts.get(rule, 0)
            gap = current - expected
            reward = (gap + self.epsilon) / (expected + self.epsilon)
            rewards.append(reward)
            
        if not rewards:
            return 0
            
        return -np.mean(rewards) / (5/self.epsilon)

class QuotaViolationWrapper(RewardWrapper):
    """Wrapper for quota violation penalties"""
    def __init__(self, env: gym.Env, penalty: float = 1.0):
        super().__init__(env)
        self.penalty = penalty
        
    def reward(self, reward: float) -> float:
        info = self.get_step_info()
        remaining_quota = info.get('remaining_quota', 0)
        
        if remaining_quota >= 1.5:
            violation_reward = -remaining_quota**2 * self.penalty
            info['violation_reward'] = violation_reward
            return violation_reward
            
        return reward

# Example usage:
if __name__ == "__main__":
    # Create your base environment
    env = gym.make('splunk_train-v32')
    
    # Add reward wrappers as needed
    env = DistributionRewardWrapper(env, gamma=0.2)
    env = EnergyRewardWrapper(env, alpha=0.5)
    env = AlertRewardWrapper(env, beta=0.3)
    env = QuotaViolationWrapper(env, penalty=1.0)
    
    # Now the environment will automatically combine all rewards
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            break