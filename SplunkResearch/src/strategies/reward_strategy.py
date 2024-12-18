from abc import ABC, abstractmethod
import datetime
import json
import os
import random
import subprocess
from time import sleep
import time
import numpy as np
from scipy import stats
from scipy.stats import entropy, wasserstein_distance
import pandas as pd
from scipy.spatial.distance import jensenshannon
import logging
import tensorflow as tf
from env_utils import *
from scipy.special import rel_entr

logger = logging.getLogger(__name__)
CPU_TDP = 200
PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.92.1.el7.x86_64/Splunk Enterprise SIEM/Power Saver Plan/One Scan/'



class RewardStrategy(ABC):
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        self.average_energy = 0
        self.average_alert = 0
        self.average_duration = 0
        self.epsilon = 0.000000000001
        logger.info(f"epsilon: {self.epsilon}")
        self.action_upper_bound = 1
        self.reward_dict = {'energy': [], 'alerts': [], 'distributions': [], 'duration': [], 'total': []}
        self.reward_values_dict = {'energy': [], 'alerts': [], 'distributions': [], 'duration': [], "num_of_rules":[], "p_values":[], "t_values":[], "degrees_of_freedom":[], "cpu":[]}
        self.dt_manager = dt_manager
        self.splunk_tools = splunk_tools
        self.rule_frequency = self.splunk_tools.rule_frequency
        self.time_rules_energy = []
        self.current_measurement_path = ''
        self.num_of_searches = num_of_searches  
        self.distribution_threshold = 0.5
        self.alert_threshold = 0
        self.measurment_tool = measurment_tool
        self.alpha = alpha
        self.betta = beta
        self.gamma = gamma
        self.no_agent_table_path = no_agent_table_path
        try: 
            self.no_agent_values = pd.read_csv(no_agent_table_path)
        except:
            self.no_agent_values = pd.DataFrame(columns=['start_time', 'end_time', 'alert_values', 'duration_values'])
            self.no_agent_values.to_csv(no_agent_table_path, index=False)
        self.no_agent_last_row = None
        self.no_agent_current_row = None
        self.current_distributions_distance = 0

    def get_no_agent_reward(self, time_range):
        relevant_row = self.no_agent_values[(self.no_agent_values['start_time'] == time_range[0]) & (self.no_agent_values['end_time'] == time_range[1])]
        if not relevant_row.empty:
            combined_rules_metrics = self.rules_metrics_combiner(alert=relevant_row['alert'].values[0], duration=relevant_row['duration'].values[0], std_duration=relevant_row['std_duration'].values[0], cpu=relevant_row['cpu'].values[0], std_cpu=relevant_row['std_cpu'].values[0], read_count=relevant_row['read_count'].values[0], write_count=relevant_row['write_count'].values[0], read_bytes=relevant_row['read_bytes'].values[0], write_bytes=relevant_row['write_bytes'].values[0], median_cpu_usage=relevant_row['median_cpu_usage'].values[0])
        else:
            logger.info('Measure no agent reward values')
            new_line, combined_rules_metrics = self.get_rules_metrics(time_range)
            self.no_agent_values = pd.concat([self.no_agent_values, pd.DataFrame(
                new_line
            )])
            random_val = np.random.randint(0, 10)
            if random_val % 3 == 0:
                self.no_agent_values.to_csv(self.no_agent_table_path, index=False)
            relevant_row = self.no_agent_values[(self.no_agent_values['start_time'] == time_range[0]) & (self.no_agent_values['end_time'] == time_range[1])]
            
        self.no_agent_current_row = relevant_row
        
        return combined_rules_metrics
        
        
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        after_metrics =  self.get_duration_reward_values(time_range)
        clean_env(self.splunk_tools, time_range)
        before_metrics =  self.get_no_agent_reward(time_range)
        self.no_agent_last_row = self.no_agent_current_row
        return after_metrics, before_metrics

    
    @abstractmethod
    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        pass

    
    def get_partial_reward_values(self, real_distribution, fake_distribution):
        distributions_val = self.compare_distributions(real_distribution, fake_distribution)     
        if distributions_val == 0:
            distributions_val = distributions_val + self.epsilon
        self.reward_values_dict['distributions'].append(distributions_val)
        logger.info(f"distributions value: {distributions_val}")
        self.current_distributions_distance = distributions_val
        return distributions_val
    
    def kl_divergence(self, p, q, epsilon=1e-10):
        """
        Calculate Kullback-Leibler divergence between two probability distributions.
        
        Args:
            p (array-like): First probability distribution
            q (array-like): Second probability distribution
            epsilon (float): Small constant to avoid log(0)
        
        Returns:
            float: KL divergence from q to p
        """
        # Convert to numpy arrays
        p = np.array(p)
        q = np.array(q)
        
        # Add small epsilon to avoid log(0)
        q = q + epsilon
        p = p + epsilon
        
        # Renormalize to ensure they sum to 1
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate KL divergence
        return np.sum(p * np.log(p / q))
    
    def compare_distributions(self, dist1, dist2):
        if random.random() > 0.5:
            logger.info(f"dist1: {dist1}")
            logger.info(f"dist2: {dist2}")
        # return wasserstein_distance(dist2, dist1)
        # return entropy(dist2, dist1)
        # return jensenshannon(dist2, dist1)#**2
        # return np.sum(rel_entr(dist2, dist1))
        return self.kl_divergence(dist2, dist1)
    
    def run_saved_searches(self, time_range):
        alert_vals, duration_vals, std_duration_vals, saved_searches, mean_cpu_integrals, std_cpu_integrals, read_count, write_count, read_bytes, write_bytes = self.splunk_tools.run_saved_searches(time_range)
        return {"alert":alert_vals, "duration":duration_vals, "std_duration":std_duration_vals, "saved_searches":saved_searches, "cpu":mean_cpu_integrals, "std_cpu":std_cpu_integrals, "read_count":read_count, "write_count":write_count, "read_bytes":read_bytes, "write_bytes":write_bytes}
    
    def post_process_metrics(self, time_range, saved_searches, combined_rules_metrics, rules_metrics):
        logger.info(f"rules_metrics: {rules_metrics}")
        return {'start_time':[time_range[0]],
                'end_time':[time_range[1]],
                **{f"rule_{rule_metric}_{saved_search}": rules_metrics[rule_metric][i] 
                for rule_metric in rules_metrics  if rule_metric != "total_cpu_usage"
                for i, saved_search in enumerate(saved_searches)},
                **{rule_metric: combined_rules_metrics[rule_metric] for rule_metric in combined_rules_metrics}}
                
    def rules_metrics_combiner(self, **rules_metrics):
        result = {}
        for rule_metric in rules_metrics:
            # if rule_metric == 'total_cpu_usage':
            #     result[rule_metric] = np.mean(rules_metrics[rule_metric])
            # else:
            result[rule_metric] = np.sum(rules_metrics[rule_metric])
        return result
        
    def get_rules_metrics(self, time_range):
        rules_metrics = self.run_saved_searches(time_range)
        saved_searches = rules_metrics['saved_searches']
        del rules_metrics['saved_searches']
        combined_rules_metrics = self.rules_metrics_combiner(**rules_metrics)
        new_line = self.post_process_metrics(time_range, saved_searches, combined_rules_metrics, rules_metrics)
        return new_line, combined_rules_metrics
        
    def get_duration_reward_values(self, time_range):
        new_line, combined_rules_metrics = self.get_rules_metrics(time_range)
        self.time_rules_energy.append(new_line)
        
        return combined_rules_metrics

class RewardStrategy1(RewardStrategy):
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        # no_agent_alert_val = self.no_agent_reward_values_dict['alerts'][-1]
        # alert_reward = ((no_agent_alert_val+1) - alert_val)/(no_agent_alert_val+1)
        alert_gap = alert_val - no_agent_alert_val
        # alert_reward = alert_gap/(no_agent_alert_val)
        # normalized_alert_reward = max(0, min(1, alert_reward)) # Normalize to be between 0 and 1
        
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # no_agent_duration_val = self.no_agent_reward_values_dict['duration'][-1]
        duration_reward = (duration_val - no_agent_duration_val)/(no_agent_duration_val)
        duration_reward = (duration_reward + 1)/2 # Normalize to be between 0 and 1
        self.reward_dict['duration'].append(duration_reward)

        ###### Distributions Component ######
        distributions_reward = self.get_partial_reward(real_distribution, fake_distribution)
        
        ###### Total Reward ######
        # total_reward = self.alpha * alert_reward + self.betta * distributions_reward + self.gamma * duration_reward
        total_reward = self.betta * distributions_reward + self.gamma * duration_reward
        
        if (no_agent_alert_val == 0) & (alert_gap > 3):
            total_reward = -(total_reward * alert_gap)
            self.reward_dict['alerts'].append(alert_gap)
            logger.info(f"alert gap as reward: {alert_gap}")
        elif (no_agent_alert_val == 0) & (alert_gap <= 3):
            total_reward = total_reward
            self.reward_dict['alerts'].append(alert_gap)
            logger.info(f"alert gap as reward: {alert_gap}")
            
        elif no_agent_alert_val > 0:
            alert_reward = alert_gap/no_agent_alert_val
            self.reward_dict['alerts'].append(alert_reward)
            
            if alert_reward <= 0.6:
                total_reward = total_reward * 100
            if alert_reward <= 1 and alert_reward > 0.6:
                total_reward = total_reward
            if alert_reward > 1:
                total_reward = -(100* alert_reward)

        
        return total_reward


class RewardStrategy2(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        # no_agent_alert_val = self.no_agent_reward_values_dict['alerts'][-1]
        # alert_reward = ((no_agent_alert_val+1) - alert_val)/(no_agent_alert_val+1)
        alert_gap = alert_val - no_agent_alert_val
        # alert_reward = alert_gap/(no_agent_alert_val)
        # normalized_alert_reward = max(0, min(1, alert_reward)) # Normalize to be between 0 and 1
        
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # no_agent_duration_val = self.no_agent_reward_values_dict['duration'][-1]
        duration_reward = (duration_val - no_agent_duration_val)/(no_agent_duration_val)
        duration_reward = (duration_reward + 1)/2 # Normalize to be between 0 and 1
        self.reward_dict['duration'].append(duration_reward)

        ###### Distributions Component ######
        distributions_reward = self.get_partial_reward(real_distribution, fake_distribution)
        
        ###### Total Reward ######
        # total_reward = self.alpha * alert_reward + self.betta * distributions_reward + self.gamma * duration_reward
        total_reward = self.betta * distributions_reward + self.gamma * duration_reward
        
        if (no_agent_alert_val == 0) & (alert_gap > 3):
            total_reward = -(total_reward * alert_gap)
            self.reward_dict['alerts'].append(alert_gap)
            logger.info(f"alert gap as reward: {alert_gap}")
        elif (no_agent_alert_val == 0) & (alert_gap <= 3):
            total_reward = total_reward
            self.reward_dict['alerts'].append(alert_gap)
            logger.info(f"alert gap as reward: {alert_gap}")
        elif no_agent_alert_val > 0:
            alert_reward = alert_gap/no_agent_alert_val
            self.reward_dict['alerts'].append(alert_reward)
            
            if alert_reward <= 0.6:
                total_reward = total_reward * 100
            if alert_reward <= 1 and alert_reward > 0.6:
                total_reward = total_reward
            if alert_reward > 1:
                total_reward = -(total_reward * alert_reward)

        
        return total_reward
    

class RewardStrategy3(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)

        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        # no_agent_alert_val = self.no_agent_reward_values_dict['alerts'][-1]
        # alert_reward = ((no_agent_alert_val+1) - alert_val)/(no_agent_alert_val+1)
        alert_gap = alert_val - no_agent_alert_val
        # alert_reward = alert_gap/(no_agent_alert_val)
        # normalized_alert_reward = max(0, min(1, alert_reward)) # Normalize to be between 0 and 1
        
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # no_agent_duration_val = self.no_agent_reward_values_dict['duration'][-1]
        duration_reward = (duration_val - no_agent_duration_val)/(no_agent_duration_val)
        duration_reward = (duration_reward + 1)/2 # Normalize to be between 0 and 1
        self.reward_dict['duration'].append(duration_reward)

        ###### Distributions Component ######
        distributions_reward = self.get_partial_reward(real_distribution, fake_distribution)
        
        ###### Total Reward ######
        # total_reward = self.alpha * alert_reward + self.betta * distributions_reward + self.gamma * duration_reward
        total_reward = self.betta * distributions_reward + self.gamma * duration_reward
        
        if (no_agent_alert_val == 0) & (alert_gap > 3):
            total_reward = distributions_reward * alert_gap
            self.reward_dict['alerts'].append(alert_gap)
            logger.info(f"alert gap as reward: {alert_gap}")
        elif (no_agent_alert_val == 0) & (alert_gap <= 3):
            total_reward = total_reward
            self.reward_dict['alerts'].append(alert_gap)
            logger.info(f"alert gap as reward: {alert_gap}")
        elif no_agent_alert_val > 0:
            alert_reward = alert_gap/no_agent_alert_val
            self.reward_dict['alerts'].append(alert_reward)
            
            if alert_reward <= 0.6:
                total_reward = total_reward * 100
            if alert_reward <= 1 and alert_reward > 0.6:
                total_reward = total_reward
            if alert_reward > 1:
                total_reward = distributions_reward * alert_reward

        
        return total_reward

class RewardStrategy4(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        # no_agent_alert_val = self.no_agent_reward_values_dict['alerts'][-1]
        # alert_reward = ((no_agent_alert_val+1) - alert_val)/(no_agent_alert_val+1)
        alert_gap = alert_val - no_agent_alert_val
        # alert_reward = alert_gap/(no_agent_alert_val)
        # normalized_alert_reward = max(0, min(1, alert_reward)) # Normalize to be between 0 and 1
        
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # no_agent_duration_val = self.no_agent_reward_values_dict['duration'][-1]
        duration_reward = (duration_val - no_agent_duration_val)/(no_agent_duration_val)
        duration_reward = (duration_reward + 1)/2 # Normalize to be between 0 and 1
        self.reward_dict['duration'].append(duration_reward)

        ###### Distributions Component ######
        distributions_reward = self.get_partial_reward(real_distribution, fake_distribution)
        
        ###### Total Reward ######
        # total_reward = self.alpha * alert_reward + self.betta * distributions_reward + self.gamma * duration_reward
        total_reward = self.betta * distributions_reward + self.gamma * duration_reward
        num_of_steps = len(self.reward_values_dict['distributions'])/len(self.reward_values_dict['alerts'])
        logger.info(f"num of steps: {num_of_steps}")
        logger.info(f"alert gap: {alert_gap}")
        if (no_agent_alert_val == 0) & (alert_gap > 3):
            total_reward = distributions_reward * alert_gap
            self.reward_dict['alerts'].append(alert_gap)
            logger.info(f"alert gap as reward: {alert_gap}")
        elif (no_agent_alert_val == 0) & (alert_gap <= 3):
            total_reward = total_reward
            self.reward_dict['alerts'].append(alert_gap)
            logger.info(f"alert gap as reward: {alert_gap}")
        elif no_agent_alert_val > 0:
            alert_reward = alert_gap/no_agent_alert_val
            self.reward_dict['alerts'].append(alert_reward)
            
            if alert_reward <= 0.6:
                total_reward = total_reward * 1000
            if alert_reward <= 1 and alert_reward > 0.6:
                total_reward = total_reward * num_of_steps
            if alert_reward > 1:
                total_reward = distributions_reward * alert_reward
            logger.info(f"alet reward: {alert_reward}")

        
        return total_reward

class RewardStrategy5(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        alert_gap = alert_val - no_agent_alert_val

        
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # no_agent_duration_val = self.no_agent_reward_values_dict['duration'][-1]
        duration_reward = (duration_val - no_agent_duration_val)/(no_agent_duration_val)
        duration_reward = (duration_reward + 1)/2 # Normalize to be between 0 and 1
        self.reward_dict['duration'].append(duration_reward)

        ###### Distributions Component ######
        distributions_reward = self.get_partial_reward(real_distribution, fake_distribution)
        
        ###### Total Reward ######
        # total_reward = self.alpha * alert_reward + self.betta * distributions_reward + self.gamma * duration_reward
        total_reward = self.betta * distributions_reward + self.gamma * duration_reward
        num_of_steps = len(self.reward_values_dict['distributions'])/len(self.reward_values_dict['alerts'])
        logger.info(f"num of steps: {num_of_steps}")
        logger.info(f"alert gap: {alert_gap}")
        if (no_agent_alert_val == 0) & (alert_gap > 3):
            total_reward = distributions_reward * alert_gap
            self.reward_dict['alerts'].append(alert_gap)
            logger.info(f"alert gap as reward: {alert_gap}")
        elif (no_agent_alert_val == 0) & (alert_gap <= 3):
            total_reward = total_reward
            self.reward_dict['alerts'].append(alert_gap)
            logger.info(f"alert gap as reward: {alert_gap}")
        elif no_agent_alert_val > 0:
            alert_reward = alert_gap/no_agent_alert_val
            self.reward_dict['alerts'].append(alert_reward)
            
            if alert_reward <= 1:
                total_reward = total_reward * 1000
            if alert_reward <= 2 and alert_reward > 1:
                total_reward = total_reward * num_of_steps
            if alert_reward > 2:
                total_reward = distributions_reward * alert_reward
            logger.info(f"alet reward: {alert_reward}")

        
        return total_reward

class RewardStrategy6(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # no_agent_duration_val = self.no_agent_reward_values_dict['duration'][-1]
        duration_reward = (duration_val - no_agent_duration_val)/(no_agent_duration_val)
        duration_reward = (duration_reward + 1)/2 # Normalize to be between 0 and 1
        self.reward_dict['duration'].append(duration_reward)        
        return duration_reward

class RewardStrategy7(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)

    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")       
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # no_agent_duration_val = self.no_agent_reward_values_dict['duration'][-1]
        duration_reward = (duration_val - no_agent_duration_val)/(no_agent_duration_val)
        self.reward_dict['duration'].append(duration_reward)        
        return duration_reward

class RewardStrategy8(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)

    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")       
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # no_agent_duration_val = self.no_agent_reward_values_dict['duration'][-1]
        duration_reward = duration_val
        self.reward_dict['duration'].append(duration_reward)        
        return duration_val
    
class RewardStrategy9(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")       
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # no_agent_duration_val = self.no_agent_reward_values_dict['duration'][-1]
        mean_duration_gap = (duration_val - no_agent_duration_val)
        std_duration_gap = (std_duration_val - no_agent_std_duration_val)
        duration_reward = (mean_duration_gap + std_duration_gap)/(no_agent_duration_val + no_agent_std_duration_val)
        self.reward_dict['duration'].append(duration_reward)        
        return duration_reward   

class RewardStrategy10(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        self.reward_values_dict['p_values'].append(p_value)
        duration_reward = 1 - p_value
        self.reward_dict['duration'].append(duration_reward)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")     
        return duration_reward

class RewardStrategy11(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        self.reward_values_dict['p_values'].append(p_value)
        duration_reward = t / p_value
        self.reward_dict['duration'].append(duration_reward)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        return duration_reward
    
class RewardStrategy12(RewardStrategy11):
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)

    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        sum_current_action = sum(current_action)
        if  sum_current_action > 3:
            logger.info(f"Action vaiolation: {sum_current_action}")
            return 1000 * (1 - sum_current_action)
        if  sum_current_action > 2:
            logger.info(f"Action vaiolation: {sum_current_action}")
            return 100 * (1 - sum_current_action)
        if  sum_current_action > 1:
            logger.info(f"Action vaiolation: {sum_current_action}")
            return 10 * (1 - sum_current_action)
        else:
            return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        partial_reward = self.get_partial_reward(real_distribution, fake_distribution, current_action)
        if partial_reward:
            return partial_reward
        return super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)

class RewardStrategy13(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        self.reward_values_dict['p_values'].append(p_value)
        duration_reward = 1 - p_value
        self.reward_dict['duration'].append(duration_reward)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        return duration_reward
    
class RewardStrategy14(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        distributions_distance = self.get_partial_reward_values(real_distribution, fake_distribution)
        distributions_reward = - distributions_distance
        self.reward_dict['distributions'].append(distributions_reward)
        return distributions_reward
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        partial_reward = self.get_partial_reward(real_distribution, fake_distribution, current_action)
        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        self.reward_values_dict['p_values'].append(p_value)
        self.reward_dict['duration'].append(p_value)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        if p_value < 0.05:
            return 0
        return partial_reward
    
class RewardStrategy15(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        action_variance = np.var(current_action)
        return action_variance
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        partial_reward = self.get_partial_reward(real_distribution, fake_distribution, current_action)
        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        self.reward_values_dict['p_values'].append(p_value)
        self.reward_dict['duration'].append(mean_duration_gap/no_agent_duration_val)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        if p_value < 0.05:
            return 10 * mean_duration_gap/no_agent_duration_val
        elif p_value > 0.05 and p_value < 0.1:
            return 5 * mean_duration_gap/no_agent_duration_val
        else:
            return partial_reward
        
class RewardStrategy16(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        self.reward_values_dict['p_values'].append(p_value)
        if p_value < 0.05:
            duration_reward = 1 - p_value
        else:
            duration_reward = -10 * p_value
        self.reward_dict['duration'].append(duration_reward)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        return duration_reward
    
class RewardStrategy17(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        distributions_distance = self.get_partial_reward_values(real_distribution, fake_distribution)
        distributions_reward = - distributions_distance
        self.reward_dict['distributions'].append(distributions_reward)
        return distributions_reward
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        self.reward_values_dict['p_values'].append(p_value)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        self.reward_dict['duration'].append(1 - p_value)
        if alert_val - no_agent_alert_val > 3:
            return -100 * (alert_val - no_agent_alert_val)
        if alert_val - no_agent_alert_val > 1:
            return -10 * (alert_val - no_agent_alert_val)
        if p_value < 0.05:
            return 10 * mean_duration_gap/no_agent_duration_val
        elif p_value > 0.05 and p_value < 0.1:
            return 5 * mean_duration_gap/no_agent_duration_val
        else:
            return 0
class RewardStrategy18(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)


        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        self.reward_values_dict['p_values'].append(p_value)
        if p_value < 0.05:
            duration_reward = (1 - p_value) * 1000
        elif p_value > 0.05 and p_value < 0.1:
            duration_reward = (1 - p_value) * 10
        else:
            duration_reward = -100 * p_value
        self.reward_dict['duration'].append(duration_reward)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        return duration_reward

class RewardStrategy19(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)
        alert_val, duration_val, std_duration_val = values['alert'], values['duration'], values['std_duration']
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['duration'], no_agent_values['std_duration']

        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        self.reward_values_dict['p_values'].append(p_value)
        duration_reward = 100000000000**(-p_value)
        self.reward_dict['duration'].append(duration_reward)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        return duration_reward
    
    
class RewardStrategy20(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)
        alert_val, duration_val, std_duration_val = values['alerts'], values['duration'], values['std_duration']
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alerts'], no_agent_values['duration'], no_agent_values['std_duration']

        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        self.reward_values_dict['p_values'].append(p_value)
        duration_reward = -np.log(p_value)
        self.reward_dict['duration'].append(duration_reward)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        return duration_reward
    
class RewardStrategy21(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)
        alert_val, duration_val, std_duration_val = values['alert'], values['duration'], values['std_duration']
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['duration'], no_agent_values['std_duration']

        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        self.reward_values_dict['p_values'].append(p_value)
        duration_reward = -np.log10(20*p_value)
        self.reward_dict['duration'].append(duration_reward)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        return duration_reward

class RewardStrategy22(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        if remaining_quota < 0:
            return -(remaining_quota**2)
        
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)
        alert_val, duration_val, std_duration_val = values['alert'], values['duration'], values['std_duration']
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['duration'], no_agent_values['std_duration']

        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        
        self.reward_values_dict['p_values'].append(p_value)
        duration_reward = 10 * ((1/p_value)**2)
        self.reward_dict['duration'].append(duration_reward)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        return duration_reward

class RewardStrategy23(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0

    def check_episodic_agent_violation(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        return 0
    
    def get_step_violation_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        if remaining_quota < 0:
            return -(remaining_quota**2)
        elif remaining_quota > 0:
            return -remaining_quota
        else:
            return 0
    
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)
        alert_val, duration_val, std_duration_val = values['alert'], values['duration'], values['std_duration']
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['duration'], no_agent_values['std_duration']
        logger.info(f"alert value: {alert_val}, no_agent_alert_val: {no_agent_alert_val}")
        logger.info(f"duration value: {duration_val}, no_agent_duration_val: {no_agent_duration_val}")
        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        
        self.reward_values_dict['p_values'].append(p_value)
        duration_reward = 100 * ((1/p_value)**2)
        self.reward_dict['duration'].append(duration_reward)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        return duration_reward

class RewardStrategy24(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)
        self.p_value_threshold = 0.3
    
    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        distributions_distance = self.current_distributions_distance
        # distributions_distance = self.get_partial_reward_values(real_distribution, fake_distribution)
        self.current_distributions_distance = distributions_distance
        if distributions_distance > self.distribution_threshold:
            return -1000*distributions_distance
        if distributions_distance == self.epsilon:
            return 0
        return -distributions_distance

    def check_episodic_agent_violation(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        distribution_reward = self.get_partial_reward(real_distribution, fake_distribution, current_action)
        if self.current_distributions_distance > self.distribution_threshold:
            return distribution_reward
        else:
            return 0
    
    def get_step_violation_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        if remaining_quota < 0:
            return -(remaining_quota**2)
        else:
            return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)
        alert_val, duration_val, std_duration_val = values['alert'], values['duration'], values['std_duration']
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['duration'], no_agent_values['std_duration']
        logger.info(f"alert value: {alert_val}, no_agent_alert_val: {no_agent_alert_val}")
        logger.info(f"duration value: {duration_val}, no_agent_duration_val: {no_agent_duration_val}")
        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        self.reward_values_dict['p_values'].append(p_value)
        
        if p_value > self.p_value_threshold:
            duration_reward = -100000*p_value
            total_reward = duration_reward 
        else:
            duration_reward = 100*((1/p_value)**5)
            total_reward = duration_reward
            
        self.reward_dict['duration'].append(duration_reward)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        return total_reward
    
class RewardStrategy25(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)
        self.p_value_threshold = 0.3
        self.alert_threshold = 0.2
    
    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        distributions_distance = self.current_distributions_distance
        # distributions_distance = self.get_partial_reward_values(real_distribution, fake_distribution)
        self.current_distributions_distance = distributions_distance
        if distributions_distance > self.distribution_threshold:
            distribution_reward = -1000*distributions_distance
        elif distributions_distance == self.epsilon:
            distribution_reward = 0
        else:
            distribution_reward = -distributions_distance
        self.reward_dict['distributions'].append(distribution_reward)            
        logger.info(f"current_distributions_distance: {self.current_distributions_distance}")
        
    def check_episodic_agent_violation(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        distribution_reward = self.get_partial_reward(real_distribution, fake_distribution, current_action)
        if self.current_distributions_distance > self.distribution_threshold:
            return distribution_reward
        else:
            return 0
    
    def get_step_violation_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        if remaining_quota < 0:
            return -(remaining_quota**2)
        else:
            return 0
    
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)
        alert_val, duration_val, std_duration_val = values['alert'], values['duration'], values['std_duration']
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['duration'], no_agent_values['std_duration']
        logger.info(f"alert value: {alert_val}, no_agent_alert_val: {no_agent_alert_val}")
        logger.info(f"duration value: {duration_val}, no_agent_duration_val: {no_agent_duration_val}")
        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        self.reward_values_dict['p_values'].append(p_value)
        alert_gap = alert_val - no_agent_alert_val
        if no_agent_alert_val == 0:
            if alert_gap > 2:
                alert_reward = -1000*alert_gap
            else:
                alert_reward = 0
        if no_agent_alert_val > 0:
            alert_increase = alert_gap / no_agent_alert_val
            if alert_increase > self.alert_threshold:
                alert_reward = -1000*alert_increase
            else:
                alert_reward = 0
        self.reward_dict['alerts'].append(alert_reward)
        if p_value > self.p_value_threshold:
            duration_reward = -100000*p_value
            total_reward = duration_reward 
        else:
            duration_reward = 100*((1/p_value)**5)
            total_reward = duration_reward
            
        self.reward_dict['duration'].append(duration_reward)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        return total_reward

class RewardStrategy26(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0

    def check_episodic_agent_violation(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        return 0
    
    def get_step_violation_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        if remaining_quota < 0:
            return -(remaining_quota**2)
        elif remaining_quota > 0:
            return -remaining_quota
        else:
            return 0
    
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values, no_agent_values = super().get_full_reward(time_range, real_distribution, fake_distribution, current_action, remaining_quota)
        alert_val, duration_val, std_duration_val = values['alert'], values['cpu'], values['std_cpu']
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['cpu'], no_agent_values['std_cpu']
        logger.info(f"alert value: {alert_val}, no_agent_alert_val: {no_agent_alert_val}")
        logger.info(f"duration value: {duration_val}, no_agent_duration_val: {no_agent_duration_val}")
        # welch t test
        n = self.splunk_tools.num_of_measurements
        mean_duration_gap = (duration_val - no_agent_duration_val)
        t = mean_duration_gap/np.sqrt((std_duration_val**2)/n + (no_agent_std_duration_val**2)/n)
        self.reward_values_dict['t_values'].append(t)
        degree_of_freedom = (std_duration_val**2/n + no_agent_std_duration_val**2/n)**2 / ((std_duration_val**2/n)**2/(n-1) + (no_agent_std_duration_val**2/n)**2/(n-1))
        self.reward_values_dict['degrees_of_freedom'].append(degree_of_freedom)
        p_value = 1 - stats.t.cdf(t, degree_of_freedom)
        
        self.reward_values_dict['p_values'].append(p_value)
        duration_reward = 100 * ((1/p_value)**2)
        self.reward_dict['duration'].append(duration_reward)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        return duration_reward
class RewardStrategy27(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)
        self.cpus = []
        self.cpu_avg = 4300

    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0

    def check_episodic_agent_violation(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        return 0
    
    def get_step_violation_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        if remaining_quota < 0:
            return -(remaining_quota**2)
        elif remaining_quota > 0:
            return -remaining_quota
        else:
            return 0

    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values =  self.get_duration_reward_values(time_range)
        clean_env(self.splunk_tools, time_range)
        alert_val, duration_val, std_duration_val, total_cpu_usage = values['alert'], values['cpu'], values['std_cpu'], values['total_cpu_usage']
        # no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['cpu'], no_agent_values['std_cpu']
        # logger.info(f"alert value: {alert_val}, no_agent_alert_val: {no_agent_alert_val}")
        logger.info(f"total_cpu_usage value: {total_cpu_usage}")
        # self.cpus.append(total_cpu_usage)
        # self.cpu_avg = np.mean(self.cpus)
        if total_cpu_usage > self.cpu_avg:
            return total_cpu_usage**2
        else:
            return total_cpu_usage
    
    def get_rules_metrics(self, time_range):
        rules_metrics = self.run_saved_searches(time_range)
        saved_searches = rules_metrics['saved_searches']
        del rules_metrics['saved_searches']
        combined_rules_metrics = self.rules_metrics_combiner(**rules_metrics)
        new_line = self.post_process_metrics(time_range, saved_searches, combined_rules_metrics, rules_metrics)
        return new_line, combined_rules_metrics
        
    def get_duration_reward_values(self, time_range):
        new_line, combined_rules_metrics = self.get_rules_metrics(time_range)
        self.time_rules_energy.append(new_line)
        return combined_rules_metrics
        
    def run_saved_searches(self, time_range):
        alert_vals, duration_vals, std_duration_vals, saved_searches, mean_cpu_integrals, std_cpu_integrals, read_count, write_count, read_bytes, write_bytes, total_cpu_usage = self.splunk_tools.run_saved_searches_parallel(time_range)
        return {"alert":alert_vals, "duration":duration_vals, "std_duration":std_duration_vals, "saved_searches":saved_searches, "cpu":mean_cpu_integrals, "std_cpu":std_cpu_integrals, "read_count":read_count, "write_count":write_count, "read_bytes":read_bytes, "write_bytes":write_bytes, "total_cpu_usage":total_cpu_usage}

class RewardStrategy28(RewardStrategy27):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)
        self.cpus = []
        
    def rules_metrics_combiner(self, **rules_metrics):
        result = {}
        for rule_metric in rules_metrics:
            if rule_metric == 'total_cpu_usage':
                result['median_cpu_usage'] = np.median(rules_metrics[rule_metric])
            else:
                result[rule_metric] = np.sum(rules_metrics[rule_metric])
        return result
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values =  self.get_duration_reward_values(time_range)
        clean_env(self.splunk_tools, time_range)
        alert_val, duration_val, std_duration_val, median_cpu_usage = values['alert'], values['cpu'], values['std_cpu'], values['median_cpu_usage']
        # no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['cpu'], no_agent_values['std_cpu']
        # logger.info(f"alert value: {alert_val}, no_agent_alert_val: {no_agent_alert_val}")
        logger.info(f"median_cpu_usage value: {median_cpu_usage}")
        # self.cpus.append(total_cpu_usage)
        # self.cpu_avg = np.mean(self.cpus)
        return median_cpu_usage

class RewardStrategy29(RewardStrategy27):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)
        self.cpus = []
        
    def rules_metrics_combiner(self, **rules_metrics):
        result = {}
        for rule_metric in rules_metrics:
            if rule_metric == 'total_cpu_usage':
                result['median_cpu_usage'] = np.median(rules_metrics[rule_metric])
            else:
                result[rule_metric] = np.sum(rules_metrics[rule_metric])
        return result
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values =  self.get_duration_reward_values(time_range)
        clean_env(self.splunk_tools, time_range)
        alert_val, duration_val, std_duration_val, median_cpu_usage = values['alert'], values['duration'], values['std_cpu'], values['median_cpu_usage']
        # no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['cpu'], no_agent_values['std_cpu']
        # logger.info(f"alert value: {alert_val}, no_agent_alert_val: {no_agent_alert_val}")
        logger.info(f"duration_val value: {duration_val}")
        # self.cpus.append(total_cpu_usage)
        # self.cpu_avg = np.mean(self.cpus)
        return duration_val

class RewardStrategy30(RewardStrategy27):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)
        self.cpus = []
        
    def rules_metrics_combiner(self, **rules_metrics):
        result = {}
        for rule_metric in rules_metrics:
            if rule_metric == 'total_cpu_usage':
                result['median_cpu_usage'] = np.median(rules_metrics[rule_metric])
            else:
                result[rule_metric] = np.sum(rules_metrics[rule_metric])
        return result
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values =  self.get_duration_reward_values(time_range)
        clean_env(self.splunk_tools, time_range)
        alert_val, cpu_val, std_duration_val, median_cpu_usage = values['alert'], values['cpu'], values['std_cpu'], values['median_cpu_usage']
        # no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['cpu'], no_agent_values['std_cpu']
        # logger.info(f"alert value: {alert_val}, no_agent_alert_val: {no_agent_alert_val}")
        logger.info(f"cpu_val value: {cpu_val}")
        # self.cpus.append(total_cpu_usage)
        # self.cpu_avg = np.mean(self.cpus)
        return cpu_val
    
class RewardStrategy31(RewardStrategy27):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)
        self.cpus = []
        
    def rules_metrics_combiner(self, **rules_metrics):
        result = {}
        for rule_metric in rules_metrics:
            if rule_metric == 'total_cpu_usage':
                result['median_cpu_usage'] = np.median(rules_metrics[rule_metric])
            else:
                result[rule_metric] = np.sum(rules_metrics[rule_metric])
        return result
    

    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        after_metrics =  self.get_duration_reward_values(time_range)
        clean_env(self.splunk_tools, time_range)
        before_metrics =  self.get_no_agent_reward(time_range)
        self.no_agent_last_row = self.no_agent_current_row
        after_alert_val, after_cpu_val, after_std_duration_val, after_median_cpu_usage = after_metrics['alert'], after_metrics['cpu'], after_metrics['std_cpu'], after_metrics['median_cpu_usage']
        before_alert_val, before_cpu_val, before_std_duration_val, before_median_cpu_usage = before_metrics['alert'], before_metrics['cpu'], before_metrics['std_cpu'], before_metrics['median_cpu_usage']
        logger.info(f"CPU gap: {after_median_cpu_usage - before_median_cpu_usage}")
        return after_median_cpu_usage - before_median_cpu_usage

class RewardStrategy32(RewardStrategy27):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)
        self.cpus = []
        
    def rules_metrics_combiner(self, **rules_metrics):
        result = {}
        for rule_metric in rules_metrics:
            if rule_metric == 'total_cpu_usage':
                result['median_cpu_usage'] = np.median(rules_metrics[rule_metric])
            else:
                result[rule_metric] = np.sum(rules_metrics[rule_metric])
        return result
    

    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        after_metrics =  self.get_duration_reward_values(time_range)
        clean_env(self.splunk_tools, time_range)
        before_metrics =  self.get_no_agent_reward(time_range)
        self.no_agent_last_row = self.no_agent_current_row
        after_alert_val, after_cpu_val, after_std_duration_val, after_median_cpu_usage = after_metrics['alert'], after_metrics['cpu'], after_metrics['std_cpu'], after_metrics['median_cpu_usage']
        before_alert_val, before_cpu_val, before_std_duration_val, before_median_cpu_usage = before_metrics['alert'], before_metrics['cpu'], before_metrics['std_cpu'], before_metrics['median_cpu_usage']
        logger.info(f"CPU gap: {after_cpu_val - before_cpu_val}")
        return after_cpu_val - before_cpu_val

class RewardStrategy33(RewardStrategy27):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)
        self.cpus = []
        
    def rules_metrics_combiner(self, **rules_metrics):
        result = {}
        for rule_metric in rules_metrics:
            if rule_metric == 'total_cpu_usage':
                result['median_cpu_usage'] = np.median(rules_metrics[rule_metric])
            else:
                result[rule_metric] = np.sum(rules_metrics[rule_metric])
        return result
    

    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        after_metrics =  self.get_duration_reward_values(time_range)
        clean_env(self.splunk_tools, time_range)
        before_metrics =  self.get_no_agent_reward(time_range)
        self.no_agent_last_row = self.no_agent_current_row
        after_alert_val, after_cpu_val, after_std_duration_val, after_median_cpu_usage = after_metrics['alert'], after_metrics['cpu'], after_metrics['std_cpu'], after_metrics['median_cpu_usage']
        before_alert_val, before_cpu_val, before_std_duration_val, before_median_cpu_usage = before_metrics['alert'], before_metrics['cpu'], before_metrics['std_cpu'], before_metrics['median_cpu_usage']
        logger.info(f"CPU gap: {after_cpu_val - before_cpu_val}")
        return after_cpu_val - before_cpu_val
    
class RewardStrategy34(RewardStrategy27):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)
        self.cpus = []
        
    def rules_metrics_combiner(self, **rules_metrics):
        result = {}
        for rule_metric in rules_metrics:
            if rule_metric == 'total_cpu_usage':
                result['median_cpu_usage'] = np.sum(rules_metrics[rule_metric])
            else:
                result[rule_metric] = np.sum(rules_metrics[rule_metric])
        return result
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values =  self.get_duration_reward_values(time_range)
        clean_env(self.splunk_tools, time_range)
        alert_val, duration_val, std_duration_val, median_cpu_usage = values['alert'], values['cpu'], values['std_cpu'], values['median_cpu_usage']
        # no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['cpu'], no_agent_values['std_cpu']
        # logger.info(f"alert value: {alert_val}, no_agent_alert_val: {no_agent_alert_val}")
        logger.info(f"median_cpu_usage value: {median_cpu_usage}")
        # self.cpus.append(total_cpu_usage)
        # self.cpu_avg = np.mean(self.cpus)
        return median_cpu_usage**2
    
class RewardStrategy35(RewardStrategy27):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)
        self.cpus = []
        
    def rules_metrics_combiner(self, **rules_metrics):
        result = {}
        for rule_metric in rules_metrics:
            if rule_metric == 'total_cpu_usage':
                result['median_cpu_usage'] = np.trapz(rules_metrics[rule_metric], dx=.1)
            else:
                result[rule_metric] = np.sum(rules_metrics[rule_metric])
        return result
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        values =  self.get_duration_reward_values(time_range)
        clean_env(self.splunk_tools, time_range)
        before_metrics =  self.get_no_agent_reward(time_range)
        self.no_agent_last_row = self.no_agent_current_row
        alert_val, duration_val, std_duration_val, median_cpu_usage = values['alert'], values['cpu'], values['std_cpu'], values['median_cpu_usage']
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val, no_agent_median_cpu_usage = before_metrics['alert'], before_metrics['cpu'], before_metrics['std_cpu'], before_metrics['median_cpu_usage']
        # no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['cpu'], no_agent_values['std_cpu']
        # logger.info(f"alert value: {alert_val}, no_agent_alert_val: {no_agent_alert_val}")
        logger.info(f"median_cpu_usage value: {median_cpu_usage}")
        logger.info(f"no_agent_median_cpu_usage value: {no_agent_median_cpu_usage}")
        # self.cpus.append(total_cpu_usage)
        # self.cpu_avg = np.mean(self.cpus)
        return (median_cpu_usage-no_agent_median_cpu_usage)
    
class RewardStrategy36(RewardStrategy27):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)
        self.cpus = []
        self.distribution_threshold = 10
        
    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        # distributions_distance = self.current_distributions_distance
        distributions_distance = self.get_partial_reward_values(real_distribution, fake_distribution)
        self.current_distributions_distance = distributions_distance
        # if distributions_distance == np.inf:
        #     return -1000
        # if distributions_distance > self.distribution_threshold:
        #     distribution_reward = -100*distributions_distance
        # elif distributions_distance == self.epsilon:
        #     distribution_reward = 0
        # else:
        #     distribution_reward = -distributions_distance
        # self.reward_dict['distributions'].append(distribution_reward)            
        logger.info(f"current_distributions_distance: {self.current_distributions_distance}")
        # return distribution_reward
        # random_val = np.random.randint(0, 10)
        # if random_val < 5:
        #     return 1
        if distributions_distance == self.epsilon:
            return 0
        return 1/distributions_distance

    def check_episodic_agent_violation(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        distribution_reward = self.get_partial_reward(real_distribution, fake_distribution, current_action)
        # if self.current_distributions_distance > self.distribution_threshold:
        #     return -self.current_distributions_distance
        # else:
        return 0
        
    def get_step_violation_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota, step_counter):
        if remaining_quota < 0:
            return (remaining_quota)/step_counter
        # elif remaining_quota > 0:
        #     return -remaining_quota
        else:
            return 0
    
    def rules_metrics_combiner(self, **rules_metrics):
        result = {}
        for rule_metric in rules_metrics:
            if rule_metric == 'total_cpu_usage':
                result['median_cpu_usage'] = np.trapz(np.array(rules_metrics[rule_metric])/100, dx=.1)
            else:
                result[rule_metric] = np.sum(rules_metrics[rule_metric])
        return result
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota, step_counter):
        if self.current_distributions_distance == self.epsilon:
            return 0
        values =  self.get_duration_reward_values(time_range)
        clean_env(self.splunk_tools, time_range)
        before_metrics =  self.get_no_agent_reward(time_range)
        self.no_agent_last_row = self.no_agent_current_row
        alert_val, duration_val, std_duration_val, median_cpu_usage = values['alert'], values['cpu'], values['std_cpu'], values['median_cpu_usage']
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val, no_agent_median_cpu_usage = before_metrics['alert'], before_metrics['cpu'], before_metrics['std_cpu'], before_metrics['median_cpu_usage']
        # no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['cpu'], no_agent_values['std_cpu']
        # logger.info(f"alert value: {alert_val}, no_agent_alert_val: {no_agent_alert_val}")
        logger.info(f"median_cpu_usage value: {median_cpu_usage}")
        logger.info(f"no_agent_median_cpu_usage value: {no_agent_median_cpu_usage}")
        # self.cpus.append(total_cpu_usage)
        # self.cpu_avg = np.mean(self.cpus)
        energy_reward = (median_cpu_usage-no_agent_median_cpu_usage)/no_agent_median_cpu_usage
        if energy_reward < 0:
            energy_reward = 0 
        self.reward_dict['duration'].append(energy_reward)

        return energy_reward*step_counter/self.current_distributions_distance
    
class RewardStrategy37(RewardStrategy27):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)
        self.cpus = []
        self.distribution_threshold = 10
        
    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        # distributions_distance = self.current_distributions_distance
        distributions_distance = self.get_partial_reward_values(real_distribution, fake_distribution)
        self.current_distributions_distance = distributions_distance
        # if distributions_distance == np.inf:
        #     return -1000
        # if distributions_distance > self.distribution_threshold:
        #     distribution_reward = -100*distributions_distance
        # elif distributions_distance == self.epsilon:
        #     distribution_reward = 0
        # else:
        #     distribution_reward = -distributions_distance
        # self.reward_dict['distributions'].append(distribution_reward)            
        logger.info(f"current_distributions_distance: {self.current_distributions_distance}")
        # return distribution_reward
        # random_val = np.random.randint(0, 10)
        # if random_val < 5:
        #     return 1
        # if distributions_distance == self.epsilon:
        #     return 0
        # return 1/distributions_distance
        return -distributions_distance

    def check_episodic_agent_violation(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota):
        distribution_reward = self.get_partial_reward(real_distribution, fake_distribution, current_action)
        # if self.current_distributions_distance > self.distribution_threshold:
        #     return -self.current_distributions_distance
        # else:
        return 0
        
    def get_step_violation_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota, step_counter):
        # if remaining_quota < 0:
        #     return (remaining_quota)/step_counter
        if remaining_quota >= 1.5:
            return -remaining_quota**2
        # elif remaining_quota > 0:
        #     return -remaining_quota
        else:
            return 0
    
    def rules_metrics_combiner(self, **rules_metrics):
        result = {}
        for rule_metric in rules_metrics:
            if rule_metric == 'total_cpu_usage':
                result['median_cpu_usage'] = np.trapz(np.array(rules_metrics[rule_metric])/100, dx=.1)
            else:
                result[rule_metric] = np.sum(rules_metrics[rule_metric])
        return result
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota, step_counter):
        if self.current_distributions_distance == self.epsilon:
            return 0
        values =  self.get_duration_reward_values(time_range)
        clean_env(self.splunk_tools, time_range)
        before_metrics =  self.get_no_agent_reward(time_range)
        self.no_agent_last_row = self.no_agent_current_row
        alert_val, duration_val, std_duration_val, median_cpu_usage = values['alert'], values['cpu'], values['std_cpu'], values['median_cpu_usage']
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val, no_agent_median_cpu_usage = before_metrics['alert'], before_metrics['cpu'], before_metrics['std_cpu'], before_metrics['median_cpu_usage']
        # no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['cpu'], no_agent_values['std_cpu']
        # logger.info(f"alert value: {alert_val}, no_agent_alert_val: {no_agent_alert_val}")
        logger.info(f"median_cpu_usage value: {median_cpu_usage}")
        logger.info(f"no_agent_median_cpu_usage value: {no_agent_median_cpu_usage}")
        # self.cpus.append(total_cpu_usage)
        # self.cpu_avg = np.mean(self.cpus)
        energy_reward = (median_cpu_usage-no_agent_median_cpu_usage)/no_agent_median_cpu_usage
        if energy_reward < 0:
            energy_reward = 0 
        self.reward_dict['duration'].append(energy_reward)

        return energy_reward/self.current_distributions_distance

class RewardStrategy38(RewardStrategy37):
    
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma, no_agent_table_path=no_agent_table_path)
        self.cpus = []
        self.distribution_threshold = 10
        

    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action, remaining_quota, step_counter):
        if self.current_distributions_distance == self.epsilon:
            return 0
        values =  self.get_duration_reward_values(time_range)
        clean_env(self.splunk_tools, time_range)
        before_metrics =  self.get_no_agent_reward(time_range)
        self.no_agent_last_row = self.no_agent_current_row
        alert_val, duration_val, std_duration_val, median_cpu_usage = values['alert'], values['cpu'], values['std_cpu'], values['median_cpu_usage']
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val, no_agent_median_cpu_usage = before_metrics['alert'], before_metrics['cpu'], before_metrics['std_cpu'], before_metrics['median_cpu_usage']
        # no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = no_agent_values['alert'], no_agent_values['cpu'], no_agent_values['std_cpu']
        # logger.info(f"alert value: {alert_val}, no_agent_alert_val: {no_agent_alert_val}")
        logger.info(f"median_cpu_usage value: {median_cpu_usage}")
        logger.info(f"no_agent_median_cpu_usage value: {no_agent_median_cpu_usage}")
        # self.cpus.append(total_cpu_usage)
        # self.cpu_avg = np.mean(self.cpus)
        energy_reward = (median_cpu_usage-no_agent_median_cpu_usage)/no_agent_median_cpu_usage
        if energy_reward < 0:
            energy_reward = 0 
        self.reward_dict['duration'].append(energy_reward)

        return energy_reward/self.current_distributions_distance
