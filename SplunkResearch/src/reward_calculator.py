import datetime
import os
import subprocess
from time import sleep
import numpy as np
from scipy.stats import entropy, wasserstein_distance
import pandas as pd
from scipy.spatial.distance import jensenshannon
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)
CPU_TDP = 200
PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.92.1.el7.x86_64/Splunk Enterprise SIEM/Power Saver Plan/One Scan/'

class RewardCalc:
    def __init__(self, top_logtypes, dt_manager, splunk_tools, rule_frequency, num_of_searches, measurment_tool, alpha, beta, gamma, summary_writer):
        self.average_energy = 0
        self.average_alert = 0
        self.average_duration = 0
        self.epsilon = 0
        self.action_upper_bound = 1
        self.reward_dict = {'energy': [0], 'alerts': [0], 'distributions': [], 'duration': [], 'total': []}
        self.reward_values_dict = {'energy': [], 'alerts': [], 'distributions': [], 'duration': [], "num_of_rules":[]}
        self.top_logtypes = top_logtypes
        self.dt_manager = dt_manager
        self.splunk_tools = splunk_tools
        self.rule_frequency = rule_frequency
        self.time_rules_energy = []
        self.current_measurement_path = ''
        self.num_of_searches = num_of_searches  
        self.distribution_threshold = 0.3
        self.alert_threshold = 0
        self.measurment_tool = measurment_tool
        self.alpha = alpha
        self.betta = beta
        self.gamma = gamma
        self.no_agent_reward_values_dict = {'energy': [], 'alerts': [], 'distributions': [], 'duration': []}
        
        
        self.summary_writer = summary_writer

    def get_partial_reward(self, real_distribution, current_state):
        distributions_distance = self.get_partial_reward_values(real_distribution, current_state)
        return 1/(distributions_distance+1)
    
    def get_no_agent_reward(self, time_range):
        alert_vals, duration_vals = self.splunk_tools.run_saved_searches(time_range)
        alert_val = sum(alert_vals)
        duration_val = sum(duration_vals)
        self.no_agent_reward_values_dict['alerts'].append(alert_val)
        self.no_agent_reward_values_dict['duration'].append(duration_val)
        logger.info(f"no agent alert value: {alert_val}")
        logger.info(f"no agent duration value: {duration_val}")
        with self.summary_writer.as_default():
            tf.summary.scalar('no_agent_alert_val', alert_val, step=len(self.no_agent_reward_values_dict['alerts']))
            tf.summary.scalar('no_agent_duration_val', duration_val, step=len(self.no_agent_reward_values_dict['duration']))
    
    def get_full_reward(self, time_range, real_distribution, current_state):
        alert_vals, duration_vals = self.splunk_tools.run_saved_searches(time_range)
        alert_val = sum(alert_vals)
        duration_val = sum(duration_vals)
        self.time_rules_energy.append({'time_range':str(time_range), 'rules':duration_vals})
        
        ###### Alert Component ######
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        no_agent_alert_val = self.no_agent_reward_values_dict['alerts'][-1]
        # alert_reward = ((no_agent_alert_val+1) - alert_val)/(no_agent_alert_val+1)
        alert_reward = (alert_val - no_agent_alert_val)/(no_agent_alert_val+0.0000000001)
        # normalized_alert_reward = max(0, min(1, alert_reward)) # Normalize to be between 0 and 1
        self.reward_dict['alerts'].append(alert_reward)
        
        ###### Duration Component ######
        logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['duration'].append(duration_val)
        no_agent_duration_val = self.no_agent_reward_values_dict['duration'][-1]
        duration_reward = (duration_val - no_agent_duration_val)/(no_agent_duration_val)
        duration_reward = (duration_reward + 1)/2 # Normalize to be between 0 and 1
        self.reward_dict['duration'].append(duration_reward)

        ###### Distributions Component ######
        distributions_reward = self.get_partial_reward(real_distribution, current_state)
        self.reward_dict['distributions'].append(distributions_reward)
        
        ###### Total Reward ######
        # total_reward = self.alpha * alert_reward + self.betta * distributions_reward + self.gamma * duration_reward
        total_reward = self.betta * distributions_reward + self.gamma * duration_reward
        
        if alert_reward < 0.5:
            total_reward = total_reward * 1.5
        if alert_reward < 1 and alert_reward > 0.5:
            total_reward = total_reward
        if alert_reward > 1:
            total_reward = -(total_reward ** 2)
        
        # if alert_reward < -1:
        #     total_reward = alert_reward
        # Log reward values to TensorBoard
        with self.summary_writer.as_default():
            tf.summary.scalar('alert_reward', alert_reward, step=len(self.reward_dict['alerts']))
            tf.summary.scalar('distributions_reward', distributions_reward, step=len(self.reward_dict['distributions']))
            tf.summary.scalar('duration_reward', duration_reward, step=len(self.reward_dict['duration']))
            tf.summary.scalar('alert_val', alert_val, step=len(self.reward_values_dict['alerts']))
            tf.summary.scalar('distributions_val', distributions_reward, step=len(self.reward_values_dict['distributions']))
            tf.summary.scalar('duration_val', duration_val, step=len(self.reward_values_dict['duration']))
            

        
        return total_reward

    def update_average_values(self):
        self.average_energy = sum(self.reward_values_dict['energy']) / len(self.reward_values_dict['energy'])
        self.average_alert = sum(self.reward_values_dict['alerts']) / len(self.reward_values_dict['alerts'])
        self.average_duration = sum(self.reward_values_dict['duration']) / len(self.reward_values_dict['duration'])
        logger.info(f"average energy: {self.average_energy}")
        logger.info(f"average alert: {self.average_alert}")
        logger.info(f"average duration: {self.average_duration}")
        
    def get_full_reward_values(self, time_range):
        logger.info('wait til next rule frequency')
        self.dt_manager.wait_til_next_rule_frequency(self.rule_frequency)
        rule_total_energy_dict = self.measurment_tool.measure(time_range=time_range, time_delta=self.rule_frequency)
        self.time_rules_energy.append({'time_range':str(time_range), 'rules':rule_total_energy_dict})
        energy_val = 0
        if 'CPU(J)' in rule_total_energy_dict[0]:
            energy_val = sum([rule['CPU(J)'] for rule in rule_total_energy_dict])
            logger.info(f"energy value: {energy_val}")
            self.reward_values_dict['energy'].append(energy_val)
        duration_val = sum([rule['run_duration'] for rule in rule_total_energy_dict])           

        sids = [rule['sid'] for rule in rule_total_energy_dict]
        alert_val = self.splunk_tools.get_alert_count(sids)

        energy_increase = 0
        duration_increase = 0
        return alert_val, energy_val, energy_increase, duration_val, duration_increase
    
    def get_partial_reward_values(self, real_distribution, current_state):
        distributions_val = self.compare_distributions(current_state[:len(self.top_logtypes)], current_state[len(self.top_logtypes):])     
        if distributions_val == 0:
            distributions_val = distributions_val + 0.000000000001
        self.reward_values_dict['distributions'].append(distributions_val)
        logger.info(f"distributions value: {distributions_val}")
        return distributions_val
    
    def compare_distributions(self, dist1, dist2):
        logger.info(f"dist1: {dist1}")
        logger.info(f"dist2: {dist2}")
        return wasserstein_distance(dist1, dist2)
