import datetime
import os
import subprocess
from time import sleep
import numpy as np
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import pandas as pd
from scipy.spatial.distance import jensenshannon  

CPU_TDP=200
PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.92.1.el7.x86_64/Splunk Enterprise SIEM/Power Saver Plan/One Scan/'
import logging
logger = logging.getLogger(__name__)

class RewardCalc:
    def __init__(self, relevant_logtypes, dt_manager, splunk_tools, rule_frequency, num_of_searches, measurment_tool, alpha, beta, gamma):
        # self.previous_energy = 1
        # self.previous_alert = 0
        self.average_energy = 0
        self.average_alert = 0
        self.average_duration = 0
        self.epsilon = 0
        self.action_upper_bound = 1
        self.reward_dict = {'energy': [0], 'alerts': [0], 'distributions': [], 'fraction': [], 'total': []}
        self.reward_values_dict = {'energy': [], 'alerts': [], 'distributions': [], 'fraction': [], 'duration': [], "num_of_rules":[]}
        self.relevant_logtypes = relevant_logtypes
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
      
    def get_previous_full_reward(self):
        return self.reward_dict['alerts'][-1], self.reward_dict['energy'][-1]

    def get_partial_reward(self, real_distribution, current_state):
        fraction_val, distributions_val = self.get_partial_reward_values(real_distribution, current_state)
        # return 0.5/(distributions_val+0.000000000001) + 0.5*(fraction_val)
        # if distributions_val > self.distribution_threshold:
        #     return -1
        # else:
        #     return 1
        return 1/(distributions_val+1)
    
    def get_log_type_index(self, current_state):
        return current_state[-4]
    
    def get_fraction_state(self, current_state):
        return current_state[-1]
    
    def get_full_reward(self, time_range, real_distribution, current_state):
        # fraction_val, distributions_val = self.get_partial_reward_values(real_distribution, current_state)
        alert_val, energy_val, energy_increase, duration_val, duration_increase = self.get_full_reward_values(time_range=time_range)
        # self.update_average_values()
        # if alert_val > self.alert_threshold:
        #     return -10
        if alert_val > int(self.num_of_searches/2):
            return -alert_val
        if duration_val > 1.5*self.num_of_searches*2:
            return 10
        alert_reward = 1/(alert_val+1)
        distributions_reward = self.get_partial_reward(real_distribution, current_state)
        duration_reward = duration_val/(self.num_of_searches*2)
        return self.alpha*alert_reward + self.betta*distributions_reward + self.gamma*duration_reward
        
    def update_average_values(self):
        self.average_energy = sum(self.reward_values_dict['energy'])/len(self.reward_values_dict['energy'])
        self.average_alert = sum(self.reward_values_dict['alerts'])/len(self.reward_values_dict['alerts'])
        self.average_duration = sum(self.reward_values_dict['duration'])/len(self.reward_values_dict['duration'])
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
        logger.info(f"duration value: {duration_val}")
        
        self.reward_values_dict['duration'].append(duration_val)
        sids = [rule['sid'] for rule in rule_total_energy_dict]
        alert_val = self.splunk_tools.get_alert_count(sids)
        self.reward_values_dict['alerts'].append(alert_val)
        logger.info(f"alert value: {alert_val}")
        
        energy_increase = 0
        duration_increase = 0
        # if self.average_energy == 0:
        #     energy_increase = 0
        # else:
        #     energy_increase = (energy_val - self.average_energy)/self.average_energy
        # if self.average_duration == 0:
        #     duration_increase = 0
        # else:
        #     duration_increase = (duration_val - self.average_duration)/self.average_duration
        # # self.previous_energy = energy_val        
        # logger.info(f"incease in energy: {energy_increase}")
        # logger.info(f"incease in duration: {duration_increase}")
        return alert_val, energy_val, energy_increase, duration_val, duration_increase
    

    def get_partial_reward_values(self, real_distribution, current_state):
        action_upper_bound = 1
        fraction_val = action_upper_bound - 1#float(self.get_fraction_state(current_state))
        distributions_val = self.compare_distributions(current_state[:len(self.relevant_logtypes)], current_state[len(self.relevant_logtypes):])     
        if distributions_val == 0:
            distributions_val = distributions_val + 0.000000000001
        self.reward_values_dict['distributions'].append(distributions_val)
        self.reward_values_dict['fraction'].append(fraction_val)
        logger.info(f"distributions value: {distributions_val}")
        logger.info(f"fraction value: {fraction_val}")
        return fraction_val,distributions_val
    
    def compare_distributions(self, dist1, dist2):#tool
        # Placeholder for your distribution comparison function
        # This could use a metric like KL divergence

        # complete distributions
        dist1 = np.append(dist1, [1-sum(dist1)])
        dist2 = np.append(dist2, [1-sum(dist2)])
        logger.info(f"dist1: {dist1}")
        logger.info(f"dist2: {dist2}")
        # return entropy(dist1, dist2)
        return jensenshannon(dist1, dist2)
        
        # return wasserstein_distance(dist1, dist2)