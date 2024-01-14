import datetime
import os
import subprocess
from time import sleep
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import pandas as pd

from measurement import Measurement
CPU_TDP=200
PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.92.1.el7.x86_64/Splunk Enterprise SIEM/Power Saver Plan/One Scan/'

class RewardCalc:
    def __init__(self, relevant_logtypes, dt_manager, logger, splunk_tools, rule_frequency, num_of_searches, distribution_learner=True):
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
        self.logger = logger
        self.splunk_tools = splunk_tools
        self.rule_frequency = rule_frequency
        self.time_rules_energy = []
        self.current_measurement_path = ''
        self.num_of_searches = num_of_searches
        self.distribution_learner = distribution_learner
        self.distribution_learner_counter = 0
        if not self.distribution_learner:
            self.distribution_learner_counter = 5    
        self.distribution_threshold = 0.2
        self.measurment_tool = Measurement(self.logger, self.splunk_tools, self.num_of_searches)
      
    def get_previous_full_reward(self):
        return self.reward_dict['alerts'][-1], self.reward_dict['energy'][-1]

    def get_partial_reward(self, real_distribution, current_state):
        fraction_val, distributions_val = self.get_partial_reward_values(real_distribution, current_state)
        # return 0.5/(distributions_val+0.000000000001) + 0.5*(fraction_val)
        return 1/(distributions_val)
        
    def get_is_limit_learner(self, current_state):
        return current_state[-2]
    
    def get_is_distribution_learner(self):
        return self.distribution_learner
    
    def get_log_type_index(self, current_state):
        return current_state[-4]
    
    def get_fraction_state(self, current_state):
        return current_state[-1]
    
    def get_full_reward(self, time_range, real_distribution, current_state):
        fraction_val, distributions_val = self.get_partial_reward_values(real_distribution, current_state)
        alert_val, energy_val, energy_increase, duration_val, duration_increase = self.get_full_reward_values(time_range=time_range)
        self.update_average_values()
        return duration_val/distributions_val
        
    def update_average_values(self):
        self.average_energy = sum(self.reward_values_dict['energy'])/len(self.reward_values_dict['energy'])
        self.average_alert = sum(self.reward_values_dict['alerts'])/len(self.reward_values_dict['alerts'])
        self.average_duration = sum(self.reward_values_dict['duration'])/len(self.reward_values_dict['duration'])
        self.logger.info(f"average energy: {self.average_energy}")
        self.logger.info(f"average alert: {self.average_alert}")
        self.logger.info(f"average duration: {self.average_duration}")
        
    def get_full_reward_values(self, time_range):
        self.logger.info('wait til next rule frequency')
        self.dt_manager.wait_til_next_rule_frequency(self.rule_frequency)
        rule_total_energy_dict = self.measurment_tool.measure(time_range=time_range, time_delta=self.rule_frequency)
                
        self.time_rules_energy.append({'time_range':str(time_range), 'rules':rule_total_energy_dict})
        energy_val = sum([rule['CPU(J)'] for rule in rule_total_energy_dict])
        duration_val = sum([rule['run_duration'] for rule in rule_total_energy_dict])
        
        self.logger.info(f"energy value: {energy_val}")
        self.logger.info(f"duration value: {duration_val}")
        self.reward_values_dict['energy'].append(energy_val)
        self.reward_values_dict['duration'].append(duration_val)
        alert_val = 0#self.splunk_tools.get_alert_count(time_range) #TODO: change to get_alert_count
        if self.average_energy == 0:
            energy_increase = 0
        else:
            energy_increase = (energy_val - self.average_energy)/self.average_energy
        if self.average_duration == 0:
            duration_increase = 0
        else:
            duration_increase = (duration_val - self.average_duration)/self.average_duration
        # self.previous_energy = energy_val        
        self.reward_values_dict['alerts'].append(alert_val)
        self.logger.info(f"incease in energy: {energy_increase}")
        self.logger.info(f"incease in duration: {duration_increase}")
        self.logger.info(f"alert value: {alert_val}")
        return alert_val, energy_val, energy_increase, duration_val, duration_increase
    

    def get_partial_reward_values(self, real_distribution, current_state):
        action_upper_bound = 1
        fraction_val = action_upper_bound - 1#float(self.get_fraction_state(current_state))
        distributions_val = self.compare_distributions(current_state[:len(self.relevant_logtypes)], current_state[len(self.relevant_logtypes):])     
        distributions_val = distributions_val + 0.000000000001
        self.reward_values_dict['distributions'].append(distributions_val)
        self.reward_values_dict['fraction'].append(fraction_val)
        self.logger.info(f"distributions value: {distributions_val}")
        # self.distribution_learner_control(distributions_val)
        self.logger.info(f"fraction value: {fraction_val}")
        return fraction_val,distributions_val
    
    def compare_distributions(self, dist1, dist2):#tool
        # Placeholder for your distribution comparison function
        # This could use a metric like KL divergence
        self.logger.info(f"dist1: {dist1}")
        self.logger.info(f"dist2: {dist2}")
        epsilon = 1e-10  # or any small positive value
        dist1 = [max(prob, epsilon) for prob in dist1]
        dist2 = [max(prob, epsilon) for prob in dist2]
        return entropy(dist1, dist2)
        
        # return wasserstein_distance(dist1, dist2)