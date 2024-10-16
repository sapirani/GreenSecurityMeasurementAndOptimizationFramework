from abc import ABC, abstractmethod
import datetime
import json
import os
import subprocess
from time import sleep
import numpy as np
from scipy import stats
from scipy.stats import entropy, wasserstein_distance
import pandas as pd
from scipy.spatial.distance import jensenshannon
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)
CPU_TDP = 200
PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.92.1.el7.x86_64/Splunk Enterprise SIEM/Power Saver Plan/One Scan/'



class RewardStrategy(ABC):
    def __init__(self,  dt_manager, splunk_tools, num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        self.average_energy = 0
        self.average_alert = 0
        self.average_duration = 0
        self.epsilon = 0
        self.action_upper_bound = 1
        self.reward_dict = {'energy': [], 'alerts': [], 'distributions': [], 'duration': [], 'total': []}
        self.reward_values_dict = {'energy': [], 'alerts': [], 'distributions': [], 'duration': [], "num_of_rules":[], "p_values":[], "t_values":[], "degrees_of_freedom":[]}
        self.dt_manager = dt_manager
        self.splunk_tools = splunk_tools
        self.rule_frequency = self.splunk_tools.rule_frequency
        self.time_rules_energy = []
        self.current_measurement_path = ''
        self.num_of_searches = num_of_searches  
        self.distribution_threshold = 0.3
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
        

    
    def get_no_agent_reward(self, time_range):
        relevant_row = self.no_agent_values[(self.no_agent_values['start_time'] == time_range[0]) & (self.no_agent_values['end_time'] == time_range[1])]
        if not relevant_row.empty:
            alert_val = sum([relevant_row[col].values[0] for col in relevant_row.columns if col.startswith('rule_alert_')])
            duration_val = sum([relevant_row[col].values[0] for col in relevant_row.columns if col.startswith('rule_duration_')])
            std_duration_val = sum([relevant_row[col].values[0] for col in relevant_row.columns if col.startswith('rule_std_duration_')])
        else:
            logger.info('Measure no agent reward values')
            alert_vals, duration_vals, std_duration_vals, saved_searches = self.splunk_tools.run_saved_searches(time_range)
            alert_val = sum(alert_vals)
            duration_val = sum(duration_vals)
            std_duration_val = sum(std_duration_vals)
            self.no_agent_values = pd.concat([self.no_agent_values, pd.DataFrame({'start_time':[time_range[0]],
                                                                                  'end_time':[time_range[1]],
                                                                                  **{f"rule_alert_{saved_searches[i]}":[alert_vals[i]] for i in range(len(saved_searches))},
                                                                                  **{f"rule_duration_{saved_searches[i]}":[duration_vals[i]] for i in range(len(saved_searches))},
                                                                                  **{f"rule_std_duration_{saved_searches[i]}": [std_duration_vals[i]] for i in range(len(saved_searches))},
                                                                                  'alert_values':[alert_val], 'duration_values':[duration_val], 'std_duration_values':[std_duration_val]})])
            random_val = np.random.randint(0, 10)
            if random_val % 3 == 0:
                self.no_agent_values.to_csv(self.no_agent_table_path, index=False)
            relevant_row = self.no_agent_values[(self.no_agent_values['start_time'] == time_range[0]) & (self.no_agent_values['end_time'] == time_range[1])]
        self.no_agent_last_row = relevant_row
        logger.info(f"no agent alert value: {alert_val}")
        logger.info(f"no agent duration value: {duration_val}")
        return alert_val, duration_val, std_duration_val
        
        
    @abstractmethod
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        pass
    @abstractmethod
    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        pass
    
    
    
    def get_partial_reward_values(self, real_distribution, fake_distribution):
        distributions_val = self.compare_distributions(real_distribution, fake_distribution)     
        if distributions_val == 0:
            distributions_val = distributions_val + 0.000000000001
        self.reward_values_dict['distributions'].append(distributions_val)
        logger.info(f"distributions value: {distributions_val}")
        return distributions_val
    
    def compare_distributions(self, dist1, dist2):
        logger.info(f"dist1: {dist1}")
        logger.info(f"dist2: {dist2}")
        # return wasserstein_distance(dist1, dist2)
        # return entropy(dist1, dist2)
        return jensenshannon(dist1, dist2)#**2
    
    def get_duration_reward_values(self, time_range):
        alert_vals, duration_vals, std_duration_vals, saved_searches = self.splunk_tools.run_saved_searches(time_range)
        alert_val = sum(alert_vals)
        duration_val = sum(duration_vals)
        std_duration_val = sum(std_duration_vals)
        self.time_rules_energy.append({'time_range':str(time_range),
                                        **{f"rule_duration_{saved_searches[i]}":duration_vals[i] for i in range(len(saved_searches))}, 
                                        **{f"rule_alert_{saved_searches[i]}":alert_vals[i] for i in range(len(saved_searches))}, 
                                        **{f"rule_std_duration_{saved_searches[i]}":std_duration_vals[i] for i in range(len(saved_searches))}})
        
        return alert_val, duration_val, std_duration_val
    
    def get_hardware_reward_values(self, time_range):
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



class RewardStrategy1(RewardStrategy):
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)        
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)

    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)

    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)

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
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        partial_reward = self.get_partial_reward(real_distribution, fake_distribution, current_action)
        if partial_reward:
            return partial_reward
        return super().get_full_reward(time_range, real_distribution, fake_distribution, current_action)

class RewardStrategy13(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        distributions_distance = self.get_partial_reward_values(real_distribution, fake_distribution)
        distributions_reward = - distributions_distance
        self.reward_dict['distributions'].append(distributions_reward)
        return distributions_reward
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        action_variance = np.var(current_action)
        return action_variance
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        distributions_distance = self.get_partial_reward_values(real_distribution, fake_distribution)
        distributions_reward = - distributions_distance
        self.reward_dict['distributions'].append(distributions_reward)
        return distributions_reward
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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
        duration_reward = 100000000000**(-p_value)
        self.reward_dict['duration'].append(duration_reward)
        logger.info(f"t: {t}")
        logger.info(f"p_value: {p_value}")
        logger.info(f"degree_of_freedom: {degree_of_freedom}")
        return duration_reward
    
    
class RewardStrategy20(RewardStrategy):
    
    def __init__(self,  dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=None):
        super().__init__( dt_manager, splunk_tools,  num_of_searches, measurment_tool, alpha, beta, gamma,  no_agent_table_path=no_agent_table_path)


    def get_partial_reward(self, real_distribution, fake_distribution, current_action):
        return 0
    
    def get_full_reward(self, time_range, real_distribution, fake_distribution, current_action):
        alert_val, duration_val, std_duration_val = self.get_duration_reward_values(time_range)
        no_agent_alert_val, no_agent_duration_val, no_agent_std_duration_val = self.get_no_agent_reward(time_range)
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