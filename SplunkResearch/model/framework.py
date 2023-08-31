import datetime
import subprocess
from threading import Thread, Timer
import time
import numpy as np
import pandas as pd
import sys
import urllib3
import logging
from scipy.stats import entropy


sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment')
import os
from dotenv import load_dotenv
load_dotenv('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/.env')
urllib3.disable_warnings()
from log_generator import LogGenerator
import gym
from gym import spaces
from logtypes import logtypes

# BUG: Splunk doesnt parse all the logs
PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.88.1.el7.x86_64/Splunk Enterprise SIEM/Power Saver Plan/One Scan/'
INFINITY = 100000
CPU_TDP = 200
class Framework(gym.Env):
    def __init__(self, log_generator_instance, splunk_tools_instance, dt_manager, time_range, rule_frequency, reward_parameter_dict, max_actions_value=10):
        self.current_measurement_path = None
        self.log_generator = log_generator_instance
        self.splunk_tools = splunk_tools_instance
        self.dt_manager = dt_manager
        self.time_range = time_range
        self.rule_frequency = rule_frequency
        self.previous_energy = INFINITY  
        self.previous_alert = INFINITY
        self.previous_energy_reward_component = 1
        self.previous_alert_reward_component = 1
        self.previous_fraction_reward_component = 1
        self.previous_sum_of_action_values = 1
        self.action_upper_bound = 1
        self.action_space = spaces.Box(low=1/max_actions_value, high=self.action_upper_bound, shape=(1,), dtype=np.float64)
        self.max_actions_value = max_actions_value
        self.observation_space = spaces.Box(low=np.array([0]*((len(logtypes)+2))), high=np.array([INFINITY] * len(logtypes)+[len(logtypes)]+[1]))
        self.action_duration = self.rule_frequency*60/len(logtypes)
        self.current_action = None
        self.state = None  # Initialize state
        self.gamma = reward_parameter_dict['gamma']
        self.beta = reward_parameter_dict['beta']
        self.delta = reward_parameter_dict['delta']
        self.alpha = reward_parameter_dict['alpha']
        self.logtype_index = 0
        self.sum_of_action_values = 0
        self.sum_of_fractions = 0
        self.time_action_dict= {}
        self.time_rules_energy_dict = {}
        self.current_step = 0
        self.max_steps = 100
        self.time_left_till_next_measurement = None
        self.reward_dict = {'energy': [], 'alerts': [], 'distributions': [], 'fraction': [], 'total': []}
        self.reward_values_dict = {'energy': [], 'alerts': [], 'distributions': [], 'fraction': []}
        self.fake_distribution = [0]*len(logtypes)
        self.real_distribution = [0]*len(logtypes)
        self.done = False
        self.epsilon = 0.0000001
        # self.distribution_reward_normal = self.compare_distributions([self.epsilon]*len(logtypes), [self.max_actions_value]*len(logtypes))
        
    def logtype_index_counter(self):
        self.logtype_index += 1
        if self.done:
            self.logtype_index = 0           
                                           
    def calculate_energy_reward(self, current_energy):
        # TODO: checke it
        # if self.previous_energy < current_energy:
        #     energy_reward = 0
        # elif self.previous_energy >= current_energy:
        #     energy_reward = -1
        # else:
        #     energy_reward = 0
        energy_reward = (current_energy - self.previous_energy)/(self.previous_energy+self.epsilon)
        self.previous_energy_reward_component = energy_reward
        self.previous_energy = current_energy
        return min(energy_reward, 1)
        
    def calculate_alert_reward(self):
        alert_count = self.splunk_tools.get_alert_count(self.time_range)
        self.reward_values_dict['alerts'].append(alert_count)
        # alert_reward = (alert_count - self.previous_alert)/(self.previous_alert + 1/1000000)
        # self.previous_alert_reward_component = -alert_reward
        # self.previous_alert = alert_count
        if alert_count:
            alert_reward = 0
        else:
            alert_reward = 1
        return alert_reward
  
    def get_energy_reward_component(self):
        # if not self.done:
        #         return self.previous_energy_reward_component
        current_energy = self.measure(self.rule_frequency)
        self.dt_manager.log(f"current energy: {current_energy}")
        self.reward_values_dict['energy'].append(current_energy)
        energy_reward = self.calculate_energy_reward(current_energy)
        self.dt_manager.log(f"dir name of last measurement: {self.current_measurement_path.split('/')[-1]}")
        # self.after_measure_clean()
        return energy_reward

    def get_alerts_reward_component(self):
        # if not self.done:
        #     return self.previous_alert_reward_component
        alert_reward = self.calculate_alert_reward()
        return alert_reward
   
    def get_distributions_reward_component(self):
        accetpable_distance = 0.1
        # TODO: Change to percentage of logs 
        # TODO: should the right distribution be the current distribution or the current with out the fake logs?
        distributions_distance = self.compare_distributions(self.real_distribution, self.state[:len(logtypes)])
        self.dt_manager.log(f'distributions distance: {distributions_distance}')
        self.reward_values_dict['distributions'].append(distributions_distance)
        # if distributions_distance < accetpable_distance:
        #     return 1
        # elif distributions_distance > accetpable_distance:
        #     return -1
        # else:
        #     return 0
        return 1-1*min(distributions_distance, 1)
    
    def get_fraction_reward_component(self):    
        if self.sum_of_fractions > 1 and self.logtype_index <= (len(logtypes)-1):
            fraction_reward = -1000*self.sum_of_fractions
            # fraction_reward = -(self.sum_of_fractions-1)/(self.sum_of_fractions)
            # fraction_reward = -1
        elif self.sum_of_fractions == 1 and self.logtype_index == (len(logtypes)-1):
            fraction_reward = 1
        else:
            fraction_reward = -(self.sum_of_fractions-1)       
        self.reward_values_dict['fraction'].append(self.sum_of_fractions)
        self.previous_fraction_reward_component = fraction_reward
        return fraction_reward
                
    def get_reward_componnents(self):
        fraction_reward = self.get_fraction_reward_component()
        alerts_reward = self.get_alerts_reward_component()
        distributions_reward = self.get_distributions_reward_component()
        energy_reward = self.get_energy_reward_component()        
        return energy_reward, alerts_reward, distributions_reward, fraction_reward
    
    

    def measure(self, time_delta=60): #tool
        # self.measurement()
        # run the measurement script with subprocess
        self.dt_manager.log('measuring')
        cmd = subprocess.run('python ../Scanner/scanner.py', shell=True, capture_output=True, text=True)
        self.dt_manager.log(cmd.stdout)
        self.dt_manager.log(cmd.stderr)
        # find the latest measurement folder
        measurement_num = max([int(folder.split(' ')[1]) for folder in os.listdir(PATH) if folder.startswith('Measurement')])
        self.current_measurement_path = os.path.join(PATH,f'Measurement {measurement_num}')
        self.dt_manager.log(self.current_measurement_path)
        
        pids_energy_df = pd.read_csv(os.path.join(self.current_measurement_path, 'processes_data.csv'))
        pids_energy_df['Time(sec)'] = pd.to_datetime(pids_energy_df['Time(sec)'], unit='s').dt.to_pydatetime()
        rules_pids = self.splunk_tools.get_rules_pids(time_delta)
        rules_energy_df = self.filter_rules_energy(rules_pids, pids_energy_df)
        grouped_rules_enegry_df = self.aggregate_rules_energy(rules_energy_df)
        self.time_rules_energy_dict[str(self.time_range)] = grouped_rules_enegry_df[['name', 'CPU(J)']].to_dict('records')
        current_energy = grouped_rules_enegry_df['CPU(J)'].sum()
        return current_energy
    
  
    
    def aggregate_rules_energy(self, rules_energy_df):
        time_field = 'Time(sec)'
        # create a new column for the time interval
        rules_energy_df[time_field] = pd.to_datetime(rules_energy_df[time_field])
        rules_energy_df = rules_energy_df.sort_values(by=['name', time_field])
        rules_energy_df['delta_time'] = rules_energy_df.groupby('name')[time_field].diff().dt.total_seconds().fillna(0)
        rules_energy_df['CPU(W)'] = rules_energy_df['CPU(%)'] * CPU_TDP / 100
        rules_energy_df['CPU(J)'] = rules_energy_df['CPU(W)'] * rules_energy_df['delta_time']
        grouped_rules_enegry_df = rules_energy_df.groupby('name')['CPU(J)'].sum().reset_index()
        grouped_rules_enegry_df.to_csv(os.path.join(self.current_measurement_path, 'grouped_rules_energy.csv'))
        return grouped_rules_enegry_df
    
    def filter_rules_energy(self, rules_pids, pids_energy_df): #tool
        data = []
        for name, rules in rules_pids.items():
            for e in rules:
                sid, pid, time, run_duration = e[:4]
                data.append((name, sid, pid, time, run_duration)) 
        rules_pids_df = pd.DataFrame(data, columns=['name', 'sid', 'pid', 'time', 'run_duration'])
        rules_pids_df.time = pd.to_datetime(rules_pids_df.time)
        rules_pids_df.sort_values('time', inplace=True)
        splunk_pids_energy_df = pids_energy_df[pids_energy_df['PID'].isin(rules_pids_df.pid.values)].sort_values('Time(sec)') 
        rules_energy_df = pd.merge(splunk_pids_energy_df, rules_pids_df, left_on='PID', right_on='pid')
        rules_energy_df.to_csv(os.path.join(self.current_measurement_path, 'rules_energy.csv'))
        return rules_energy_df

             
    def compare_distributions(self, dist1, dist2):#tool
        # Placeholder for your distribution comparison function
        # This could use a metric like KL divergence
        return entropy(dist1, dist2)
    
    
    def get_reward(self):
        if self.done:
            energy_reward, alerts_reward, distributions_reward, fraction_reward = self.get_reward_componnents()
            reward = self.alpha  * energy_reward + self.beta *fraction_reward  + self.gamma*distributions_reward  + self.delta * alerts_reward
            self.reward_dict['energy'].append(energy_reward)
            self.reward_dict['alerts'].append(alerts_reward)
            self.reward_dict['fraction'].append(fraction_reward)
            self.dt_manager.log(f"energy reward: {energy_reward}")
            self.dt_manager.log(f"fraction reward: {fraction_reward}")
            self.dt_manager.log(f"alerts reward: {alerts_reward}")
        else:
            distributions_reward = self.get_distributions_reward_component()
            reward = self.alpha  * self.previous_energy_reward_component + self.beta *self.previous_fraction_reward_component + self.gamma*distributions_reward  + self.delta * self.previous_alert_reward_component
        self.reward_dict['distributions'].append(distributions_reward)
        self.reward_dict['total'].append(reward)
        self.dt_manager.log(f"distributions reward: {distributions_reward}")
        self.dt_manager.log(f"total reward: {reward}")
        return reward

    def step(self, action):
        self.dt_manager.log(f"Current time: {self.dt_manager.get_fake_current_datetime()}")      
        self.perform_action(action)
        self.update_state()
        # self.calculate_time_to_next_step()
        if self.check_done():
            self.dt_manager.round_to_next_rule_frequency(self.rule_frequency)
            self.dt_manager.wait_til_next_rule_frequency(self.rule_frequency)
            self.done = True
        reward = self.get_reward()
        self.logtype_index_counter()
        self.dt_manager.log(f"########################################################################################################################")
        return self.state, reward, self.done, {}

    def before_first_step(self):
        self.dt_manager.log("waiting for first measurement")
        first_alert_reward = self.get_alerts_reward_component()
        self.dt_manager.wait_til_next_rule_frequency(self.rule_frequency)
        self.previous_energy = self.measure(self.rule_frequency)
        self.dt_manager.log(f'first energy: {self.previous_energy}')
        self.dt_manager.log(f"first alert reward: {first_alert_reward}")
        # self.reset() # no need because that the model starts with reset
    def update_action_space(self):
        self.action_space = spaces.Box(low=1/self.max_actions_value, high=self.action_upper_bound, shape=(1,), dtype=np.float64)
    
    def perform_action(self, action):
        # calculate the current time range according to the time left till the next measurement
        now = self.dt_manager.get_fake_current_datetime()
        time_range = (now, self.dt_manager.add_time(now, seconds=self.action_duration))
        self.current_action = action[0]
        logtype = logtypes[self.logtype_index]
        logsource = logtype[0].lower()
        eventcode = logtype[1]
        action_value = int(self.current_action*self.max_actions_value)
        self.time_action_dict[str(self.time_range)][str(logtype)] = action_value
        self.dt_manager.log(f"action: {self.current_action}, action value: {action_value}, logtype: {logtype}")
        fake_logs = self.log_generator.generate_logs(logsource, eventcode, time_range, action_value)
        self.dt_manager.log(f"{len(fake_logs)}: fake logs were generated")
        self.sum_of_action_values += action_value
        self.sum_of_fractions += self.current_action
        self.fake_distribution[self.logtype_index] += action_value
        self.splunk_tools.insert_logs(fake_logs, logsource)
        new_current_fake_time = self.dt_manager.add_time(now, seconds=self.action_duration)
        self.dt_manager.set_fake_current_datetime(new_current_fake_time)
    
    def check_done(self):
        # Define the termination conditions based on the current state or other criteria
        if self.logtype_index == (len(logtypes)-1): #or self.sum_of_fractions >= 1:
            return True
        else:
            return False
        
    def reset(self):
        self.dt_manager.log("resetting")
        self.current_log_type = 0
        self.sum_of_fractions = 0
        self.done = False
        self.sum_of_action_values = 0
        # Reset the environment to an initial state
        self.update_state()
        # define time range to delete
        date = self.time_range[1].split(':')[0]
        time_range = (f'{date}:00:00:00', f'{date}:23:59:59')
        self.splunk_tools.delete_fake_logs(time_range)    
        new_start_time = self.dt_manager.get_fake_current_datetime()
        new_end_time = self.dt_manager.add_time(new_start_time, minutes=self.rule_frequency)

        self.dt_manager.log(f'current time_range: {self.time_range}')
        self.dt_manager.log(f'update start_time to {new_start_time}')
        self.dt_manager.log(f'update end_time to {new_end_time}')
        self.time_range = (new_start_time, new_end_time)
        self.time_action_dict[str(self.time_range)] = {}
        self.dt_manager.log(f'update time_range to {self.time_range}')        
        self.splunk_tools.update_all_searches(self.splunk_tools.update_search_time_range, self.time_range) 
        self.fake_distribution = [0]*len(logtypes)  
        self.current_measurement_path = None     
        
        return self.state  # reward, done, info can't be included
    
    def render(self, mode='human'):
        self.dt_manager.log(f"Current state: {self.state}")

        
    def update_state(self):
        state = []
        real_distribution = self.splunk_tools.extract_distribution(self.time_range[0], self.dt_manager.get_fake_current_datetime())
        self.dt_manager.log(f"extraceted real {real_distribution}")
        # fake_distribution = self.splunk_tools.extract_distribution(self.time_range[0], self.dt_manager.get_fake_current_datetime(), fake=True)
        # self.dt_manager.log(f"extraceted fake {fake_distribution}")
        self.fake_distribution = [self.fake_distribution[i] if self.fake_distribution[i] else self.epsilon for i in range(len(logtypes)) ]
        self.real_distribution = [real_distribution[f"{logtype[0].lower()} {logtype[1]}"] if f"{logtype[0].lower()} {logtype[1]}" in real_distribution else self.epsilon for logtype in logtypes]
        self.dt_manager.log(f"real distribution: {self.real_distribution}")
        self.dt_manager.log(f"fake distribution: {self.fake_distribution}")
        state = [x + y for x, y in zip(self.fake_distribution, self.real_distribution)]
        sum_state = sum(state)
        state = [x/sum_state for x in state]
        state.append(self.logtype_index)
        state.append(self.sum_of_fractions)
        self.dt_manager.log(f"state: {state}")
        self.state = np.array(state)
        
if __name__=="__main__":
    # test the action performing
    from SplunkResearch.model.utils import MockedDatetimeManager
    from SplunkResearch.splunk_tools import SplunkTools
    from config import replacement_dicts as big_replacement_dicts

    fake_start_datetime = datetime.datetime(2023,6,22, 8, 0, 0)
    rule_frequency = 3
    time_range = ('06/22/2023:08:00:00', '06/22/2023:08:03:00')
    current_dir = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/tests'
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"test_{timestamp}"
    dt_manager = MockedDatetimeManager(fake_start_datetime=fake_start_datetime, log_file_path=f"{current_dir}/{log_file_name}.txt")
    splunk_tools_instance = SplunkTools(dt_manager=dt_manager)
    
    log_generator_instance = LogGenerator(logtypes, big_replacement_dicts, splunk_tools_instance)
    env = Framework(log_generator_instance, splunk_tools_instance, dt_manager, time_range, rule_frequency)
    for i in enumerate(logtypes):
        env.perform_action([1])
        env.update_state()
        env.logtype_index_counter()
        

        
    