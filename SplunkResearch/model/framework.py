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
    def __init__(self, log_generator_instance, splunk_tools_instance, dt_manager, time_range, rule_frequency, baseline, max_actions_value=10):
        self.current_measurement_path = None
        self.baseline = baseline
        self.log_generator = log_generator_instance
        self.splunk_tools = splunk_tools_instance
        self.dt_manager = dt_manager
        self.time_range = time_range
        self.rule_frequency = rule_frequency
        self.previous_energy = INFINITY  
        self.previous_alert = INFINITY
        self.previous_energy_reward_component = 0
        self.previous_alert_reward_component = 0
        self.previous_sum_of_action_values = 1
        self.action_space = spaces.Box(low=1/max_actions_value, high=1, shape=(1,), dtype=np.float64)
        self.max_actions_value = max_actions_value
        self.observation_space = spaces.Box(low=np.array([0]*((len(logtypes)+2))), high=np.array([INFINITY] * len(logtypes)+[len(logtypes)]+[1]))
        # self.current_time = self.time_range[0]
        self.current_action = None
        self.state = None  # Initialize state
        self.gamma = 1/2
        self.beta = 1/4
        self.delta = 1/8
        self.alpha = 1/8
        self.logtype_index = 0
        self.sum_of_action_values = 0
        self.current_step = 0
        self.max_steps = 100
        self.measure_thread = None
        self.time_left_till_next_measurement = None
        self.reward_dict = {'energy': [], 'alerts': [], 'distributions': [], 'fraction': [], 'total': []}
        self.reward_values_dict = {'energy': [], 'alerts': [], 'distributions': [], 'fraction': []}
        self.fake_distribution = [0]*len(logtypes)
        self.real_distribution = [0]*len(logtypes)
        self.done = False
        
    def logtype_index_counter(self):
        self.logtype_index += 1
        if self.logtype_index == len(logtypes):
            self.logtype_index = 0           
                                           
    def calculate_energy_reward(self, current_energy):
        # TODO: checke it
        energy_reward =  (current_energy - self.previous_energy)/(self.previous_energy + 1/1000000)
        self.previous_energy_reward_component = energy_reward
        self.previous_energy = current_energy
        return energy_reward
        
    def calculate_alert_reward(self):
        alert_count = self.splunk_tools.get_alert_count(self.time_range)
        self.reward_values_dict['alerts'].append(alert_count)
        # alert_reward = (alert_count - self.previous_alert)/(self.previous_alert + 1/1000000)
        # self.previous_alert_reward_component = -alert_reward
        # self.previous_alert = alert_count
        if alert_count:
            alert_reward = 1
        else:
            alert_reward = 0
        return -alert_reward
  
    def get_energy_reward_component(self):
        # if not self.done:
        #         return self.previous_energy_reward_component
        # self.measure_thread.join()
        current_energy = self.measure(self.rule_frequency)
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
        # TODO: Change to percentage of logs 
        # epsilon = 0.0000001
        # real_distribution = self.state[:len(logtypes)]
        # for i in range(len(real_distribution)):
        #     real_distribution[i] += epsilon
        # fake_distribution = real_distribution + self.fake_distribution

        # TODO: should the right distribution be the current distribution or the current with out the fake logs?
        distributions_distance = self.compare_distributions(self.real_distribution, self.fake_distribution)
        self.reward_values_dict['distributions'].append(distributions_distance)
        return -distributions_distance
    
    def get_fraction_reward_component(self):    
        if self.sum_of_fractions > 1:
            # fraction_reward = -(self.sum_of_action_values-1)/(self.sum_of_action_values)
            fraction_reward = -1
        else:
            # fraction_reward = self.sum_of_action_values
            fraction_reward = 0
        self.reward_values_dict['fraction'].append(self.sum_of_fractions)
        # self.previous_sum_of_action_values = self.sum_of_action_values
        return fraction_reward
                
    def get_reward_componnents(self):
        fraction_reward = self.get_fraction_reward_component()
        alerts_reward = self.get_alerts_reward_component()
        distributions_reward = self.get_distributions_reward_component()
        energy_reward = self.get_energy_reward_component()        
        return energy_reward, alerts_reward, distributions_reward, fraction_reward
    
    
    
    # def update_current_time(self):
    #     while True:
    #         self.calculate_time_to_next_step()
    #         self.current_time = self.change_time(self.time_range[1], seconds=-self.time_left_till_next_measurement)
    #         time.sleep(1)
    #         if self.measure_thread is None:
    #             return 
    # def get_fake_distribution(self): #tool
    #     fake_distribution = self.state[:len(logtypes)].copy()
    #     # self.dt_manager.log(f"current distribution: {self.state[:len(logtypes)]}")
    #     fake_distribution[self.logtype_index] += int(self.current_action*self.max_actions_value)
    #     # self.dt_manager.log(f"fake distribution: {fake_distribution}")
    #     return fake_distribution
    
    # def calculate_time_to_next_step(self): #tool
    #     now = self.dt_manager.get_current_datetime()
    #     datetime_now = datetime.datetime.strptime(now, '%m/%d/%Y:%H:%M:%S')
    #     seconds_since_hour = (datetime_now.minute * 60) + datetime_now.second
    #     seconds_into_step = seconds_since_hour % (self.rule_frequency * 60)
    #     seconds_until_next_step = (self.rule_frequency * 60) - seconds_into_step
    #     self.time_left_till_next_measurement = seconds_until_next_step
        
    # def wait_to_measure(self): #tool
    #     self.calculate_time_to_next_step()
    #     # Display a countdown timer while we wait
    #     for i in range(seconds_until_next_step-20, 0, -1):
    #         time.sleep(1)
        
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
        current_energy = grouped_rules_enegry_df['CPU(J)'].sum()
        return current_energy
    
    # def change_time(self, original_time, hours=0, minutes=0, seconds=0):
    #     return (pd.to_datetime(original_time,format='%m/%d/%Y:%H:%M:%S') + datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)).strftime('%m/%d/%Y:%H:%M:%S')
    
    def after_measure_clean(self): #tool
        # update time range of rules with rule frequency
        minutes = int(self.time_range[0].split(':')[2])
        seconds = int(self.time_range[0].split(':')[3])
        self.dt_manager.log(f'current time_range: {self.time_range}')
        start_time = self.dt_manager.subtract_time(self.time_range[0], minutes=(minutes%self.rule_frequency), seconds=seconds%60)
        self.dt_manager.log(f'update start_time to {start_time}')
        start_time = self.dt_manager.add_time(start_time, minutes=self.rule_frequency)
        self.dt_manager.log(f'update start_time to {start_time}')
        end_time = self.dt_manager.add_time(start_time, minutes=self.rule_frequency)
        self.dt_manager.log(f'update end_time to {end_time}')
        self.time_range = (start_time, end_time)
        self.dt_manager.log(f'update time_range to {self.time_range}')
        self.splunk_tools.update_all_searches(self.splunk_tools.update_search_time_range, self.time_range) 
        self.fake_distribution = [0]*len(logtypes)  
        self.current_measurement_path = None
        self.measure_thread = None
    
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
            reward = distributions_reward
        self.reward_dict['distributions'].append(distributions_reward)
        self.reward_dict['total'].append(reward)
        self.dt_manager.log(f"distributions reward: {distributions_reward}")
        self.dt_manager.log(f"total reward: {reward}")
        return reward

    def step(self, action):
        print(f"current time: {self.dt_manager.get_current_datetime()}")      
        self.perform_action(action)
        self.update_state()
        # self.calculate_time_to_next_step()
        if self.logtype_index == len(logtypes)-1:
            self.dt_manager.log(f"waiting for next measurement")
            self.dt_manager.wait_til_next_rule_frequency(self.rule_frequency)
            self.done = True
        reward = self.get_reward()
        self.logtype_index_counter()
        self.dt_manager.log(f"########################################################################################################################")
        return self.state, reward, self.done, {}

    def before_first_step(self):
        self.dt_manager.log("waiting for first measurement")
        # self.measure_thread.join()
        first_alert_reward = self.get_alerts_reward_component()
        first_energy_reward = self.get_energy_reward_component()
        self.dt_manager.log(f"first energy reward: {first_energy_reward}")
        self.dt_manager.log(f"first alert reward: {first_alert_reward}")
        # self.reset() # no need because that the model starts with reset
    
    def perform_action(self, action):
        # calculate the current time range according to the time left till the next measurement
        now = self.dt_manager.get_current_datetime()
        time_range = (now, self.dt_manager.add_time(now, seconds=1))
        self.current_action = action[0]
        logtype = logtypes[self.logtype_index]
        logsource = logtype[0].lower()
        eventcode = logtype[1]
        action_value = int(self.current_action*self.max_actions_value)
        self.dt_manager.log(f"action: {self.current_action}, action value: {action_value}, logtype: {logtype}")
        fake_logs = self.log_generator.generate_logs(logsource, eventcode, time_range, action_value)
        self.dt_manager.log(f"{len(fake_logs)}: fake logs were generated")
        self.sum_of_action_values += action_value
        self.sum_of_fractions += self.current_action
        self.fake_distribution[self.logtype_index] += action_value
        self.splunk_tools.insert_logs(fake_logs, logsource)
    
    def check_done(self):
        # Define the termination conditions based on the current state or other criteria
        if self.current_step >= self.max_steps:
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
        delete_response = self.splunk_tools.delete_fake_logs(time_range)
        self.dt_manager.log(delete_response)
        
        self.after_measure_clean()
        return self.state  # reward, done, info can't be included
    
    def render(self, mode='human'):
        self.dt_manager.log(f"Current state: {self.state}")

        
    def update_state(self):
        state = []
        real_distribution = self.splunk_tools.extract_distribution(self.time_range[0], self.dt_manager.get_current_datetime())
        self.dt_manager.log(f"extraceted real {real_distribution}")
        fake_distribution = self.splunk_tools.extract_distribution(self.time_range[0], self.dt_manager.get_current_datetime(), fake=True)
        self.dt_manager.log(f"extraceted fake {fake_distribution}")
        # for logtype in logtypes:
        #     logtype = f"{logtype[0].lower()} {logtype[1]}"
        #     if logtype in real_distribution and logtype in fake_distribution:
        #         state.append(real_distribution[logtype] + fake_distribution[logtype])
        #     elif logtype in fake_distribution:
        #         state.append(fake_distribution[logtype])
        #     elif logtype in real_distribution:
        #         state.append(real_distribution[logtype])
        #     else:
        #         state.append(0)
        epsilon = 0.0000001
        self.fake_distribution = [fake_distribution[f"{logtype[0].lower()} {logtype[1]}"]+epsilon if f"{logtype[0].lower()} {logtype[1]}" in fake_distribution else epsilon for logtype in logtypes]
        self.real_distribution = [real_distribution[f"{logtype[0].lower()} {logtype[1]}"]+epsilon if f"{logtype[0].lower()} {logtype[1]}" in real_distribution else epsilon for logtype in logtypes]
        self.dt_manager.log(f"real distribution: {self.real_distribution}")
        self.dt_manager.log(f"fake distribution: {self.fake_distribution}")
        state = [x + y for x, y in zip(self.fake_distribution, self.real_distribution)]
        state.append(self.logtype_index)
        state.append(self.sum_of_fractions)
        self.dt_manager.log(f"state: {state}")
        self.state = np.array(state)