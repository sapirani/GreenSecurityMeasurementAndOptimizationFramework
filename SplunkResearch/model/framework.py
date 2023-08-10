import datetime
import subprocess
from threading import Thread
import time
import numpy as np
import pandas as pd
import sys
import urllib3
import logging
from scipy.stats import entropy
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
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
    def __init__(self, log_generator_instance, splunk_tools_instance, time_range, rule_frequency, baseline, max_actions_value=10):
        self.current_measurement_path = None
        self.baseline = baseline
        self.log_generator = log_generator_instance
        self.splunk_tools = splunk_tools_instance
        self.time_range = time_range
        self.rule_frequency = rule_frequency
        self.previous_energy = None
        self.previous_energy = INFINITY  
        self.previous_alert = INFINITY
        self.previous_energy_reward_component = 0
        self.previous_alert_reward_component = 0
        self.previous_sum_of_action_values = 1
        self.action_space = spaces.Box(low=1/max_actions_value, high=1, shape=(1,), dtype=np.float64)
        self.max_actions_value = max_actions_value
        self.observation_space = spaces.Box(low=np.array([0]*((len(logtypes)+2))), high=np.array([INFINITY] * len(logtypes)+[len(logtypes)]+[1]))
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
        self.waiting_thread = None
        self.reward_dict = {'energy': 0, 'alerts': 0, 'distributions': 0, 'fraction': 0, 'total': 0}
    
    def logtype_index_counter(self):
        self.logtype_index += 1
        if self.logtype_index == len(logtypes):
            self.logtype_index = 0           
                                           
    def reset(self):
        logging.info("resetting")
        self.current_log_type = 0
        self.sum_of_fractions = 0
        # Reset the environment to an initial state
        self.update_state()
        # define time range to delete
        date = self.time_range[1].split(':')[0]
        time_range = (f'{date}:00:00:00', f'{date}:23:59:59')
        delete_response = self.splunk_tools.delete_fake_logs(time_range)
        logging.info(delete_response)
        return self.state  # reward, done, info can't be included
    
    def render(self, mode='human'):
        logging.info(f"Current state: {self.state}")

        
    def update_state(self):
        state = []
        current_distribution = self.splunk_tools.extract_distribution(*self.time_range)
        for logtype in logtypes:
            logtype = f"{logtype[0].lower()} {logtype[1]}"
            if logtype in current_distribution:
                state.append(current_distribution[logtype])
            else:
                state.append(0)
        state.append(self.logtype_index)
        state.append(self.sum_of_action_values)
        logging.info(f"state: {state}")
        self.state = np.array(state)
        

  
    def get_energy_reward_component(self):
        if self.waiting_thread.is_alive() and self.logtype_index < len(logtypes)-1:
                return self.previous_energy_reward_component
        self.waiting_thread.join()
        current_energy = self.measure(self.rule_frequency)
        energy_reward = self.calculate_energy_reward(current_energy)
        self.after_measure_clean()
        return energy_reward

    def calculate_energy_reward(self, current_energy):
        energy_reward =  (current_energy - self.previous_energy)/(self.previous_energy + 1/1000000)
        self.previous_energy_reward_component = energy_reward
        self.previous_energy = current_energy
        return energy_reward
        
    def get_alerts_reward_component(self):
        if self.waiting_thread.is_alive() and self.logtype_index < len(logtypes): # continue to learn till the next measurement
            return self.previous_alert_reward_component
        alert_reward = self.calculate_alert_reward()
        return alert_reward

    def calculate_alert_reward(self):
        alert_count = self.splunk_tools.get_alert_count(self.time_range)
        alert_reward = (alert_count - self.previous_alert)/(self.previous_alert + 1/1000000)
        self.previous_alert_reward_component = -alert_reward
        self.previous_alert = alert_count
        return -alert_reward
    
    def get_distributions_reward_component(self):
        fake_distribution = self.get_fake_distribution()
        # TODO: should the right distribution be the current distribution or the current with out the fake logs?
        distributions_distance = self.compare_distributions(self.state[:len(logtypes)], fake_distribution)
        return -distributions_distance
    
    def get_fraction_reward_component(self):
        if self.logtype_index < len(logtypes)-1: # continue to learn till the next measurement
            return 0        
        fraction_reward = (self.previous_sum_of_action_values - self.sum_of_action_values)/self.previous_sum_of_action_values
        if self.sum_of_action_values < 1:
            fraction_reward = -fraction_reward
        else:
            fraction_reward = 1
        self.previous_sum_of_action_values = self.sum_of_action_values
        self.sum_of_action_values = 0
        return fraction_reward
                
    def get_reward_componnents(self):
        fraction_reward = self.get_fraction_reward_component()
        alerts_reward = self.get_alerts_reward_component()
        distributions_reward = self.get_distributions_reward_component()
        energy_reward = self.get_energy_reward_component()        
        return energy_reward, alerts_reward, distributions_reward, fraction_reward
    
    def get_reward(self):
        energy_reward, alerts_reward, distributions_reward, fraction_reward = self.get_reward_componnents()
        reward = self.alpha  * energy_reward + self.beta *fraction_reward  + self.gamma*distributions_reward  + self.delta * alerts_reward
        self.reward_dict['energy'] += energy_reward
        self.reward_dict['alerts'] += alerts_reward
        self.reward_dict['distributions'] += distributions_reward
        self.reward_dict['fraction'] += fraction_reward
        self.reward_dict['total'] += reward
        logging.info(f"energy reward: {energy_reward}")
        logging.info(f"fraction reward: {fraction_reward}")
        logging.info(f"alerts reward: {alerts_reward}")
        logging.info(f"distributions reward: {distributions_reward}")
        logging.info(f"total reward: {reward}")
        return reward

    def step(self, action):
        self.current_step += 1
        if self.waiting_thread is None:
            self.waiting_thread = Thread(target=self.wait_to_measure)
            self.waiting_thread.start()
        if self.current_step == 1:
            self.before_first_step()
            return self.state, 0, False, {}       
        self.perform_action(action)
        self.update_state()
        reward = self.get_reward()
        self.logtype_index_counter()
        done = self.check_done()
        if not self.current_measurement_path is None:
            logging.info(f"dir name of last measurement: {self.current_measurement_path.split('/')[-1]}")
        logging.info(f"########################################################################################################################")
        return self.state, reward, done, {}

    def before_first_step(self):
        logging.info("waiting for first measurement")
        self.waiting_thread.join()
        first_alert_reward = self.get_alerts_reward_component()
        first_energy_reward = self.get_energy_reward_component()
        logging.info(f"first energy reward: {first_energy_reward}")
        logging.info(f"first alert reward: {first_alert_reward}")
    
    def perform_action(self, action):
        self.current_action = action[0]
        logtype = logtypes[self.logtype_index]
        logsource = logtype[0].lower()
        eventcode = logtype[1]
        action_value = int(self.current_action*self.max_actions_value)
        logging.info(f"action: {self.current_action}, action value: {action_value}, logtype: {logtype}")
        fake_logs = self.log_generator.generate_logs(logsource, eventcode, self.time_range, action_value)
        logging.info(f"{len(fake_logs)}: fake logs were generated")
        self.sum_of_action_values += self.current_action
        self.splunk_tools.insert_logs(fake_logs, logsource)
            
    def get_fake_distribution(self): #tool
        fake_distribution = self.state[:len(logtypes)].copy()
        # logging.info(f"current distribution: {self.state[:len(logtypes)]}")
        fake_distribution[self.logtype_index] += int(self.current_action*self.max_actions_value)
        # logging.info(f"fake distribution: {fake_distribution}")
        return fake_distribution
    
    def calculate_time_to_next_step(self): #tool
        now = datetime.datetime.now()
        seconds_since_hour = (now.minute * 60) + now.second
        seconds_into_step = seconds_since_hour % (self.rule_frequency * 60)
        seconds_until_next_step = (self.rule_frequency * 60) - seconds_into_step
        return seconds_until_next_step
        
    def wait_to_measure(self): #tool
        seconds_until_next_step = self.calculate_time_to_next_step()
        # Display a countdown timer while we wait
        for i in range(seconds_until_next_step-20, 0, -1):
            time.sleep(1)
        
    def measure(self, time_delta=60): #tool
        # self.measurement()
        # run the measurement script with subprocess
        logging.info('measuring')
        cmd = subprocess.run('python ../Scanner/scanner.py', shell=True, capture_output=True, text=True)
        logging.info(cmd.stdout)
        logging.info(cmd.stderr)
        # find the latest measurement folder
        measurement_num = max([int(folder.split(' ')[1]) for folder in os.listdir(PATH) if folder.startswith('Measurement')])
        self.current_measurement_path = os.path.join(PATH,f'Measurement {measurement_num}')
        logging.info(self.current_measurement_path)
        
        pids_energy_df = pd.read_csv(os.path.join(self.current_measurement_path, 'processes_data.csv'))
        pids_energy_df['Time(sec)'] = pd.to_datetime(pids_energy_df['Time(sec)'], unit='s').dt.to_pydatetime()
        rules_pids = self.splunk_tools.get_rules_pids(time_delta)
        rules_energy_df = self.filter_rules_energy(rules_pids, pids_energy_df)
        grouped_rules_enegry_df = self.aggregate_rules_energy(rules_energy_df)
        current_energy = grouped_rules_enegry_df['CPU(J)'].sum()
        return current_energy
    
    def after_measure_clean(self): #tool
        # update time range of rules with rule frequency
        self.time_range = ((pd.to_datetime(self.time_range[0],format='%m/%d/%Y:%H:%M:%S') + datetime.timedelta(minutes=self.rule_frequency)).strftime('%m/%d/%Y:%H:%M:%S'), (pd.to_datetime(self.time_range[1],format='%m/%d/%Y:%H:%M:%S') + datetime.timedelta(minutes=self.rule_frequency)).strftime('%m/%d/%Y:%H:%M:%S'))
        logging.info('update time range of rules')
        self.splunk_tools.update_all_searches(self.splunk_tools.update_search_time_range, self.time_range)   
        self.current_measurement_path = None
        self.waiting_thread = None
    
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
    
    def check_done(self):
        # Define the termination conditions based on the current state or other criteria
        if self.current_step >= self.max_steps:
            return True
        else:
            return False
