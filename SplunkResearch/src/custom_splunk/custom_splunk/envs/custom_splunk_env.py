import asyncio
import datetime
import random
import subprocess
from threading import Thread, Timer
import time
import numpy as np
import pandas as pd
import sys
import urllib3
import logging
from scipy.stats import entropy
from env_utils import *
from measurement import Measurement
from datetime_manager import MockedDatetimeManager
from reward_calculator import RewardCalc

import tensorflow as tf
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment')
import os
from dotenv import load_dotenv
load_dotenv('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/.env')
urllib3.disable_warnings()

import gym
from gym import spaces
import logging
logger = logging.getLogger(__name__)

PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.92.1.el7.x86_64/Splunk Enterprise SIEM/Power Saver Plan/One Scan/'
INFINITY = 100000
CPU_TDP = 200
class SplunkEnv(gym.Env):
    def __init__(self, log_generator_instance, splunk_tools_instance, fake_start_datetime, rule_frequency, search_window, num_of_searches, reward_parameters, is_measure_energy, tf_log_path, relevant_logtypes=[], span_size=1, total_additional_logs=None, logs_per_minute = 300, additional_percentage = 0.1):
        
        self.log_generator = log_generator_instance
        self.splunk_tools = splunk_tools_instance
        
        self.relevant_logtypes = relevant_logtypes
        self.top_logtypes = pd.read_csv("resources/top_logtypes.csv")
        self.top_logtypes = self.top_logtypes.sort_values(by='count', ascending=False)[['source', "EventCode"]].values.tolist()[:50]
        self.top_logtypes = [(x[0].lower(), str(x[1])) for x in self.top_logtypes]
        self.top_logtypes = set(self.top_logtypes)|set(self.relevant_logtypes)
        self.rule_frequency = rule_frequency
        self.search_window = search_window
        if total_additional_logs:
            self.step_size = total_additional_logs
            self.total_additional_logs = total_additional_logs
            self.time_range_update = False
            self.total_steps = 1
            self.action_duration = self.search_window*60/max(self.total_steps, 1)
        else:
            self.total_additional_logs = additional_percentage*logs_per_minute*self.search_window #//60
            self.time_range_update = True        
            self.action_duration = span_size #TODO change span_size to minutes
            self.step_size = int((self.total_additional_logs//self.search_window)*self.action_duration//60)
            self.total_steps = self.search_window*60//self.action_duration
            logger.debug(f"total steps: {self.total_steps} action duration: {self.action_duration} step size: {self.step_size} total additional logs: {self.total_additional_logs}")
        
        self.step_counter = 1
        self.action_upper_bound = 1
        self.current_step = 0
        self.epsilon = 0
        
        # create the action space - a vector of size max_actions_value with values between 0 and 1
        self.action_space = spaces.Box(low=0,high=self.action_upper_bound,shape=((len(self.relevant_logtypes)-1)*2+1, ),dtype=np.float64)
        self.observation_space = spaces.Box(low=0,high=self.action_upper_bound,shape=(len(self.top_logtypes)*2,),dtype=np.float64)
        self.action_per_episode = []
        self.current_episode_accumulated_action = np.zeros(self.action_space.shape)
        self.fake_logtypes_counter = {}
        self.real_logtypeps_counter = {}
        self.fake_distribution = np.zeros(len(self.top_logtypes))
        self.real_distribution = np.zeros(len(self.top_logtypes))
        self.state = np.zeros(self.observation_space.shape).tolist()
        self.abs_distribution = {}
        self.done = False
                
        fake_start_datetime  = datetime.datetime.strptime(fake_start_datetime, '%m/%d/%Y:%H:%M:%S')
        clean_env(splunk_tools_instance, (fake_start_datetime.strftime('%m/%d/%Y:%H:%M:%S'), (fake_start_datetime+datetime.timedelta(days=30)).strftime('%m/%d/%Y:%H:%M:%S')))
        
        self.dt_manager = MockedDatetimeManager(fake_start_datetime=fake_start_datetime)
        end_time = self.dt_manager.get_fake_current_datetime()
        start_time = self.dt_manager.subtract_time(end_time, minutes=search_window)
        time_range = (start_time, end_time)
        self.time_range = time_range
        update_rules_frequency_and_time_range(splunk_tools_instance, rule_frequency, time_range)
        
        alpha, beta, gamma = reward_parameters['alpha'], reward_parameters['beta'], reward_parameters['gamma']
        measurment_tool = Measurement(splunk_tools_instance, num_of_searches, measure_energy=is_measure_energy)
        
        # TensorBoard setup
        self.summary_writer = tf.summary.create_file_writer(tf_log_path)
        self.reward_calculator = RewardCalc(self.top_logtypes, self.dt_manager, splunk_tools_instance, rule_frequency, num_of_searches, measurment_tool, alpha, beta, gamma, self.summary_writer)
 


    def get_reward(self):
 
        logger.debug(self.done)
        
        fraction_real_distribution = [x/sum(self.real_distribution) if sum(self.real_distribution) != 0 else 1/len(self.real_distribution) for x in self.real_distribution ]
        if self.done:
            reward = self.reward_calculator.get_full_reward(self.time_range, fraction_real_distribution, self.state)
        else:
            # reward = self.reward_calculator.get_previous_full_reward()
            reward = self.reward_calculator.get_partial_reward(fraction_real_distribution, self.state)
            
        self.reward_calculator.reward_dict['total'].append(reward)
        with self.summary_writer.as_default():
            tf.summary.scalar('total_reward', reward, step=len(self.reward_calculator.reward_dict['total']))
        logger.info(f"total reward: {reward}")               
        return reward
    
    def evaluate_no_agent(self):
        logger.debug(f"baseline evaluation")
        self.done = True
        self.update_state()
        reward = self.get_reward()
        return reward
    
    def step(self, action):
        logger.debug(f"step number: {self.step_counter}")
        action = self.action_preprocess(action)  
        # asyncio.run(self.perform_action(action))
        self.perform_action(action)
        if self.check_done():
            self.done = True
        self.update_state()   
        reward = self.get_reward()
        logger.info(f"########################################################################################################################")
        self.step_counter += 1
        return self.state, reward, self.done, {}

    def action_preprocess(self, action):
        logger.info(f"action before preprocessing: {action}")
        # for i in range(len(action)):
        #     act = action[i]
        #     if act < 1/self.step_size:
        #         action[i] = 0            
        action_norm_fcator = sum(action)
        if action_norm_fcator > 0:
            action /= action_norm_fcator
        return action
    

    def perform_action(self, action):
        # self.current_episode_accumulated_action += action*self.step_size
        logger.info(f"action: {action}")              
        time_range = self.dt_manager.get_time_range_action(self.action_duration)
        for i, logtype in enumerate(self.relevant_logtypes): 
            for istrigger in range(2):
                act  = action[i*2+istrigger]
                if act:
                    self.perform_act(time_range, i, istrigger, act)  
                if i ==len(self.relevant_logtypes)-1:
                    break
        # non_action = action[-1]
        # if non_action:
        #     self.perform_act(time_range, "non_action", "non_action", 0, non_action)       
        logger.debug(f"Current time: {self.dt_manager.set_fake_current_datetime(time_range[-1])}") 

    def perform_act(self, time_range, i, istrigger, act):
        logtype = self.relevant_logtypes[i]
        logsource = logtype[0].lower()
        eventcode = logtype[1]
        absolute_act = int(act*self.step_size)
        self.current_episode_accumulated_action[i*2+istrigger] += absolute_act
        fake_logs = self.log_generator.generate_logs(logsource, eventcode, istrigger,time_range, absolute_act)
        self.splunk_tools.write_logs_to_monitor(fake_logs, logsource)
        logger.debug(f"inserted {len(fake_logs)} logs of type {logsource} {eventcode} {istrigger}")
        if f"{logsource} {eventcode}" in self.fake_logtypes_counter:
            self.fake_logtypes_counter[f"{logsource} {eventcode}"] += absolute_act
        else:
            self.fake_logtypes_counter[f"{logsource} {eventcode}"] = absolute_act


    def set_max_actions_value(self, max_actions_value): 
        self.step_size = max_actions_value

    
    def update_state(self):
        now = self.dt_manager.get_fake_current_datetime()
        previous_now = self.dt_manager.subtract_time(now, seconds=self.action_duration)
        real_state, fake_state = self.update_distributions(now, previous_now)
        self.state = np.concatenate((real_state, fake_state))
        logger.info(f"state: {self.state}")

    def update_distributions(self, now, previous_now):
        real_distribution_dict = self.splunk_tools.get_real_distribution(previous_now, now)
        logger.debug(f"real distribution: {real_distribution_dict}")
        logger.debug(f"fake distribution: {self.fake_logtypes_counter}")
        for logtype in real_distribution_dict:
            if logtype in self.real_logtypeps_counter:
                self.real_logtypeps_counter[logtype] += real_distribution_dict[logtype]
            else:
                self.real_logtypeps_counter[logtype] = real_distribution_dict[logtype]
        real_state = []
        fake_state = []
        for i, logtype in enumerate(self.top_logtypes):
            # create state vector
            logtype = ' '.join(logtype)
            if logtype in self.real_logtypeps_counter:
                real_count = self.real_logtypeps_counter[logtype]
                real_state.append(real_count)
                fake_state.append(real_count)
            else:
                real_state.append(0)
                fake_state.append(0)
            if logtype in self.fake_logtypes_counter:
                fake_state[i] += self.fake_logtypes_counter[logtype]

        real_total_sum = sum(real_state)
        fake_total_sum = sum(fake_state)
        real_state = [x/real_total_sum if real_total_sum!= 0 else 1/len(real_state) for x in real_state]
        fake_state = [x/fake_total_sum if fake_total_sum != 0 else 1/len(fake_state) for x in fake_state]
        return real_state,fake_state
            
     
    
    def check_done(self):
        # Define the termination conditions based on the current state or other criteria
        if self.step_counter == self.total_steps:
            return True
        else:
            return False
        
    def reset(self):
        logger.info("resetting")
        self.action_per_episode.append(self.current_episode_accumulated_action)
        self.current_log_type = 0
        self.done = False
        self.step_counter = 1
        self.fake_distribution = np.zeros(len(self.top_logtypes))
        self.real_distribution = np.zeros(len(self.top_logtypes))
        self.abs_distribution = {}
        self.fake_logtypes_counter = {}
        self.real_logtypeps_counter = {}
        self.current_episode_accumulated_action = np.zeros(self.action_space.shape)
        self.state = np.zeros(self.observation_space.shape)
        
        if self.time_range_update:
            self.update_timerange()
        logger.debug(f"Current time: {self.dt_manager.set_fake_current_datetime(self.time_range[0])}") 
        
        self.reward_calculator.get_no_agent_reward(self.time_range)        
        
        return self.state 

    def update_timerange(self):
        # new_start_time = self.time_range[1]
        new_start_time = self.time_range[1]
        new_end_time = self.dt_manager.add_time(new_start_time, minutes=self.search_window)
        logger.debug(f'current time_range: {self.time_range}')
        self.time_range = (new_start_time, new_end_time)
        logger.debug(f'new time_range: {self.time_range}')       
            
    def render(self, mode='human'):
        logger.info(f"Current state: {self.state}")

        

        
# if __name__=="__main__":
#     # test the framework
