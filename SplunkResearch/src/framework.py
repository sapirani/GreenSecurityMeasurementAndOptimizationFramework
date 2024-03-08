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
from gym.spaces import Discrete, Tuple


sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment')
import os
from dotenv import load_dotenv
load_dotenv('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/.env')
urllib3.disable_warnings()
from log_generator import LogGenerator
from reward_calculator import RewardCalc
import gym
from gym import spaces

# BUG: Splunk doesnt parse all the logs
PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.92.1.el7.x86_64/Splunk Enterprise SIEM/Power Saver Plan/One Scan/'
INFINITY = 100000
CPU_TDP = 200
class Framework(gym.Env):
    def __init__(self, log_generator_instance, splunk_tools_instance, reward_calculator_instance, dt_manager, logger, time_range, rule_frequency, search_window, relevant_logtypes=[], max_actions_value=10, total_additional_logs=None):
        
        self.reward_calculator = reward_calculator_instance
        self.log_generator = log_generator_instance
        self.splunk_tools = splunk_tools_instance
        self.dt_manager = dt_manager
        self.logger = logger
        
        self.relevant_logtypes = relevant_logtypes
        self.time_range = time_range
        self.rule_frequency = rule_frequency
        self.search_window = search_window
        if total_additional_logs:
            self.step_size = total_additional_logs
            self.total_additional_logs = total_additional_logs
            self.time_range_update = False
            self.total_steps = 1
            self.action_duration = self.search_window*60/max(self.total_steps-1, 1)
        else:
            self.step_size = max_actions_value
            self.total_additional_logs = 0.1*100000*self.search_window//60
            self.time_range_update = True        
            self.total_steps = self.total_additional_logs//self.step_size
            self.action_duration = self.search_window*60/max(self.total_steps-1, 1)
        self.step_counter = 1
        self.action_upper_bound = 1
        self.sum_of_fractions = 0
        self.current_step = 0
        self.epsilon = 0
        
        # create the action space - a vector of size max_actions_value with values between 0 and 1
        self.action_space = spaces.Box(low=0,high=self.action_upper_bound,shape=(len(self.relevant_logtypes), 2),dtype=np.float64)
        self.observation_space = spaces.Box(low=0,high=self.action_upper_bound,shape=(2*len(self.relevant_logtypes),),dtype=np.float64)
        self.action_per_episode = []
        self.current_episode_accumulated_action = np.zeros(self.action_space.shape)
        self.fake_distribution = np.zeros(len(self.relevant_logtypes))
        self.real_distribution = np.zeros(len(self.relevant_logtypes))
        self.state = np.zeros(len(self.relevant_logtypes)*2).tolist()
        self.distribution = np.zeros(len(self.relevant_logtypes))
        self.done = False

    def get_reward(self):
        '''
        reward use cases description:
        if the sum of fractions is bigger than 1  the reward is very negative
        if the sum  of fractions is smaller or equal to 1 the reward is the sum of fractions
        if the sum of fractions is smaller or equal to 1 and the energy is bigger than the previous energy in more then 10% the reward is very positive 
        '''
        self.logger.debug(self.done)
        
        fraction_real_distribution = [x/sum(self.real_distribution) if sum(self.real_distribution) != 0 else 1/len(self.real_distribution) for x in self.real_distribution ]
        if self.done:
            reward = self.reward_calculator.get_full_reward(self.time_range, fraction_real_distribution, self.state)
        else:
            # reward = self.reward_calculator.get_previous_full_reward()
            reward = self.reward_calculator.get_partial_reward(fraction_real_distribution, self.state)
            
        # total_reward = self.alpha*energy_reward + self.beta*alert_reward + self.delta*distributions_reward + self.gamma*fraction_reward
        self.reward_calculator.reward_dict['total'].append(reward)
        self.logger.info(f"total reward: {reward}")               
        return reward
    
    def evaluate_no_agent(self):
        self.logger.debug(f"baseline evaluation")
        self.done = True
        self.update_state()
        reward = self.get_reward()
        return reward
    
    def step(self, action):
        action = self.action_preprocess(action)  
        # asyncio.run(self.perform_action(action))
        self.perform_action(action)
        if self.check_done():
            # asyncio.run(self.perform_action([max(self.action_upper_bound-self.sum_of_fractions,0)]))
            self.done = True
        self.update_state()   
        reward = self.get_reward()
        # if self.done:
        #     self.limit_learner_control()
        self.logger.info(f"########################################################################################################################")
        self.step_counter += 1
        return self.state, reward, self.done, {}

    def action_preprocess(self, action):
        self.logger.debug(f"step number: {self.step_counter}")
        action_norm_fcator = sum(sum(action))        
        if action_norm_fcator > 0:
            action /= action_norm_fcator
        return action
    

    def perform_action(self, action):
        self.current_episode_accumulated_action += action*self.step_size
        self.logger.info(f"action: {action}")              
        time_range = self.dt_manager.get_time_range_action(self.action_duration)
        for i, acts in enumerate(action):           
            logtype = self.relevant_logtypes[i]    
            logsource = logtype[0].lower()
            eventcode = logtype[1]
            for istrigger, act in enumerate(acts):
                if act:
                    fake_logs = self.log_generator.generate_logs(logsource, eventcode, istrigger,time_range, int(act*self.step_size))
                    # await self.splunk_tools.insert_logs(fake_logs, logsource, eventcode, istrigger)
                    self.splunk_tools.write_logs_to_monitor(fake_logs, logsource)
                    self.logger.debug(f"inserted {len(fake_logs)} logs of type {logsource} {eventcode} {istrigger}")      
        
        self.logger.debug(f"Current time: {self.dt_manager.set_fake_current_datetime(time_range[-1])}") 


    def set_max_actions_value(self, max_actions_value): 
        self.step_size = max_actions_value
    
    def update_state(self):
        state = []
        now = self.dt_manager.get_fake_current_datetime()
        previous_now = self.dt_manager.subtract_time(now, seconds=self.action_duration)
        distribution = self.splunk_tools.extract_distribution(previous_now, now)
        self.logger.debug(f"extraceted distribution {distribution}")
        for i, (source, event_code) in enumerate(self.relevant_logtypes):
            logtype = f"{source.lower()} {event_code}"
            if f"{logtype} 0" in distribution:
                self.real_distribution[i] += distribution[f"{logtype} 0"]
                self.fake_distribution[i] += distribution[f"{logtype} 0"]
                
            if f"{logtype} 1" in distribution:
                self.fake_distribution[i] += distribution[f"{logtype} 1"]
        real_sum = sum(self.real_distribution)
        fake_sum = sum(self.fake_distribution)
        self.logger.debug(f"real distribution: {self.real_distribution}")
        self.logger.debug(f"fake distribution: {self.fake_distribution}")
        self.logger.debug(f"real sum: {real_sum}")
        self.logger.debug(f"fake sum: {fake_sum}")
        state = self.real_distribution / max(real_sum, 1)
        state = np.concatenate((state, self.fake_distribution / max(fake_sum, 1)))
        self.state = state
        self.logger.info(f"state: {self.state}")
     
    
    def check_done(self):
        # Define the termination conditions based on the current state or other criteria
        if self.sum_of_fractions > self.action_upper_bound or self.step_counter == self.total_steps:
            return True
        else:
            return False
        
    def reset(self):
        self.logger.info("resetting")
        self.action_per_episode.append(self.current_episode_accumulated_action)
        self.current_log_type = 0
        self.sum_of_fractions = 0
        self.done = False
        self.step_counter = 1
        self.fake_distribution = np.zeros(len(self.relevant_logtypes))
        self.real_distribution = np.zeros(len(self.relevant_logtypes))
        self.current_episode_accumulated_action = np.zeros((len(self.relevant_logtypes), 2))
        if self.time_range_update:
            self.update_timerange()
        
        self.splunk_tools.delete_fake_logs(self.time_range)
        self.logger.debug(f"Current time: {self.dt_manager.set_fake_current_datetime(self.time_range[0])}") 

        
        # Reset the environment to an initial state
        self.update_state()
         
        # self.logtype_index_counter()  
        self.splunk_tools.update_all_searches(self.splunk_tools.update_search_time_range, self.time_range) 
        return self.state 

    def update_timerange(self):
        new_start_time = self.time_range[1]
        new_end_time = self.dt_manager.add_time(new_start_time, minutes=self.search_window)
        self.logger.debug(f'current time_range: {self.time_range}')
        self.time_range = (new_start_time, new_end_time)
        self.logger.debug(f'new time_range: {self.time_range}')       
            
    def render(self, mode='human'):
        self.logger.info(f"Current state: {self.state}")

        

        
# if __name__=="__main__":
#     # test the framework
