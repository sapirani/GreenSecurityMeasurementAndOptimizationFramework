import asyncio
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
    def __init__(self, log_generator_instance, splunk_tools_instance, dt_manager, logger, time_range, rule_frequency, search_window, reward_parameter_dict=None, relevant_logtypes=[], max_actions_value=10):
        self.log_generator = log_generator_instance
        self.splunk_tools = splunk_tools_instance
        self.dt_manager = dt_manager
        self.logger = logger
        self.time_range = time_range
        self.rule_frequency = rule_frequency
        self.search_window = search_window
        self.relevant_logtypes = relevant_logtypes
        self.max_actions_value = max_actions_value
        self.atomic_action = 1
        self.action_upper_bound = 1
        # create the action space - a vector of size max_actions_value with values between 0 and 1
        self.action_space = spaces.MultiBinary(len(self.relevant_logtypes))
        self.observation_space = spaces.Box(low=np.array([0]*((len(self.relevant_logtypes)))), high=np.array([self.action_upper_bound] * len(self.relevant_logtypes)))
        self.action_duration = self.search_window*60/len(self.relevant_logtypes)
        self.current_action = None
        self.state = None  # Initialize stat
        self.logtype_index = -1
        self.sum_of_action_values = 0
        self.sum_of_fractions = 0
        self.time_action_dict= [[]]
        self.current_step = 0
        self.fake_distribution = [0]*len(self.relevant_logtypes)
        self.real_distribution = [0]*len(self.relevant_logtypes)
        self.done = False
        self.epsilon = 0
        self.reward_calculator = RewardCalc(self.relevant_logtypes, self.dt_manager, self.logger, self.splunk_tools, self.rule_frequency)
        self.experiment_name = f"{self.search_window}_{self.max_actions_value}"
        self.limit_learner = True
        self.limit_learner_counter = 0
    
    def limit_learner_turn_off(self):
        self.limit_learner = False
        self.limit_learner_counter = 5
                                      
    def get_reward(self):
        '''
        reward use cases description:
        if the sum of fractions is bigger than 1  the reward is very negative
        if the sum  of fractions is smaller or equal to 1 the reward is the sum of fractions
        if the sum of fractions is smaller or equal to 1 and the energy is bigger than the previous energy in more then 10% the reward is very positive 
        '''
        self.logger.info(self.done)
        
        fraction_real_distribution = self.real_distribution
        # fraction_real_distribution = [x/sum(self.real_distribution) for x in self.real_distribution]
        if self.done:
            reward = self.reward_calculator.get_full_reward(self.time_range, fraction_real_distribution, self.state)
        else:
            # reward = self.reward_calculator.get_previous_full_reward()
            reward = self.reward_calculator.get_partial_reward(fraction_real_distribution, self.state)
            
        # total_reward = self.alpha*energy_reward + self.beta*alert_reward + self.delta*distributions_reward + self.gamma*fraction_reward
        self.reward_calculator.reward_dict['total'].append(reward)
        self.logger.info(f"total reward: {reward}")               
        return reward
        
    def step(self, action):
        asyncio.run(self.perform_action(action))
        if self.check_done():
            self.logtype_index_counter()                     
            asyncio.run(self.perform_action([max(self.action_upper_bound-self.sum_of_fractions,0)]))
            self.done = True
        self.update_state()   
        self.logtype_index_counter()        
        reward = self.get_reward()
        if self.done:
            self.limit_learner_control()
        self.logger.info(f"########################################################################################################################")
        return self.state, reward, self.done, {}

    def limit_learner_control(self):
        if self.sum_of_fractions != 1:
            self.limit_learner_counter = max(0, self.limit_learner_counter-1)
        else:
            self.limit_learner_counter = min(5, self.limit_learner_counter+1)
        if self.limit_learner_counter > 4:
            self.limit_learner = False
        else:
            self.limit_learner = True
        self.logger.info(f"limit learner: {self.limit_learner}")
        self.logger.info(f"limit learner counter: {self.limit_learner_counter}")
    
    def blind_step(self, action):
        asyncio.run(self.perform_action(action))
        if self.check_done():
            self.logtype_index_counter()
            asyncio.run(self.perform_action([max(self.action_upper_bound-self.sum_of_fractions,0)]))
            self.done = True
                  
        self.logtype_index_counter()           
        reward = self.get_reward()
        self.logger.info(f"########################################################################################################################")
        return self.state, reward, self.done, {}

    async def perform_action(self, action):
        # action = int(action)//self.max_actions_value
        self.logger.info(f"logindex: {self.logtype_index}")
        # calculate the current time range according to the time left till the next measurement
        now = self.dt_manager.get_fake_current_datetime()
        self.logger.info(f"Current time: {now}")
        self.action_duration = sum(action)*self.search_window*60/len(self.max_actions_value)    
        time_range = (now, self.dt_manager.add_time(now, seconds=self.action_duration))
        for i,act in enumerate(action):
            
            logtype = self.relevant_logtypes[i]    
            logsource = logtype[0].lower()
            eventcode = logtype[1]
            if act:
                fake_logs = self.log_generator.generate_logs(logsource, eventcode, time_range, act)
                self.logger.info(f"{len(fake_logs)}: fake logs were generated")
                await self.splunk_tools.insert_logs(fake_logs, logsource)

        self.sum_of_fractions += sum(action)
        new_current_fake_time = self.dt_manager.add_time(now, seconds=self.action_duration)
        self.dt_manager.set_fake_current_datetime(new_current_fake_time)
        self.logger.info(f"Current time: {self.dt_manager.get_fake_current_datetime()}")      

    def set_max_actions_value(self, max_actions_value): 
        self.max_actions_value = max_actions_value
    
    def update_state(self):
        state = []
        # if not self.limit_learner:
        distribution = self.splunk_tools.extract_distribution(self.time_range[0], self.dt_manager.get_fake_current_datetime())
        self.logger.info(f"extraceted distribution {distribution}")
        for i, logtype in enumerate(self.relevant_logtypes):
            if f"{logtype[0].lower()} {logtype[1]}" in distribution:
                if distribution[f"{logtype[0].lower()} {logtype[1]}"][1] == 0:
                    self.real_distribution[i] = distribution[f"{logtype[0].lower()} {logtype[1]}"][0]
                else:
                    self.fake_distribution[i] = distribution[f"{logtype[0].lower()} {logtype[1]}"][0]
            else:
                self.real_distribution[i] = 0
                self.fake_distribution[i] = 0
            # self.real_distribution = [distribution[f"{logtype[0].lower()} {logtype[1]}"][0] if f"{logtype[0].lower()} {logtype[1]}" in distribution else self.epsilon for logtype in self.relevant_logtypes]

        # else:
        #     self.real_distribution = [0 for i in range(len(self.relevant_logtypes))]
        # self.fake_distribution = [self.fake_distribution[i] if self.fake_distribution[i] else self.epsilon for i in range(len(self.relevant_logtypes)) ]

        self.logger.info(f"real distribution: {self.real_distribution}")
        self.logger.info(f"fake distribution: {self.fake_distribution}")
        state = [x + y for x, y in zip(self.fake_distribution, self.real_distribution)]
        sum_state = sum(state)
        state = [x/sum_state if sum_state else 1/len(state) for x in state]
        state.append(self.logtype_index)
        state.append(self.action_upper_bound - self.sum_of_fractions)
        state.append(int(self.limit_learner))
        self.logger.info(f"state: {state}")
        self.state = np.array(state)
        
    
    def check_done(self):
        # Define the termination conditions based on the current state or other criteria
        if self.sum_of_fractions*self.atomic_action >= self.max_actions_value:
            return True
        else:
            return False
        
    def reset(self):
        self.logger.info("resetting")
        self.current_log_type = 0
        self.sum_of_fractions = 0
        self.done = False
        self.sum_of_action_values = 0
        self.update_timerange()
        self.dt_manager.set_fake_current_datetime(self.time_range[0])
        # Reset the environment to an initial state
        self.fake_distribution = [0]*len(self.relevant_logtypes)  
        self.update_state() 
        self.logtype_index_counter()  
        self.time_action_dict.append([])
        # self.time_action_dict[str(self.time_range)] = {}
        self.splunk_tools.update_all_searches(self.splunk_tools.update_search_time_range, self.time_range) 
        return self.state 

    def update_timerange(self):
        new_start_time = self.time_range[1]
        new_end_time = self.dt_manager.add_time(new_start_time, minutes=self.search_window)
        self.logger.info(f'current time_range: {self.time_range}')
        self.time_range = (new_start_time, new_end_time)
        self.logger.info(f'new time_range: {self.time_range}')   
            
    def logtype_index_counter(self):
        self.logger.info(f"current logtype index: {self.logtype_index}")
        self.logtype_index += 1
        if self.done:
            self.logtype_index = -1     
            
    def render(self, mode='human'):
        self.logger.info(f"Current state: {self.state}")

        

        
# if __name__=="__main__":
#     # test the action performing
#     from SplunkResearch.model.utils import MockedDatetimeManager
#     from SplunkResearch.splunk_tools import SplunkTools
#     from config import replacement_dicts as big_replacement_dicts
#     from logtypes import logtypes
#     from section_logtypes import section_logtypes
#     fake_start_datetime = datetime.datetime(2023,6,22, 8, 30, 0)
#     rule_frequency = 5
#     search_window = 5
#     max_actions_value = 5000
#     time_range = ('06/22/2023:08:30:00', '06/22/2023:08:35:00')
#     current_dir = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/model'
#     reward_parameter_dict ={'alpha': 0.6, 'beta': 0.05, 'gamma': 0.15, 'delta': 0.20}
#     logtypes=[ ('XmlWinEventLog:Microsoft-Windows-Sysmon/Operational', '3')]
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_file_name = f"test_{timestamp}"
#     dt_manager = MockedDatetimeManager(fake_start_datetime=fake_start_datetime, log_file_path=f"{current_dir}/{log_file_name}.txt")
#     splunk_tools_instance = SplunkTools(dt_manager=dt_manager)
#     log_generator_instance = LogGenerator(logtypes, big_replacement_dicts, splunk_tools_instance)
#     env = Framework(log_generator_instance, splunk_tools_instance, dt_manager, time_range, rule_frequency, search_window, max_actions_value=max_actions_value, reward_parameter_dict=reward_parameter_dict, relevant_logtypes=logtypes)
#     # env.reset()
#     # for i in enumerate(logtypes):
#     #     env.perform_action(100)
#     #     env.update_state()
#     #env.done = True
#     env.reset()
#     for i in enumerate(logtypes):
#         asyncio.run(env.perform_action(50))
#         env.update_state()
        

        
    