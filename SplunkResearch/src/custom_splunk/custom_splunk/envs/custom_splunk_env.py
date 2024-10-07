import datetime
import numpy as np
import pandas as pd
import sys
import urllib3
import logging
from env_utils import *
from datetime_manager import MockedDatetimeManager

import tensorflow as tf
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment')
import os
from dotenv import load_dotenv
load_dotenv('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/.env')
urllib3.disable_warnings()
from splunk_tools import SplunkTools
from log_generator import LogGenerator
from resources.section_logtypes import section_logtypes
import gym
from gym import spaces
import logging
logger = logging.getLogger(__name__)

PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.92.1.el7.x86_64/Splunk Enterprise SIEM/Power Saver Plan/One Scan/'
INFINITY = 100000
CPU_TDP = 200
class SplunkEnv(gym.Env):
    def __init__(self, fake_start_datetime, rule_frequency, search_window, savedsearches, state_strategy, action_strategy, span_size=1, total_additional_logs=None, logs_per_minute = 300, additional_percentage = 0.1, env_id=None, num_of_measurements=1):
        self.env_id = env_id
        relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]})) #[(x[0], str(x[1])) for x in state_span]
        relevant_logtypes.append(('wineventlog:security', '4624'))
        self.relevant_logtypes = relevant_logtypes
        logger.info(f"relevant logtypes: {self.relevant_logtypes}")
        num_of_searches = len(savedsearches)
        self.splunk_tools_instance  = SplunkTools(savedsearches, num_of_measurements, rule_frequency)
        self.log_generator = LogGenerator(relevant_logtypes, self.splunk_tools_instance)
        self.search_window = search_window
        self.total_additional_logs = additional_percentage*logs_per_minute*self.search_window #//60    
        self.action_duration = span_size #TODO change span_size to minutes 
        self.step_size = int((self.total_additional_logs//self.search_window)*self.action_duration//60)
        self.total_steps = self.search_window*60//self.action_duration
        logger.debug(f"total steps: {self.total_steps} action duration: {self.action_duration} step size: {self.step_size} total additional logs: {self.total_additional_logs}")
        # create the action space - a vector of size max_actions_value with values between 0 and 1
        self.action_strategy = action_strategy(self.relevant_logtypes, 1, self.step_size, self.action_duration, self.splunk_tools_instance, self.log_generator)
        self.action_space = self.action_strategy.create_action_space()
        self.action_per_episode = []
        self.current_action = None
        
        self.top_logtypes = pd.read_csv("resources/top_logtypes.csv")
        self.top_logtypes = self.top_logtypes.sort_values(by='count', ascending=False)[['source', "EventCode"]].values.tolist()[:50]
        self.top_logtypes = [(x[0].lower(), str(x[1])) for x in self.top_logtypes]
        self.top_logtypes = set(self.top_logtypes)|set(self.relevant_logtypes)
        self.state_strategy = state_strategy(self.top_logtypes)
        self.observation_space = self.state_strategy.create_state()
        self.splunk_tools_instance.real_logtypes_counter = {}
        self.real_distribution = np.zeros(len(self.top_logtypes))
        self.state = np.zeros(self.observation_space.shape).tolist()



        self.done = False
        self.step_counter = 1
        self.action_upper_bound = 1
        self.current_step = 0
        self.epsilon = 0
        self.fake_start_datetime = fake_start_datetime   
        fake_start_datetime  = datetime.datetime.strptime(fake_start_datetime, '%m/%d/%Y:%H:%M:%S')
        clean_env(self.splunk_tools_instance, (fake_start_datetime.timestamp(), (fake_start_datetime+datetime.timedelta(days=90)).timestamp()))
        
        self.dt_manager = MockedDatetimeManager(fake_start_datetime=fake_start_datetime)
        end_time = self.dt_manager.get_fake_current_datetime()
        start_time = self.dt_manager.subtract_time(end_time, minutes=search_window)
        time_range = (start_time, end_time)
        self.time_range = time_range
        # update_rules_frequency_and_time_range(self.splunk_tools_instance, time_range)
        self.reward_calculator = None
        self.num_of_searches = num_of_searches
        
                
        # run the saved searches for warmup
        self.splunk_tools_instance.run_saved_searches(time_range)

    def set_reward_calculator(self, reward_calculator):
        self.reward_calculator = reward_calculator

    def get_reward(self):
 
        logger.debug(self.done)
        if self.done:
            reward = self.reward_calculator.get_full_reward(self.time_range, self.state_strategy.real_state, self.state_strategy.fake_state, self.current_action)
        else:
            # reward = self.reward_calculator.get_previous_full_reward()
            reward = self.reward_calculator.get_partial_reward(self.state_strategy.real_state, self.state_strategy.fake_state, self.current_action)
            
        self.reward_calculator.reward_dict['total'].append(reward)
        logger.info(f"total reward: {reward}")               
        return reward
    
    def evaluate_no_agent(self):
        logger.debug(f"baseline evaluation")
        self.done = True
        self.update_state()
        reward = self.get_reward()
        return reward
    
    def step(self, action):
        self.current_action = action
        if self.step_counter == 1:
            self.reward_calculator.get_no_agent_reward(self.time_range)
        logger.debug(f"step number: {self.step_counter}")
        time_range = self.dt_manager.get_time_range_action(self.action_duration)        
        self.action_strategy.perform_action(action, time_range)
        logger.debug(f"Current time: {self.dt_manager.set_fake_current_datetime(time_range[-1])}") # dont remove!!
        if self.check_done():
            self.done = True
        self.update_state()   
        reward = self.get_reward()
        logger.info(f"########################################################################################################################")
        self.step_counter += 1
        return self.state, reward, self.done, {}


    def set_max_actions_value(self, max_actions_value): 
        self.step_size = max_actions_value

    def get_fake_distribution(self):
        fake_logtypes_counter = {}
        for i, logtype in enumerate(self.relevant_logtypes):
            logsource = logtype[0].lower()
            eventcode = logtype[1]
            fake_logtypes_counter[f"{logsource} {eventcode}"] = self.action_strategy.current_episode_accumulated_action[i*2]
            if i == len(self.relevant_logtypes)-1:
                break
            fake_logtypes_counter[f"{logsource} {eventcode}"] += self.action_strategy.current_episode_accumulated_action[i*2+1]
        return fake_logtypes_counter
    
    
    def update_state(self):
        now = self.dt_manager.get_fake_current_datetime()
        previous_now = self.dt_manager.subtract_time(now, seconds=self.action_duration)
        real_logtypes_counter = self.splunk_tools_instance.get_real_distribution(previous_now, now)
        fake_logtypes_counter = self.get_fake_distribution()
        self.state_strategy.update_distributions(real_logtypes_counter, fake_logtypes_counter)
        self.state = self.state_strategy.update_state()
        logger.info(f"state: {self.state}")
        return self.state
        
    def check_done(self):
        # Define the termination conditions based on the current state or other criteria
        if self.step_counter == self.total_steps:
            return True
        else:
            return False
        
    def reset(self):
        logger.info("resetting")
        self.action_per_episode.append(self.action_strategy.current_episode_accumulated_action)
        self.current_log_type = 0
        self.done = False
        self.step_counter = 1
        self.real_distribution = np.zeros(len(self.top_logtypes))
        self.state = np.zeros(self.observation_space.shape)
        self.splunk_tools_instance.real_logtypes_counter = {}
        self.update_timerange()
        logger.debug(f"Current time: {self.dt_manager.set_fake_current_datetime(self.time_range[0])}") 
        self.action_strategy.reset()
        # self.reward_calculator.get_no_agent_reward(self.time_range)        
        
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
