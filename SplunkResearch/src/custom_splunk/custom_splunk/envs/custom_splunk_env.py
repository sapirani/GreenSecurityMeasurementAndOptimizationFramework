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
    def __init__(self, fake_start_datetime, rule_frequency, search_window, savedsearches, state_strategy, span_size=1, total_additional_logs=None, logs_per_minute = 300, additional_percentage = 0.1, env_id=None, num_of_measurements=1):
        self.env_id = env_id
        relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]})) #[(x[0], str(x[1])) for x in state_span]
        relevant_logtypes.append(('wineventlog:security', '4624'))
        num_of_searches = len(savedsearches)
        self.splunk_tools_instance  = SplunkTools(savedsearches, num_of_measurements, rule_frequency)
        self.log_generator = LogGenerator(relevant_logtypes, self.splunk_tools_instance)
        self.relevant_logtypes = relevant_logtypes
        logger.info(f"relevant logtypes: {self.relevant_logtypes}")
        self.top_logtypes = pd.read_csv("resources/top_logtypes.csv")
        self.top_logtypes = self.top_logtypes.sort_values(by='count', ascending=False)[['source', "EventCode"]].values.tolist()[:50]
        self.top_logtypes = [(x[0].lower(), str(x[1])) for x in self.top_logtypes]
        self.top_logtypes = set(self.top_logtypes)|set(self.relevant_logtypes)
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
        # self.observation_space = spaces.Box(low=0,high=self.action_upper_bound,shape=(len(self.top_logtypes)*2,),dtype=np.float64)
        self.state_strategy = state_strategy(self.top_logtypes)
        self.observation_space = self.state_strategy.create_state()
        self.action_per_episode = []
        self.current_episode_accumulated_action = np.zeros(self.action_space.shape)
        self.current_action = np.zeros(self.action_space.shape)
        self.fake_logtypes_counter = {}
        self.splunk_tools_instance.real_logtypes_counter = {}
        self.real_distribution = np.zeros(len(self.top_logtypes))
        self.state = np.zeros(self.observation_space.shape).tolist()

        self.done = False
        self.fake_start_datetime = fake_start_datetime   
        fake_start_datetime  = datetime.datetime.strptime(fake_start_datetime, '%m/%d/%Y:%H:%M:%S')
        clean_env(self.splunk_tools_instance, (fake_start_datetime.timestamp(), (fake_start_datetime+datetime.timedelta(days=90)).timestamp()))
        # clean_env(splunk_tools_instance, (fake_start_datetime.strftime('%m/%d/%Y:%H:%M:%S'), (fake_start_datetime+datetime.timedelta(days=30)).strftime('%m/%d/%Y:%H:%M:%S')))
        
        self.dt_manager = MockedDatetimeManager(fake_start_datetime=fake_start_datetime)
        end_time = self.dt_manager.get_fake_current_datetime()
        start_time = self.dt_manager.subtract_time(end_time, minutes=search_window)
        time_range = (start_time, end_time)
        self.time_range = time_range
        # update_rules_frequency_and_time_range(self.splunk_tools_instance, time_range)
        self.reward_calculator = None
        self.num_of_searches = num_of_searches

    def set_reward_calculator(self, reward_calculator):
        self.reward_calculator = reward_calculator

    def get_reward(self):
 
        logger.debug(self.done)
        
        fraction_real_distribution = [x/sum(self.real_distribution) if sum(self.real_distribution) != 0 else 1/len(self.real_distribution) for x in self.real_distribution ] #BUG WTF?
        if self.done:
            reward = self.reward_calculator.get_full_reward(self.time_range, fraction_real_distribution, self.state)
        else:
            # reward = self.reward_calculator.get_previous_full_reward()
            reward = self.reward_calculator.get_partial_reward(fraction_real_distribution, self.state)
            
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
        if self.step_counter == 1:
            self.reward_calculator.get_no_agent_reward(self.time_range)
        logger.debug(f"step number: {self.step_counter}")
        action = self.action_preprocess(action)  
        self.current_action = action
        # asyncio.run(self.perform_action(action))
        self.perform_action(action)
        if self.check_done():
            self.done = True
        # self.update_state()   
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
        self.splunk_tools_instance.write_logs_to_monitor(fake_logs, logsource)
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
        real_logtypes_counter = self.splunk_tools_instance.get_real_distribution(previous_now, now)
        self.state = self.state_strategy.update_state(real_logtypes_counter, self.fake_logtypes_counter)
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
        self.action_per_episode.append(self.current_episode_accumulated_action)
        self.current_log_type = 0
        self.done = False
        self.step_counter = 1
        self.fake_distribution = np.zeros(len(self.top_logtypes))
        self.real_distribution = np.zeros(len(self.top_logtypes))
        self.fake_logtypes_counter = {}
        self.current_episode_accumulated_action = np.zeros(self.action_space.shape)
        self.current_action = np.zeros(self.action_space.shape)
        self.state = np.zeros(self.observation_space.shape)
        self.splunk_tools_instance.real_logtypes_counter = {}
        
        if self.time_range_update:
            self.update_timerange()
        logger.debug(f"Current time: {self.dt_manager.set_fake_current_datetime(self.time_range[0])}") 
        
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
