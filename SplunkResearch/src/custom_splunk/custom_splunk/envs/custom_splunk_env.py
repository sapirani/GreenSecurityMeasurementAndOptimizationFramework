import datetime
import random
import time
import numpy as np
import pandas as pd
import sys
import urllib3
import logging
from env_utils import *
from datetime_manager import MockedDatetimeManager

import tensorflow as tf
from strategies.action_strategy import ActionStrategy7, ActionStrategy8

from strategies.state_strategy import StateStrategy10, StateStrategy11, StateStrategy6, StateStrategy7, StateStrategy8
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
import concurrent.futures

PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.92.1.el7.x86_64/Splunk Enterprise SIEM/Power Saver Plan/One Scan/'
INFINITY = 100000
CPU_TDP = 200
class SplunkEnv(gym.Env):
    def __init__(self, fake_start_datetime, rule_frequency, search_window, savedsearches, state_strategy, action_strategy, span_size=1, total_additional_logs=None, logs_per_minute = 300, additional_percentage = 0.1, env_id=None, num_of_measurements=1, num_of_episodes=1000):
        self.env_id = env_id
        self.logs_per_minute = logs_per_minute
        relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]})) #[(x[0], str(x[1])) for x in state_span]
        # relevant_logtypes.append(('wineventlog:security', '4624'))
        self.relevant_logtypes = relevant_logtypes
        logger.info(f"relevant logtypes: {self.relevant_logtypes}")
        num_of_searches = len(savedsearches)
        self.rule_frequency = rule_frequency
        self.splunk_tools_instance  = SplunkTools(savedsearches, num_of_measurements, rule_frequency)
        self.log_generator = LogGenerator(relevant_logtypes, self.splunk_tools_instance)
        self.search_window = search_window
        self.action_duration = span_size #TODO change span_size to minutes 
        self.total_steps = self.search_window*60//self.action_duration
        self.additional_percentage = additional_percentage
        
        logger.debug(f"total steps: {self.total_steps} action duration: {self.action_duration} ")
        # create the action space - a vector of size max_actions_value with values between 0 and 1
        self.action_strategy = action_strategy(self.relevant_logtypes, 1, 0, self.action_duration, self.splunk_tools_instance, self.log_generator, 0)
        self.action_space = self.action_strategy.create_action_space()
        self.action_per_episode = []
        self.current_action = None
        self.step_violation = False
        self.top_logtypes = pd.read_csv("resources/top_logtypes.csv")
        # include only system and security logs
        self.top_logtypes = self.top_logtypes[self.top_logtypes['source'].str.lower().isin(['wineventlog:security', 'wineventlog:system'])]
        self.top_logtypes = self.top_logtypes.sort_values(by='count', ascending=False)[['source', "EventCode"]].values.tolist()[:50]
        self.top_logtypes = [(x[0].lower(), str(x[1])) for x in self.top_logtypes]
        self.top_logtypes = set(self.top_logtypes)|set(self.relevant_logtypes)
        if state_strategy == StateStrategy6:
            self.state_strategy = state_strategy(self.top_logtypes, self.relevant_logtypes)
        elif state_strategy == StateStrategy7:
            self.state_strategy = state_strategy(self.top_logtypes, self.splunk_tools_instance.active_saved_searches)
        else:
            self.state_strategy = state_strategy(self.top_logtypes)
        self.observation_space = self.state_strategy.create_state()
        self.splunk_tools_instance.real_logtypes_counter = {}
        self.real_distribution = np.zeros(len(self.top_logtypes))
        self.state = np.zeros(self.observation_space.shape).tolist()

        self.action_auditor = []

        self.action_done = False
        self.done = False
        self.step_counter = 1
        self.action_upper_bound = 1
        self.current_step = 0
        self.epsilon = 0
        self.fake_start_datetime = fake_start_datetime   
        fake_start_datetime  = datetime.datetime.strptime(fake_start_datetime, '%m/%d/%Y:%H:%M:%S')
        delta_time = datetime.timedelta(minutes=search_window*num_of_episodes)
        end_time_datetime = fake_start_datetime + delta_time 
        clean_env(self.splunk_tools_instance, (fake_start_datetime.timestamp(), end_time_datetime.timestamp()))
        
        self.dt_manager = MockedDatetimeManager(fake_start_datetime=fake_start_datetime)
        end_time = self.dt_manager.get_fake_current_datetime()
        start_time = self.dt_manager.subtract_time(end_time, minutes=search_window)
        time_range = (start_time, end_time)
        self.time_range = time_range
        # update_rules_frequency_and_time_range(self.splunk_tools_instance, time_range)
        self.reward_calculator = None
        self.num_of_searches = num_of_searches
        self.time_range_logs_amount = []
        # run the saved searches for warmup
        for i in range(1, 3):
            logger.info(f"Running saved searches for warmup {i}")
            self.splunk_tools_instance.run_saved_searches_parallel(time_range)
        self.all_steps_counter = 0
        self.fake_states_list = []
        self.fake_states_path = r"resources/fake_states.csv"


    def calculate_quota(self, episode_logs_number, additional_percentage):
        self.total_additional_logs = additional_percentage*self.search_window*self.logs_per_minute
        # self.total_additional_logs = additional_percentage*episode_logs_number  
        self.step_size = int((self.total_additional_logs//self.search_window)*self.action_duration//60)
        self.remaining_quota = self.step_size
        self.action_strategy.quota = self.remaining_quota

    def set_reward_calculator(self, reward_calculator):
        self.reward_calculator = reward_calculator

    def get_reward(self):
 
        if self.done:
            logger.debug(self.done)
            if self.step_violation:
                reward = self.reward_calculator.get_step_violation_reward(self.time_range, self.state_strategy.real_state, self.state_strategy.fake_state, self.current_action, self.remaining_quota, self.step_counter)
                # reward = self.reward_calculator.get_step_violation_reward(self.time_range, self.state_strategy.real_state, self.state_strategy.fake_state, self.current_action, self.remaining_quota/self.total_additional_logs, self.step_counter)
            else:
                # logger.info(f"Current time: {self.dt_manager.set_fake_current_datetime(self.time_range[-1])}") # dont remove!!
                # self.update_state()
                violation_reward = self.reward_calculator.check_episodic_agent_violation(self.time_range, self.state_strategy.real_state, self.state_strategy.fake_state, self.current_action, self.remaining_quota, self.step_counter)
                if violation_reward != 0:
                    logger.info(f"violation reward: {violation_reward}")
                    reward = violation_reward
                    self.action_done = True
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        # Submit all actions to the thread pool
                        future_to_action = {
                            executor.submit(self.action_strategy.perform_action, action, time_range): (time_range, action) for time_range, action in self.action_auditor
                        }
                        for future in concurrent.futures.as_completed(future_to_action):
                            time_range, action = future_to_action[future]
                            try:
                                data = future.result()
                            except Exception as exc:
                                logger.error(f"Action {action} generated an exception: {exc}")
                            else:
                                logger.info(f"Action {action} was performed successfully")
                    
                    log_amount = sum([action[1][0] for action in self.action_auditor]) * self.remaining_quota
                    log_amount = max(log_amount, 1)
                    
                    time_to_wait = int(np.log(log_amount))
                    logger.info(f"Waiting {time_to_wait} secondes till logs are indexed")
                    time.sleep(time_to_wait)
                    # time.sleep(30)
                    reward = self.reward_calculator.get_full_reward(self.time_range, self.state_strategy.real_state, self.state_strategy.fake_state, self.current_action, self.remaining_quota, self.step_counter)
                    self.fake_states_list.append(self.state_strategy.abs_fake)
        else:

            # reward = self.reward_calculator.get_partial_reward(self.state_strategy.step_real_state, self.state_strategy.step_fake_state, self.current_action, self.step_counter)
            reward = self.reward_calculator.get_partial_reward(self.state_strategy.real_state, self.state_strategy.fake_state, self.current_action, self.step_counter)
            
                
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
        # log the gloabal step number
        logger.info(f"global step number: {self.all_steps_counter}")
        logger.info(f"action: {action}")
        self.all_steps_counter += 1
        self.current_action = action
        # if self.step_counter == 1:
        #     self.reward_calculator.get_no_agent_reward(self.time_range)
        logger.info(f"step number: {self.step_counter}")
        time_range = self.dt_manager.get_time_range_action(self.action_duration)     
        self.action_auditor.append((time_range, action))   
        self.action_strategy.record_action(action)
        self.remaining_quota = self.action_strategy.remaining_quota
        logger.info(f"Remaining quota: {self.remaining_quota}")
        logger.debug(f"Current time: {self.dt_manager.set_fake_current_datetime(time_range[-1])}") # dont remove!!
        self.update_state()  
        self.reward_calculator.get_partial_reward_values(self.state_strategy.real_state, self.state_strategy.fake_state)
        if self.check_step_violation():
            self.step_violation = True
            self.done = True
        if self.check_done():
            self.done = True
        reward = self.get_reward()
        self.action_strategy.reset_step()
        logger.info(f"########################################################################################################################")
        self.step_counter += 1
        return self.state, reward, self.done, {}
    
    def check_step_violation(self):
        if self.remaining_quota > 1.5:
            self.action_done = True
            return True
        return False

    # def check_step_violation(self):
    #     if self.remaining_quota < 0:
    #         self.action_done = True
    #         return True
    #     return False

    def set_max_actions_value(self, max_actions_value): 
        self.step_size = max_actions_value

    def get_fake_distribution(self):
        fake_logtypes_counter = {}
        for i, logtype in enumerate(self.relevant_logtypes):
            logsource = logtype[0].lower()
            eventcode = str(logtype[1])
            if isinstance(self.action_strategy, ActionStrategy8):
                fake_logtypes_counter[(logsource, eventcode)] = self.action_strategy.current_episode_accumulated_action[i]
            else:  
                # fake_logtypes_counter[f"{logsource} {eventcode}"] = self.action_strategy.current_episode_accumulated_action[i*2]
                fake_logtypes_counter[(logsource, eventcode)] = self.action_strategy.current_step_action[i*2]
                if i == len(self.relevant_logtypes)-1:
                    break
                # fake_logtypes_counter[f"{logsource} {eventcode}"] += self.action_strategy.current_episode_accumulated_action[i*2+1]
                fake_logtypes_counter[(logsource, eventcode)] += self.action_strategy.current_step_action[i*2+1]
        return fake_logtypes_counter
    
    
    def update_state(self):
        now = self.dt_manager.get_fake_current_datetime()
        previous_now = self.dt_manager.subtract_time(now, seconds=self.action_duration)
        step_real_logtypes_counter = self.splunk_tools_instance.get_real_distribution(previous_now, now)
        step_fake_logtypes_counter = self.get_fake_distribution()
        self.state_strategy.update_distributions(step_real_logtypes_counter, step_fake_logtypes_counter)
        self.state_strategy.update_quota(self.remaining_quota/self.total_additional_logs)
        if isinstance(self.state_strategy, StateStrategy6):
            # get week day and hour of now (check if need to convert to datetime object)
            datetime_now = datetime.datetime.strptime(previous_now, '%m/%d/%Y:%H:%M:%S') 
            week_day = datetime_now.weekday()
            hour = datetime_now.hour
            self.state_strategy.update_time(week_day, hour)
            self.state_strategy.update_episodic_action(np.array(self.action_strategy.current_episode_accumulated_action)/self.total_additional_logs)
        if isinstance(self.state_strategy, StateStrategy7):
            rules_alert = np.zeros(len(self.splunk_tools_instance.active_saved_searches), dtype=int)
            if isinstance(self.action_strategy, ActionStrategy7):
                current_action = self.current_action[1:]*self.current_action[0]
            else:
                current_action = self.current_action
            for i, log_type in enumerate(current_action):
                if i%2 == 1:
                    if log_type > 0:
                        rules_alert[i//2] = 1
            self.state_strategy.update_rules_alerts(rules_alert)
            datetime_now = datetime.datetime.strptime(previous_now, '%m/%d/%Y:%H:%M:%S') 
            week_day = datetime_now.weekday()
            hour = datetime_now.hour
            self.state_strategy.update_time(week_day, hour)
            

        self.state = self.state_strategy.update_state()
        # log somtimes the state
        if random.random() < 0.3:
            logger.info(f"state: {self.state}")
        # logger.info(f"state: {self.state}")
        return self.state
        
    def check_done(self):
        # Define the termination conditions based on the current state or other criteria
        # compare distributions
        # if self.reward_calculator.current_distributions_distance > self.reward_calculator.distribution_threshold:
        #     return True
        # if self.step_counter == self.total_steps or self.action_done or self.remaining_quota == 0:
        if self.step_counter == self.total_steps or self.action_done:
            logger.info(f"done: {self.step_counter == self.total_steps} {self.action_done} {self.remaining_quota == 0}")
            return True
        else:
            return False
        
    def reset(self):
        logger.info("#############################################################\n###########################################################")
        logger.info("resetting")
        # if self.action_strategy.should_delete:
        #     logger.info("Deleting logs from the environment")
        #     clean_env(self.splunk_tools_instance, self.time_range)
        # if self.reward_calculator.current_distributions_distance <= self.reward_calculator.distribution_threshold and self.remaining_quota >= 0:
        #     self.reward_calculator.get_no_agent_reward(self.time_range)   
        self.action_per_episode.append(self.action_strategy.current_episode_accumulated_action)
        self.current_log_type = 0
        self.done = False
        self.step_counter = 1
        self.real_distribution = np.zeros(len(self.top_logtypes))
        self.splunk_tools_instance.real_logtypes_counter = {}
        # create random date from the last 90 days
        # time = self.dt_manager.get_random_datetime() 
        # self.update_timerange(time)
        new_start_time = self.get_new_start_time()
        self.update_timerange(new_start_time)
        
        logger.info(f"Current time: {self.dt_manager.set_fake_current_datetime(self.time_range[0])}") 
        self.action_auditor = []
        self.action_strategy.reset()
        self.action_done = False
        self.step_violation = False
        
        # self.reward_calculator.get_no_agent_reward(self.time_range)   

        # get the amount of logs in the time range
        log_amount = self.splunk_tools_instance.get_logs_amount(self.time_range)
        self.time_range_logs_amount.append(log_amount)
        self.calculate_quota(log_amount, self.additional_percentage)
        logger.info(f"Time range logs amount: {self.time_range_logs_amount[-1]}")
        # self.remaining_quota = self.total_additional_logs
        self.done = False
        
        if isinstance(self.state_strategy, StateStrategy6) or isinstance(self.state_strategy, StateStrategy7):
            datetime_now = datetime.datetime.strptime(new_start_time, '%m/%d/%Y:%H:%M:%S') 
            week_day = datetime_now.weekday()
            hour = datetime_now.hour
            self.state_strategy.update_time(week_day, hour)
        if isinstance(self.state_strategy, StateStrategy10) or isinstance(self.state_strategy, StateStrategy11):
             self.state_strategy.update_log_number(log_amount)
             self.quota = log_amount
        self.state = self.state_strategy.reset()
        if random.random() < 0.1:
            empty_monitored_files(r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/wineventlog:security.txt")
            empty_monitored_files(r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/wineventlog:system.txt")
        if self.search_window > self.rule_frequency:
            logger.info('Cleaning the environment')
            clean_env(self.splunk_tools_instance, self.time_range)
        return self.state 
    
    def get_new_start_time(self):
        if self.action_done:
            return self.time_range[0]
        # elif self.time_range[1] in self.problematic_time_ranges:
        #     return self.dt_manager.add_time(self.time_range[0], minutes=2*self.rule_frequency)
        else:
            return self.dt_manager.add_time(self.time_range[0], minutes=self.rule_frequency)

    def update_timerange(self, new_start_time):
        # new_start_time = self.time_range[1]
        # new_start_time = self.time_range[1]
        new_end_time = self.dt_manager.add_time(new_start_time, minutes=self.search_window)
        logger.info(f'current time_range: {self.time_range}')
        self.time_range = (new_start_time, new_end_time)
        logger.info(f'new time_range: {self.time_range}')       
            
    def render(self, mode='human'):
        logger.info(f"Current state: {self.state}")

    
    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]  

        
# if __name__=="__main__":
#     # test the framework
