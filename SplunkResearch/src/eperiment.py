import os
import pickle
import urllib3
from SplunkResearch.src.datetime_manager import MockedDatetimeManager
import datetime
import time
import json
import sys
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch')
from resources.logtypes import logtypes
from resources.section_logtypes import section_logtypes
import logging
from experiment_manager import ExperimentManager
from splunk_tools import SplunkTools
import random
import subprocess
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import gym
from stable_baselines3 import A2C
from log_generator import LogGenerator
from resources.log_generator_resources import replacement_dicts as big_replacement_dicts
urllib3.disable_warnings()


class Experiment:
    def __init__(self, experiment_dir, logger):
        self.experiment_dir = experiment_dir
        self.logger = logger

    def update_running_time(self, running_time, env_file_path):
        command = f'sed -i "s/RUNNING_TIME=.*/RUNNING_TIME={running_time}/" "{env_file_path}"'
        res = subprocess.run(command, shell=True, capture_output=True, text=True)
        self.logger.info(res.stdout)
        self.logger.error(res.stderr)

    def choose_random_rules(self, splunk_tools_instance, num_of_searches, get_only_enabled=True):
        self.logger.info('enable random rules')
        savedsearches = splunk_tools_instance.get_saved_search_names(get_only_enabled=get_only_enabled)
        random_savedsearch = random.sample(savedsearches, num_of_searches)
        for savedsearch in savedsearches:
            if savedsearch not in random_savedsearch:
                splunk_tools_instance.disable_search(savedsearch)
            else:
                splunk_tools_instance.enable_search(savedsearch)
        return random_savedsearch

    def update_rules_frequency_and_time_range(self, splunk_tools_instance, rule_frequency, time_range):
        self.logger.info('update rules frequency')
        splunk_tools_instance.update_all_searches(splunk_tools_instance.update_search_cron_expression, f'*/{rule_frequency} * * * *')
        self.logger.info('update time range of rules')
        splunk_tools_instance.update_all_searches(splunk_tools_instance.update_search_time_range, time_range)   

    def save_assets(self, current_dir, env, mode='train'):
        print(env.reward_calculator.reward_values_dict)
        # create a directory for the assets if not exists
        if not os.path.exists(f'{current_dir}/{mode}'):
            os.makedirs(f'{current_dir}/{mode}')
        with open(f'{current_dir}/{mode}/reward_dict.json', 'w') as fp:
            json.dump(env.reward_calculator.reward_dict, fp)
        with open(f'{current_dir}/{mode}/reward_values_dict.json', 'w') as fp:
            json.dump(env.reward_calculator.reward_values_dict, fp)
        with open(f'{current_dir}/{mode}/time_rules_energy.json', 'w') as fp:
            json.dump(env.reward_calculator.time_rules_energy, fp)   
        with open(f'{current_dir}/{mode}/action_dict_{mode}.json', 'w') as fp:
                json.dump(env.time_action_dict, fp)

    
    def clean_env(self, splunk_tools_instance, time_range):
        date = time_range[1].split(':')[0]
        time_range = (f'{date}:00:00:00', f'{date}:23:59:59')
        splunk_tools_instance.delete_fake_logs(time_range)
        return time_range
    
    def setup_environment(self, parameters):
        # create a list of relevant logtypes
        num_of_searches = parameters['num_of_searches']
        rule_frequency = parameters['rule_frequency']
        search_window = parameters['search_window']
        reward_parameter_dict = parameters['reward_parameter_dict']
        max_actions_value = parameters['max_actions_value']
        fake_start_datetime = parameters['fake_start_datetime']
        end_time = dt_manager.get_fake_current_datetime()
        start_time = dt_manager.subtract_time(end_time, minutes=search_window)
        time_range = (start_time, end_time)
        # create a datetime manager instance
        dt_manager = MockedDatetimeManager(fake_start_datetime=fake_start_datetime, log_file_path="test.log")
        # create a splunk tools instance
        splunk_tools_instance = SplunkTools(logger=self.logger)
        
        savedsearches = sorted(self.choose_random_rules(splunk_tools_instance, num_of_searches, get_only_enabled=True))
        relevant_logtypes =  list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]})
        relevant_logtypes = [logtype for logtype in logtypes if logtype not in section_logtypes]
        log_generator_instance = LogGenerator(relevant_logtypes, big_replacement_dicts, splunk_tools_instance)
        gym.register(
        id='splunk_attack-v0',
        entry_point='framework:Framework',  # Replace with the appropriate path       
        )
        env = gym.make('splunk_attack-v0', log_generator_instance = log_generator_instance, splunk_tools_instance = splunk_tools_instance, dt_manager=dt_manager, logger=self.logger, time_range=time_range, rule_frequency=rule_frequency, search_window=search_window, reward_parameter_dict=reward_parameter_dict, relevant_logtypes=relevant_logtypes, max_actions_value=max_actions_value)
        self.save_parameters_to_file(parameters, f'{self.experiment_dir}/parameters_train.json')
        return env
    
    def load_environment(self):
        parameters = self.load_parameters(f'{self.experiment_dir}/parameters_train.json')
        env = self.setup_environment(parameters)
        return env
    
    def save_parameters_to_file(self, parameters, filename):
        with open(filename, 'w') as fp:
            json.dump(parameters, fp)
    
    def load_parameters(self, filename):
        with open(filename, 'r') as fp:
            parameters = json.load(fp)
        return parameters

    def train_model(self, parameters):
        env = self.setup_environment(parameters)
        model = A2C(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=parameters['episodes']*len(parameters['relevant_logtypes']))
        model.save(f"{self.experiment_dir}/A2C_splunk_attack")
        self.save_assets(self.experiment_dir, env)
        return model
    
    def test_model(self, num_of_episodes):
        self.logger.info('test the model')
        model = A2C.load(f"{self.experiment_dir}/A2C_splunk_attack")
        env = self.load_environment()
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_of_episodes)
        self.logger.info(f"mean_reward:{mean_reward}, std_reward:{std_reward}")
        self.save_assets(self.experiment_dir, env, mode='test')
        return env
    
    def test_baseline_agent(self, num_of_episodes, agent_type='random'):
        env = self.load_environment()
        for i in range(num_of_episodes):
            env.reset()
            done = False
            self.run_manual_episode(agent_type, env, done)
        self.save_assets(self.experiment_dir, env, mode=f'test_{agent_type}_agent')
        return env

    def run_manual_episode(self, agent_type, env, done):
        while not done:
            if agent_type == 'random':
                action = random.uniform(0, (100-env.sum_of_fractions))
            elif agent_type == 'passive':
                action = 0
            elif agent_type == 'uniform':
                action = 100/len(env.relevant_logtypes)                    
            obs, reward, done, info = env.blind_step(action)
            env.render()
    

          

    # def retrain_model(self):
