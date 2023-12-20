import os
import pickle
import numpy as np
import urllib3

from datetime_manager import MockedDatetimeManager
import datetime
import time
import json
import sys
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch')
from resources.logtypes import logtypes
from resources.state_span import state_span
from resources.section_logtypes import section_logtypes
import logging
from experiment_manager import ExperimentManager
from splunk_tools import SplunkTools
from policy_network import CustomPolicy
from reward_calculator import RewardCalc
import random
import subprocess
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import gym
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

    def choose_random_rules(self, splunk_tools_instance, num_of_searches, is_get_only_enabled=True):
        self.logger.info('enable random rules')
        savedsearches = splunk_tools_instance.get_saved_search_names(get_only_enabled=is_get_only_enabled)
        # random_savedsearch = random.sample(savedsearches, num_of_searches)
        random_savedsearch = ['Monitor for Additions to Firewall Rules', 'Monitor for Changes to Firewall Rules', 'Monitor for Suspicious_Administrative Processes', 'Multiple Failed Logins from the Same Source', 'Multiple Network Connections to Same Port on External Hosts', 'Suspicious Remote Thread Creation']
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
        # print(env.reward_calculator.reward_values_dict)
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

    
    def clean_env(self, splunk_tools_instance, time_range=None):
        if time_range is None:
            splunk_tools_instance.delete_fake_logs()
            return time_range
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
        running_time = parameters['running_time']
        env_file_path = parameters['env_file_path']
        fake_start_datetime = parameters['fake_start_datetime']
        is_get_only_enabled = parameters['is_get_only_enabled']
        if 'limit_learner' in parameters:
            limit_learner = parameters['limit_learner']
        else:
            limit_learner = True
        if 'distribution_learner' in parameters:
            distribution_learner = parameters['distribution_learner']
        else:
            distribution_learner = True
        # savedsearches = parameters['savedsearches']
        fake_start_datetime  = datetime.datetime.strptime(fake_start_datetime, '%m/%d/%Y:%H:%M:%S')
        # create a datetime manager instance
        dt_manager = MockedDatetimeManager(fake_start_datetime=fake_start_datetime, log_file_path="test.log")
        end_time = dt_manager.get_fake_current_datetime()
        start_time = dt_manager.subtract_time(end_time, minutes=search_window)
        time_range = (start_time, end_time)
        # create a splunk tools instance
        splunk_tools_instance = SplunkTools(logger=self.logger)
        self.clean_env(splunk_tools_instance)
        self.update_running_time(running_time, env_file_path)
        self.update_rules_frequency_and_time_range(splunk_tools_instance, rule_frequency, time_range)
        savedsearches = sorted(self.choose_random_rules(splunk_tools_instance, num_of_searches, is_get_only_enabled=is_get_only_enabled))
        parameters['savedsearches'] = savedsearches
        # savedsearches = ['Modification of Executable File', 'Monitor for New Service Installs', 'Monitor for Suspicious Network IP’s', 'Multiple Network Connections to Same Port on External Hosts']
        relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]})) #[(x[0], str(x[1])) for x in state_span]
        # relevant_logtypes = [('wineventlog:security', '2005'), ('wineventlog:security', '4625'), ('wineventlog:security', '2004'), ('xmlwineventlog:microsoft-windows-sysmon/operational', '3'), ('wineventlog:security', '4688'), ('xmlwineventlog:microsoft-windows-sysmon/operational', '8')]
        # relevant_logtypes = [logtype for logtype in logtypes if logtype not in section_logtypes]
        parameters['relevant_logtypes'] = relevant_logtypes
        log_generator_instance = LogGenerator(relevant_logtypes, big_replacement_dicts, splunk_tools_instance)
        reward_calculator_instance = RewardCalc(relevant_logtypes, dt_manager, self.logger, splunk_tools_instance, rule_frequency, num_of_searches, distribution_learner)
        print("Debugging: 6")
        self.logger.info(f'current parameters:\ntime range:{time_range} \nfake_start_datetime: {fake_start_datetime}\nrule frequency: {rule_frequency}\nsearch_window:{search_window}\nrunning time: {running_time}\nnumber of searches: {num_of_searches}\nmax action value: {max_actions_value}\nreward parameter dict: {reward_parameter_dict}\nsavedsearches: {savedsearches}\nrelevantlog_types: {relevant_logtypes}')
        gym.register(
        id='splunk_attack-v0',
        entry_point='framework:Framework',  # Replace with the appropriate path       
        )
        env = gym.make('splunk_attack-v0', log_generator_instance = log_generator_instance, splunk_tools_instance = splunk_tools_instance, reward_calculator_instance = reward_calculator_instance, dt_manager=dt_manager, logger=self.logger, time_range=time_range, rule_frequency=rule_frequency, search_window=search_window, relevant_logtypes=relevant_logtypes, limit_learner=limit_learner, max_actions_value=max_actions_value)
        
        # Debugging print statement
        print("Debugging: 10")
        
        return env
    
    def load_environment(self, modifed_parameters={}):
        parameters = self.load_parameters(f'{self.experiment_dir}/parameters_train.json')
        parameters.update(modifed_parameters)
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
        self.logger.info('train the model')
        env = self.setup_environment(parameters)
        # save reward_calculator.py to the experiment directory
        with open(f'{self.experiment_dir}/reward_calculator.py', 'w') as fp:
            with open(r'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/reward_calculator.py', 'r') as fp2:
                fp.write(fp2.read())
        self.save_parameters_to_file(parameters, f'{self.experiment_dir}/parameters_train.json')
        model = A2C(MlpPolicy, env, verbose=1, ent_coef=0.01)
        # model = DQN(env=env, policy=CustomPolicy)
        model.learn(total_timesteps=parameters['episodes']*len(parameters['relevant_logtypes']))
        model.save(f"{self.experiment_dir}/splunk_attack")
        self.save_assets(self.experiment_dir, env)
        return model
    
    def retrain_model(self, parameters):
        self.logger.info('retrain the model')
        env = self.load_environment({'limit_learner':False})
        model = A2C.load(f"{self.experiment_dir}/splunk_attack")
        model.set_env(env)
        model.learn(total_timesteps=parameters['episodes']*len(env.relevant_logtypes))
        model.save(f"{self.experiment_dir}/splunk_attack")
        self.save_assets(self.experiment_dir, env)
        return model
    
    def test_model(self, num_of_episodes):
        self.logger.info('test the model')
        model = A2C.load(f"{self.experiment_dir}/splunk_attack")
        env = self.load_environment()
        env.limit_learner_turn_off()
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_of_episodes)
        self.logger.info(f"mean_reward:{mean_reward}, std_reward:{std_reward}")
        self.save_assets(self.experiment_dir, env, mode='test')
        return env
    
    def test_baseline_agent(self, num_of_episodes, agent_type='random'):
        self.logger.info(f'test baseline {agent_type} agent')
        if agent_type == 'passive':
            env = self.load_environment(modifed_parameters={'max_actions_value':0})
        else:
            env = self.load_environment()
        env.limit_learner_turn_off()
        for i in range(num_of_episodes):
            env.reset()
            done = False
            self.run_manual_episode(agent_type, env, done)
        self.save_assets(self.experiment_dir, env, mode=f'test_{agent_type}_agent')
        return env

    def run_manual_episode(self, agent_type, env, done):
        while not done:
            if agent_type == 'random':
                action = np.random.dirichlet(np.ones(len(env.relevant_logtypes)),size=1)[0]
            elif agent_type == 'passive':
                action = np.zeros(len(env.relevant_logtypes))
            elif agent_type == 'uniform':
                action = 100/len(env.relevant_logtypes)    
            if agent_type != 'passive':
                obs, reward, done, info = env.step(action)
            else:                           
                obs, reward, done, info = env.blind_step(action)
            env.render()
    
