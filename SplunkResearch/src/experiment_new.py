import os
import pickle
import numpy as np
import urllib3
from measurement import Measurement
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
from reward_calculator import RewardCalc
import random
import subprocess
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import custom_splunk
from log_generator import LogGenerator
from resources.log_generator_resources import replacement_dicts as big_replacement_dicts
urllib3.disable_warnings()
from stable_baselines3.common.logger import configure
import logging
logger = logging.getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Experiment:
    def __init__(self, experiment_dir, model=None):
        self.experiment_dir = experiment_dir



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
        with open(f'{current_dir}/{mode}/action_dict.json', 'w') as fp:
                json.dump(np.array(env.action_per_episode).tolist(), fp)


    
    def setup_environment(self, parameters):       
        if "total_additional_logs" in parameters:
            total_additional_logs = parameters['total_additional_logs']
        else:
            total_additional_logs = None
        # logger.info(f'current parameters:\ntime range:{time_range} \nfake_start_datetime: {fake_start_datetime}\nrule frequency: {rule_frequency}\nsearch_window:{search_window}\nrunning time: {running_time}\nnumber of searches: {num_of_searches}\nalpha {alpha}\nbeta {beta}\n gama {gamma}\nsavedsearches: {savedsearches}\nrelevantlog_types: {relevant_logtypes}')
        env = gym.make(parameters['env_name'],  fake_start_datetime=parameters['fake_start_datetime'], total_additional_logs=total_additional_logs, reward_parameters=parameters['reward_parameters'], is_measure_energy=parameters['measure_energy'])
        return env
    
    def load_environment(self, modifed_parameters={}):
        parameters = self.load_parameters(f'{self.experiment_dir}/parameters_train.json')
        parameters.update(modifed_parameters)
        env = self.setup_environment(parameters)
        return env
    
    def save_parameters_to_file(self, parameters, filename):
        with open(filename, 'w') as fp:
            json.dump(parameters, fp)
        # save reward_calculator.py to the experiment directory
        with open(f'{self.experiment_dir}/reward_calculator.py', 'w') as fp:
            with open(r'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/reward_calculator.py', 'r') as fp2:
                fp.write(fp2.read())
    
    def load_parameters(self, filename):
        with open(filename, 'r') as fp:
            parameters = json.load(fp)
        return parameters

    def train_model(self, parameters, model_object, episodes=1):
        logger.info('train the model')
        parameters['measure_energy'] = False
        env = self.setup_environment(parameters)

        self.save_parameters_to_file(parameters, f'{self.experiment_dir}/parameters_train.json')
        model = model_object(MlpPolicy, env, verbose=1, stats_window_size=5, tensorboard_log=f"{self.experiment_dir}/tensorboard/", ent_coef=0.02)

        model.learn(total_timesteps=episodes*env.total_steps, tb_log_name=logger.name)
        model.save(f"{self.experiment_dir}/splunk_attack")
        self.save_assets(self.experiment_dir, env)
        return model
    
    def retrain_model(self, parameters, model_object):
        logger.info('retrain the model')
        parameters['measure_energy'] = False
        env = self.load_environment()
        model = model_object.load(f"{self.experiment_dir}/splunk_attack")
        model.set_env(env)
        model.learn(total_timesteps=parameters['episodes']*env.total_steps)
        model.save(f"{self.experiment_dir}/splunk_attack")
        self.save_assets(self.experiment_dir, env)
        return model
    
    def test_model(self, num_of_episodes, model_object):
        logger.info('test the model')
        model = model_object.load(f"{self.experiment_dir}/splunk_attack")
        env = self.load_environment({'measure_energy': True})
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_of_episodes)
        logger.info(f"mean_reward:{mean_reward}, std_reward:{std_reward}")
        self.save_assets(self.experiment_dir, env, mode='test')
        return env
    
    def test_baseline_agent(self, num_of_episodes, agent_type='random'):
        logger.info(f'test baseline {agent_type} agent')
        env = self.load_environment({'measure_energy': True})
        for i in range(num_of_episodes):
            env.reset()
            self.run_manual_episode(agent_type, env)
        self.save_assets(self.experiment_dir, env, mode=f'test_{agent_type}_agent')
        return env

            
    def test_no_agent(self, num_of_episodes):
        logger.info('test no agent')
        env = self.load_environment({'measure_energy': True})
        for i in range(num_of_episodes):
            env.reset()
            env.evaluate_no_agent()
        self.save_assets(self.experiment_dir, env, mode='test_no_agent')
        return env
    
    def run_manual_episode(self, agent_type, env):
        done = False
        while not done:
            if agent_type == 'random':
                # action = np.array([random.uniform(0, 1) for i in range((len(env.relevant_logtypes)-1)*2+2)])
                action = np.random.dirichlet(np.ones(env.action_space.shape))
            elif agent_type == 'uniform':
                action = 100/len(env.relevant_logtypes)    
            obs, reward, done, info = env.step(action)
            env.render()
