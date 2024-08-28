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
from reward_calculators.reward_calculator_base import RewardCalc
import random
import subprocess
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import gym
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
        self.model = model

    def update_running_time(self, running_time, env_file_path):
        command = f'sed -i "s/RUNNING_TIME=.*/RUNNING_TIME={running_time}/" "{env_file_path}"'
        res = subprocess.run(command, shell=True, capture_output=True, text=True)
        logger.info(res.stdout)
        logger.error(res.stderr)

    def choose_random_rules(self, splunk_tools_instance, num_of_searches, is_get_only_enabled=True):
        logger.info('enable random rules')
        savedsearches = splunk_tools_instance.get_saved_search_names(get_only_enabled=is_get_only_enabled)
        # random_savedsearch = random.sample(savedsearches, num_of_searches)
        random_savedsearch = ['ESCU Network Share Discovery Via Dir Command Rule', 'Detect New Local Admin account',"Known Services Killed by Ransomware", "Non Chrome Process Accessing Chrome Default Dir"]
        for savedsearch in savedsearches:
            if savedsearch not in random_savedsearch:
                splunk_tools_instance.disable_search(savedsearch)
            else:
                splunk_tools_instance.enable_search(savedsearch)
        return random_savedsearch

    def update_rules_frequency_and_time_range(self, splunk_tools_instance, rule_frequency, time_range):
        logger.info('update rules frequency')
        splunk_tools_instance.update_all_searches(splunk_tools_instance.update_search_cron_expression, f'*/{rule_frequency} * * * *')
        logger.info('update time range of rules')
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
        with open(f'{current_dir}/{mode}/action_dict.json', 'w') as fp:
                json.dump(np.array(env.action_per_episode).tolist(), fp)

    def empty_monitored_files(self, monitored_file_path):
        logger.info(f'empty the monitored file {monitored_file_path}')
        with open(monitored_file_path, 'w') as fp:
            fp.write('')
            
    def clean_env(self, splunk_tools_instance, time_range=None):
        if time_range is None:
            time_range = ("04/29/2023:00:00:00","05/02/2023:00:00:00")
            splunk_tools_instance.delete_fake_logs(time_range)
            self.empty_monitored_files(r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/wineventlog:security.txt")
            self.empty_monitored_files(r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/wineventlog:system.txt")
            return time_range
        date = time_range[1].split(':')[0]
        time_range = (f'{date}:00:00:00', f'{date}:23:59:59')
        splunk_tools_instance.delete_fake_logs(time_range)
        self.empty_monitored_files(r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/wineventlog:security.txt")
        self.empty_monitored_files(r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/wineventlog:system.txt")
        return time_range
    
    def setup_environment(self, parameters):
        # create a list of relevant logtypes
        num_of_searches = parameters['num_of_searches']
        rule_frequency = parameters['rule_frequency']
        search_window = parameters['search_window']
        span_size = parameters['span_size']
        running_time = parameters['running_time']
        env_file_path = parameters['env_file_path']
        fake_start_datetime = parameters['fake_start_datetime']
        is_get_only_enabled = parameters['is_get_only_enabled']
        # savedsearches = parameters['savedsearches']
        fake_start_datetime  = datetime.datetime.strptime(fake_start_datetime, '%m/%d/%Y:%H:%M:%S')
        # create a datetime manager instance
        dt_manager = MockedDatetimeManager(fake_start_datetime=fake_start_datetime)
        end_time = dt_manager.get_fake_current_datetime()
        start_time = dt_manager.subtract_time(end_time, minutes=search_window)
        time_range = (start_time, end_time)
        # create a splunk tools instance
        splunk_tools_instance = SplunkTools()
        self.clean_env(splunk_tools_instance)
        self.update_running_time(running_time, env_file_path)
        self.update_rules_frequency_and_time_range(splunk_tools_instance, rule_frequency, time_range)
        savedsearches = sorted(self.choose_random_rules(splunk_tools_instance, num_of_searches, is_get_only_enabled=is_get_only_enabled))
        parameters['savedsearches'] = savedsearches
        # savedsearches = ['Modification of Executable File', 'Monitor for New Service Installs', 'Monitor for Suspicious Network IP’s', 'Multiple Network Connections to Same Port on External Hosts']
        relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]})) #[(x[0], str(x[1])) for x in state_span]
        relevant_logtypes.append(('wineventlog:security', '4624'))
        # relevant_logtypes = [('wineventlog:security', '2005'), ('wineventlog:security', '4625'), ('wineventlog:security', '2004'), ('xmlwineventlog:microsoft-windows-sysmon/operational', '3'), ('wineventlog:security', '4688'), ('xmlwineventlog:microsoft-windows-sysmon/operational', '8')]
        # relevant_logtypes = [logtype for logtype in logtypes if logtype not in section_logtypes]
        # relevant_logtypes = state_span
        # non_relevant_logtypes = random.sample([logtype for logtype in logtypes if logtype not in relevant_logtypes], len(relevant_logtypes))
        parameters['relevant_logtypes'] = relevant_logtypes #+ non_relevant_logtypes
        if "total_additional_logs" in parameters:
            total_additional_logs = parameters['total_additional_logs']
        else:
            total_additional_logs = None
        alpha, beta, gamma = parameters['alpha'], parameters['beta'], parameters['gamma']
        log_generator_instance = LogGenerator(relevant_logtypes, big_replacement_dicts, splunk_tools_instance)
        measurment_tool = Measurement(splunk_tools_instance, num_of_searches, measure_energy=parameters['measure_energy'])
        reward_calculator_instance = RewardCalc(relevant_logtypes, dt_manager, splunk_tools_instance, rule_frequency, num_of_searches, measurment_tool, alpha, beta, gamma)
        logger.info(f'current parameters:\ntime range:{time_range} \nfake_start_datetime: {fake_start_datetime}\nrule frequency: {rule_frequency}\nsearch_window:{search_window}\nrunning time: {running_time}\nnumber of searches: {num_of_searches}\nalpha {alpha}\nbeta {beta}\n gama {gamma}\nsavedsearches: {savedsearches}\nrelevantlog_types: {relevant_logtypes}')
        gym.register(
        id='splunk_attack-v0',
        entry_point='framework:Framework',  # Replace with the appropriate path       
        )
        env = gym.make('splunk_attack-v0', log_generator_instance = log_generator_instance, splunk_tools_instance = splunk_tools_instance, reward_calculator_instance = reward_calculator_instance, dt_manager=dt_manager, time_range=time_range, rule_frequency=rule_frequency, search_window=search_window, relevant_logtypes=relevant_logtypes, span_size=span_size, total_additional_logs=total_additional_logs)

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
        logger.info('train the model')
        parameters['measure_energy'] = False
        env = self.setup_environment(parameters)
        # save reward_calculator.py to the experiment directory
        with open(f'{self.experiment_dir}/reward_calculator.py', 'w') as fp:
            with open(r'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/reward_calculator.py', 'r') as fp2:
                fp.write(fp2.read())
        new_logger = configure(f"{self.experiment_dir}/tensorboard/", ["stdout", "csv", "tensorboard"])
        self.save_parameters_to_file(parameters, f'{self.experiment_dir}/parameters_train.json')
        model = self.model(MlpPolicy, env, verbose=1, stats_window_size=5, tensorboard_log=f"{self.experiment_dir}/tensorboard/", ent_coef=0.02)
        model.set_logger(new_logger)
        # model = DQN(env=env, policy=CustomPolicy)
        model.learn(total_timesteps=parameters['episodes']*env.total_steps, log_interval=10, tb_log_name=logger.name)
        model.save(f"{self.experiment_dir}/splunk_attack")
        self.save_assets(self.experiment_dir, env)
        return model
    
    def retrain_model(self, parameters):
        logger.info('retrain the model')
        parameters['measure_energy'] = False
        env = self.load_environment({'limit_learner':False})
        model = self.model.load(f"{self.experiment_dir}/splunk_attack")
        model.set_env(env)
        model.learn(total_timesteps=parameters['episodes']*env.total_steps)
        model.save(f"{self.experiment_dir}/splunk_attack")
        self.save_assets(self.experiment_dir, env)
        return model
    
    def test_model(self, num_of_episodes):
        logger.info('test the model')
        model = self.model.load(f"{self.experiment_dir}/splunk_attack")
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
