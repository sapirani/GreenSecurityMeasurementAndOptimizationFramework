import logging
import os
import pickle
import shutil
import datetime
import smtplib
from email.message import EmailMessage
import ssl
from dotenv import load_dotenv
import os
load_dotenv('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/.env')
import datetime
import os
import inspect
import numpy as np
import urllib3
import json
import sys
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch')
import logging
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import custom_splunk #dont remove!!!
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C, PPO, DQN
urllib3.disable_warnings()
from stable_baselines3.common.logger import configure
from env_utils import *
from measurement import Measurement
from reward_strategy import *
from pathlib import Path
import logging
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import HParam
from callbacks import *
logger = logging.getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model_names = {'a2c': A2C, 'ppo': PPO, 'dqn': DQN, 'recurrentppo': RecurrentPPO}
policy_names = {'mlp': MlpPolicy, 'lstm': MlpLstmPolicy}
# Dynamically find all reward calculator classes
RewardCalc_classes = {}
for name, obj in inspect.getmembers(sys.modules['reward_calculators'], inspect.isclass):
    if issubclass(obj, RewardStrategy) and obj is not RewardStrategy:
        RewardCalc_classes[name.split("RewardCalc")[1]] = obj

logger.info(f"Loaded RewardCalc_classes: {RewardCalc_classes}")


        
class ExperimentManager:
    
    def __init__(self, base_dir="experiments___", log_level=logging.INFO):
        self.log_level = log_level
        self.base_dir = base_dir
        self.train_dir = os.path.join(self.base_dir, 'train')
        self.eval_dir = os.path.join(self.base_dir, 'eval')
        self.retrain_model_dir = os.path.join(self.base_dir, 'retrain')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.train_logs_dir = os.path.join(self.train_dir, 'logs')
        self.eval_logs_dir = os.path.join(self.eval_dir, 'logs')
        self.retrain_model_logs_dir = os.path.join(self.retrain_model_dir, 'logs')
        self.train_tensorboard_dir = os.path.join(self.train_dir, 'tensorboard')
        self.eval_tensorboard_dir = os.path.join(self.eval_dir, 'tensorboard') 
        self.retrain_model_tensorboard_dir = os.path.join(self.retrain_model_dir, 'tensorboard')
        self.experiment_master_tables_dir = os.path.join(self.base_dir, 'experiment_master_tables')
        self.no_agent_baseline_experiment_dir = os.path.join(self.base_dir, 'no_agent_baseline')
        
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        Path(self.train_dir).mkdir(parents=True, exist_ok=True)
        Path(self.eval_dir).mkdir(parents=True, exist_ok=True)
        Path(self.retrain_model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.train_logs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.eval_logs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.retrain_model_logs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.train_tensorboard_dir).mkdir(parents=True, exist_ok=True)
        Path(self.eval_tensorboard_dir).mkdir(parents=True, exist_ok=True)
        Path(self.retrain_model_tensorboard_dir).mkdir(parents=True, exist_ok=True)
        Path(self.experiment_master_tables_dir).mkdir(parents=True, exist_ok=True)
        Path(self.no_agent_baseline_experiment_dir).mkdir(parents=True, exist_ok=True)
        
        self.train_master, self.eval_master, self.retrain_master , self.no_agent_master = self.load_master_tables()
        
    def load_master_tables(self):
        """Loads train_master, eval_master, and no_agent_master tables, if exists."""
        
        train_master_path = os.path.join(self.experiment_master_tables_dir, 'train_master.csv')
        eval_master_path = os.path.join(self.experiment_master_tables_dir, 'eval_master.csv')
        retrain_master_path = os.path.join(self.experiment_master_tables_dir, 'retrain_master.csv')
        no_agent_master_path = os.path.join(self.experiment_master_tables_dir, 'no_agent_master.csv')
        
        if os.path.exists(train_master_path):
            train_master = pd.read_csv(train_master_path)
        else:
            train_master = pd.DataFrame()            
        if os.path.exists(eval_master_path):
            eval_master = pd.read_csv(eval_master_path)
        else:
            eval_master = pd.DataFrame()     
        if os.path.exists(retrain_master_path):
            retrain_master = pd.read_csv(retrain_master_path)
        else:
            retrain_master = pd.DataFrame()       
        if os.path.exists(no_agent_master_path):
            no_agent_master = pd.read_csv(no_agent_master_path)
        else:
            no_agent_master = pd.DataFrame()           
        return train_master, eval_master, retrain_master, no_agent_master
    
    def save_master_tables(self):
        """Saves train_master, eval_master, and no_agent_master tables."""
        
        train_master_path = os.path.join(self.experiment_master_tables_dir, 'train_master.csv')
        eval_master_path = os.path.join(self.experiment_master_tables_dir, 'eval_master.csv')
        retrain_master_path = os.path.join(self.experiment_master_tables_dir, 'retrain_master.csv')
        no_agent_master_path = os.path.join(self.experiment_master_tables_dir, 'no_agent_master.csv')
        
        
        if not self.train_master.empty:
            self.train_master.to_csv(train_master_path, index=False)
        if not self.eval_master.empty:
            self.eval_master.to_csv(eval_master_path, index=False)
        if not self.no_agent_master.empty:
            self.no_agent_master.to_csv(no_agent_master_path, index=False)
        if not self.retrain_master.empty:
            self.retrain_master.to_csv(retrain_master_path, index=False)
            
    def setup_logging(self, mode, name):
        """Sets up logging to write to the specified log file."""
        if mode == 'train':
            log_file = os.path.join(self.train_logs_dir, f"{name}.log")
        elif mode == 'eval':
            log_file = os.path.join(self.eval_logs_dir, f"{name}.log")
        elif mode == 'retrain':
            log_file = os.path.join(self.retrain_model_logs_dir, f"{name}.log")
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file)
        
        
    def setup_envionment(self, **kwargs):
        """Sets up the environment for the experiment."""
        env_kwargs = {}
        env_kwargs['additional_percentage'] = kwargs['additional_percentage']
        # env_kwargs['fake_start_datetime'] = kwargs['fake_start_datetime']
        env_kwargs['rule_frequency'] = kwargs['rule_frequency']
        env_kwargs['span_size'] = kwargs['span_size']
        env_kwargs['logs_per_minute'] = kwargs['logs_per_minute']
        env_kwargs['num_of_measurements'] = kwargs['num_of_measurements']
        env_kwargs['id'] = kwargs['env_name']
        env_kwargs['search_window'] = kwargs['search_window']
        env = gym.make(**env_kwargs)
        
        reward_calculator = self.setup_reward_calc(kwargs, env)
        
        env.set_reward_calculator(reward_calculator)
        return env

    def setup_reward_calc(self, kwargs, env):
        measurment_tool = Measurement(env.splunk_tools_instance, env.num_of_searches, measure_energy=False)
        
        reward_calc_kwargs = {}
        reward_calc_kwargs['alpha'] = kwargs['alpha']
        reward_calc_kwargs['beta'] = kwargs['beta']
        reward_calc_kwargs['gamma'] = kwargs['gamma']
        reward_calc_kwargs['env_id'] = env.env_id
        reward_calc_kwargs['splunk_tools'] = env.splunk_tools_instance
        reward_calc_kwargs['dt_manager'] = env.dt_manager
        reward_calc_kwargs['rule_frequency'] = env.rule_frequency
        reward_calc_kwargs['num_of_searches'] = env.num_of_searches
        reward_calc_kwargs['measurment_tool'] = measurment_tool
        reward_calc_kwargs['top_logtypes'] = env.top_logtypes
        reward_calc_kwargs['no_agent_table_path'] = self.get_no_agent_table_path(kwargs, env)
        RewardCalc = RewardCalc_classes[kwargs['reward_calculator_version']]
        reward_calculator = RewardCalc(**reward_calc_kwargs)
        self.save_master_tables()
        return reward_calculator
    
    def get_no_agent_table_path(self, kwargs, env):
        """Returns the table name for the no agent baseline."""
        no_agent_kwargs = {}
        no_agent_kwargs['env_id'] = env.env_id
        no_agent_kwargs['search_window'] = kwargs['search_window']
        no_agent_kwargs['num_of_measurements'] = kwargs['num_of_measurements']
        filtered_df = self.no_agent_master.copy()
        for key, value in no_agent_kwargs.items():
            if key not in filtered_df.columns:
                break
            filtered_df = filtered_df[filtered_df[key] == value]
        if len(filtered_df) == 0:
            table_name = f"no_agent_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            no_agent_kwargs['table_name'] = table_name
            self.no_agent_master = pd.concat([self.no_agent_master, pd.DataFrame(no_agent_kwargs, index=[0])], ignore_index=True)
        else:
            table_name = filtered_df.iloc[0]['table_name']
        
        return os.path.join(self.no_agent_baseline_experiment_dir, f"{table_name}.csv")
    
    def get_train_experiment_name(self, kwargs):
        """Returns the name of the train experiment."""
        filtered_df = self.train_master.copy()
        kwargs.pop('num_of_episodes')
        for key, value in kwargs.items():
            if key not in filtered_df.columns:
                logger.error(f"Key: {key} not found in train_master.")
                break
            filtered_df = filtered_df[filtered_df[key].astype(str) == str(value)]
            if len(filtered_df) == 0:
                raise ValueError(f"Key: {key} with value: {value} not found in train_master.")
        name = filtered_df.iloc[0]['name']
        logger.info(f"Found train experiment with name: {name}")
        return name
    
    def setup_model(self, kwargs, env):
        model_object = model_names[kwargs['model']]
        model_kwargs = {}
        model_kwargs['env'] = env
        model_kwargs['policy'] = policy_names[kwargs['policy']]
        model_kwargs['learning_rate'] = kwargs['learning_rate']
        model_kwargs['ent_coef'] = kwargs['ent_coef']
        model_kwargs['gamma'] = kwargs['df']
        model_kwargs['n_steps'] = env.total_steps
        model_kwargs['tensorboard_log'] = self.train_tensorboard_dir
        model_kwargs['verbose'] = 1
        model_kwargs['stats_window_size'] = 5
        return model_object(**model_kwargs)
        
    def train_model(self, **kwargs):
        """Trains a model."""
        num_of_episodes, env, model, callback_list = self.prepare_experiment('train', kwargs)
        name = kwargs['name']
        model.learn(total_timesteps=env.total_steps*num_of_episodes, callback=callback_list, tb_log_name=name)
        self.post_experiment('train', env, model, kwargs)
        return model
        
    def test_model(self, **kwargs):
        """Evaluates a model."""
        num_episodes, env, model, callback_list = self.prepare_experiment('eval', kwargs)
        name = kwargs['name']
        episode_rewards = self.custom_evaluate_policy(model, env, callback_list, num_episodes=num_episodes)
        self.post_experiment('eval', env, model, kwargs)
        return episode_rewards
    
    def retrain_model(self, **kwargs):
        """Retrains a model."""
        num_of_episodes, env, model, callback_list = self.prepare_experiment('retrain', kwargs)
        name = kwargs['name']
        model.learn(total_timesteps=env.total_steps*num_of_episodes, callback=callback_list, tb_log_name=name)        
        self.retrain_master = self.update_master_tables(self.retrain_master, datetime.datetime.strptime(env.time_range[1], '%m/%d/%Y:%H:%M:%S'), kwargs)        
        self.post_experiment('retrain', env, model, kwargs)        
        return model

    def update_master_tables(self, master_table, end_time, kwargs):
        """Updates the master table with the end time and kwargs."""
        kwargs['end_time'] = end_time
        master_table = pd.concat([master_table, pd.DataFrame(kwargs, index=[0])], ignore_index=True)
        return master_table
    
    def prepare_experiment(self, mode, kwargs):
        """Prepares an experiment."""
        name = f"{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}" 
        self.setup_logging(mode, name)
        logger.info(f"Preparing experiment with env kwargs: {kwargs}")
        env = self.setup_envionment(**kwargs)
        kwargs['env_name'] = kwargs['env_name'].split('-v')[1]
        num_of_episodes = kwargs['num_of_episodes']
        model = self.get_model(kwargs, name, env, mode)
        kwargs['name'] = name
        callback_list = CallbackList([TrainTensorboardCallback(experiment_kwargs=kwargs, verbose=3), HparamsCallback(experiment_kwargs=kwargs, verbose=3)])
        return num_of_episodes, env, model, callback_list
    
    def post_experiment(self, mode, env, model, kwargs):
        """Post experiment actions."""
        if mode == 'train':
            self.train_master = self.update_master_tables(self.train_master, datetime.datetime.strptime(env.time_range[1], '%m/%d/%Y:%H:%M:%S'), kwargs)
            model.save(os.path.join(self.models_dir, kwargs['name']))            
        elif mode == 'eval':
            self.eval_master = self.update_master_tables(self.eval_master, datetime.datetime.strptime(env.time_range[1], '%m/%d/%Y:%H:%M:%S'), kwargs)
        elif mode == 'retrain':
            self.retrain_master = self.update_master_tables(self.retrain_master, datetime.datetime.strptime(env.time_range[1], '%m/%d/%Y:%H:%M:%S'), kwargs)
            model.save(os.path.join(self.models_dir, kwargs['name']))
            
        self.save_master_tables()
        env.reward_calculator.no_agent_values.to_csv(env.reward_calculator.no_agent_table_path, index=False)
    
    def get_model(self, kwargs, name, env, mode):
        if mode == 'train':
            return self.setup_model(kwargs, env)
        elif mode == 'eval' or mode == 'retrain':
            model_name = self.get_train_experiment_name(kwargs)
            return self.load_model(kwargs, name, env, model_name, mode)

    def load_model(self, kwargs, name, env, model_name, mode):
        if mode == 'eval':
            tb_logger = configure(os.path.join(self.eval_tensorboard_dir, name), ["stdout", "tensorboard"])
        elif mode == 'retrain':
            tb_logger = configure(os.path.join(self.retrain_model_tensorboard_dir, name), ["stdout", "tensorboard"])
        model = model_names[kwargs['model']].load(os.path.join(self.models_dir, model_name), env=env)
        model.set_logger(tb_logger)
        return model
    
    def custom_evaluate_policy(self, model, env, callbacks, num_episodes=100):
        model.num_timesteps = 0
        obs = env.reset()
        episode_rewards = []
        callbacks.init_callback(model)
        callbacks.on_training_start(globals(), locals())
        for episode in range(num_episodes):
            episode_reward = 0
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                # callbacks.update_child_locals(locals())
                callbacks.on_step()
                model.num_timesteps += 1
            episode_rewards.append(episode_reward)
            obs = env.reset()
            callbacks.on_rollout_end()
        callbacks.on_training_end()
        return episode_rewards
    



    
    # def evaluate_custom_policy(self, env, callbacks, model, custom_policy, num_episodes=100):
    #     episode_rewards = []
    #     model.num_timesteps = 0
    #     callbacks.init_callback(model)
    #     callbacks.on_training_start(globals(), locals())
    #     obs = env.reset()
    #     for episode in range(num_episodes):
    #         episode_reward = 0
    #         done = False
    #         while not done:
    #             action = custom_policy
    #             obs, reward, done, info = env.step(action)
    #             episode_reward += reward
    #             model.num_timesteps += 1
    #             callbacks.on_step()
    #         episode_rewards.append(episode_reward)
    #         obs = env.reset()
    #         callbacks.on_rollout_end()
    #     callbacks.on_training_end()
    #     return episode_rewards
    
    # def test_ideal_model(self, **kwargs):
    #     """Evaluates a the env by using the ideal policy. perform experiment without agent. each experiment with max step size for each log type"""
