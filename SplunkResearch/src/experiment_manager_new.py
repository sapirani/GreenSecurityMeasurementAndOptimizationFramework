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

from strategies.state_strategy import *
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
from stable_baselines3.ddpg.policies import MlpPolicy as DDPGMlpPolicy
from stable_baselines3.td3.policies import MlpPolicy as TD3Policy   
from stable_baselines3.sac.policies import MlpPolicy as SACPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import custom_splunk #dont remove!!!
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C, PPO, DQN, DDPG, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

urllib3.disable_warnings()
from stable_baselines3.common.logger import configure
from env_utils import *
from measurement import Measurement
from strategies.reward_strategy import *
from strategies.action_strategy import *
from pathlib import Path
import logging
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import HParam
from callbacks import *
from datetime_manager import MockedDatetimeManager
from splunk_tools import SplunkTools
from stable_baselines3.common.distributions  import DirichletDistribution
from policy import *
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# disable sb3 warning
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model_names = {'a2c': A2C, 'ppo': PPO, 'dqn': DQN, 'recurrentppo': RecurrentPPO, 'ddpg': DDPG, 'td3': TD3, 'sac': SAC}
policy_names = {'mlp': MlpPolicy, 'lstm': MlpLstmPolicy, "custommlp": CustomActor, "ddpgmlp": DDPGMlpPolicy, "td3mlp": TD3Policy, "sacmlp": SACPolicy}

# Dynamically find all reward calculator classes
RewardCalc_classes = {}
for name, obj in inspect.getmembers(sys.modules['strategies.reward_strategy'], inspect.isclass):
    if issubclass(obj, RewardStrategy) and obj is not RewardStrategy:
        RewardCalc_classes[name.split("RewardStrategy")[1]] = obj
logger.info(f"Loaded RewardCalc_classes: {RewardCalc_classes}")
StateStrategy_classes = {}
for name, obj in inspect.getmembers(sys.modules['strategies.state_strategy'], inspect.isclass):
    if issubclass(obj, StateStrategy) and obj is not StateStrategy:
        StateStrategy_classes[name.split("StateStrategy")[1]] = obj
logger.info(f"Loaded StateStrategy_classes: {StateStrategy_classes}")
ActionStrategy_classes = {}
for name, obj in inspect.getmembers(sys.modules['strategies.action_strategy'], inspect.isclass):
    if issubclass(obj, ActionStrategy) and obj is not ActionStrategy:
        ActionStrategy_classes[name.split("ActionStrategy")[1]] = obj

manual_policy_dict = {
    '4663_0': {0:1},
    '4663_1': {1:1},
    '4732_0': {2:1},
    '4732_1': {3:1},
    '4769_0': {4:1},
    '4769_1': {5:1},
    '5140_0': {6:1},
    '5140_1': {7:1},
    '7036_0': {8:1},
    '7036_1': {9:1},
    '7040_0': {10:1},
    '7040_1': {11:1},
    '7045_0': {12:1},
    '7045_1': {13:1},
    '4624_0': {14:1},
    'equal_1': {1:1/7, 3:1/7, 5:1/7, 7:1/7, 9:1/7, 11:1/7, 13:1/7},
    'equal_0': {0:1/7, 2:1/7, 4:1/7, 6:1/7, 8:1/7, 10:1/7, 12:1/7, 14:1/7},
    'no_agent': {},
}
manual_policy_dict_1 = {
    '4663_0': {0:100, 1:0, 2:0, 3:0},
    '4663_1': {0:100, 1:0, 2:1, 3:5},
    '4732_0': {0:100, 1:1, 2:0, 3:0},
    '4732_1': {0:100, 1:1, 2:1, 3:5},
    '4769_0': {0:100, 1:2, 2:0, 3:0},
    '4769_1': {0:100, 1:2, 2:1, 3:5},
    '5140_0': {0:100, 1:3, 2:0, 3:0},
    '5140_1': {0:100, 1:3, 2:1, 3:5},
    '7036_0': {0:100, 1:4, 2:0, 3:0},
    '7036_1': {0:100, 1:4, 2:1, 3:5},
    '7040_0': {0:100, 1:5, 2:0, 3:0},
    '7040_1': {0:100, 1:5, 2:1, 3:5},
    '7045_0': {0:100, 1:6, 2:0, 3:0},
    '7045_1': {0:100, 1:6, 2:1, 3:5},
    '4624_0': {0:100, 1:7, 2:0, 3:0},

    'no_agent': {0:0, 1:0, 2:0, 3:0}
}
        
class ExperimentManager:
    
    def __init__(self, base_dir="experiments____", log_level=logging.INFO):
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
        self.mode = None
        self.current_kwargs = None
        self.train_master, self.eval_master, self.retrain_master , self.no_agent_master = self.load_master_tables()
        self.total_steps = 0
        
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
        elif 'eval' in mode:
            log_file = os.path.join(self.eval_logs_dir, f"{name}.log")
        elif mode == 'retrain':
            log_file = os.path.join(self.retrain_model_logs_dir, f"{name}.log")
        logging.basicConfig(level=self.log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=log_file)
        
        
    def setup_envionment(self, **kwargs):
        """Sets up the environment for the experiment."""
        env_kwargs = {}
        env_kwargs['additional_percentage'] = kwargs['additional_percentage']
        if fake_start_datetime := kwargs.get('fake_start_datetime'):
            # convert fake_start_datetime format
            kwargs['fake_start_datetime'] = datetime.datetime.strptime(fake_start_datetime, '%Y-%m-%d %H:%M:%S')
            kwargs['fake_start_datetime'] = kwargs['fake_start_datetime'].strftime('%m/%d/%Y:%H:%M:%S')
            env_kwargs['fake_start_datetime'] = kwargs['fake_start_datetime']
        env_kwargs['rule_frequency'] = kwargs['rule_frequency']
        env_kwargs['span_size'] = kwargs['span_size']
        env_kwargs['logs_per_minute'] = kwargs['logs_per_minute']
        env_kwargs['num_of_measurements'] = kwargs['num_of_measurements']
        env_kwargs['id'] = kwargs['env_name']
        env_kwargs['search_window'] = kwargs['search_window']
        env_kwargs['state_strategy'] = StateStrategy_classes[kwargs['state_strategy_version']]
        env_kwargs['action_strategy'] = ActionStrategy_classes[kwargs['action_strategy_version']]
        env = gym.make(**env_kwargs)
        self.total_steps = env.total_steps
        # env = DummyVecEnv([lambda: env])
        env = make_vec_env(lambda: env, n_envs=1)
        env = VecNormalize(
            env,
            norm_obs=True,  # normalize observations
            norm_reward=True,  # normalize rewards
            clip_obs=10.,
            clip_reward=10.,
            gamma=kwargs['df'],
            epsilon=1e-8
        )
        reward_calculator = self.setup_reward_calc(kwargs)
        # env.set_reward_calculator(reward_calculator)
        env.env_method('set_reward_calculator', reward_calculator)
        
        return env

    def setup_reward_calc(self, kwargs):
        splunk_tools_instance = SplunkTools()
        num_of_searches = splunk_tools_instance.get_num_of_searches()
        measurment_tool = Measurement(splunk_tools_instance, num_of_searches, measure_energy=False)
        
        reward_calc_kwargs = {}
        reward_calc_kwargs['alpha'] = kwargs['alpha']
        reward_calc_kwargs['beta'] = kwargs['beta']
        reward_calc_kwargs['gamma'] = kwargs['gamma']
        reward_calc_kwargs['splunk_tools'] = splunk_tools_instance
        reward_calc_kwargs['dt_manager'] = MockedDatetimeManager()
        reward_calc_kwargs['num_of_searches'] = num_of_searches
        reward_calc_kwargs['measurment_tool'] = measurment_tool
        reward_calc_kwargs['no_agent_table_path'] = self.get_no_agent_table_path(kwargs)
        RewardCalc = RewardCalc_classes[kwargs['reward_calculator_version']]
        reward_calculator = RewardCalc(**reward_calc_kwargs)
        self.save_master_tables()
        return reward_calculator
    
    def get_no_agent_table_path(self, kwargs):
        """Returns the table name for the no agent baseline."""
        no_agent_kwargs = {}
        no_agent_kwargs['env_id'] = kwargs['env_name']
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
        if experiment_name := kwargs.get('experiment_name'):
            return experiment_name
        """Returns the name of the train experiment."""
        filtered_df = self.train_master.copy()
        for key, value in kwargs.items():
            if key not in filtered_df.columns:
                logger.error(f"Key: {key} not found in train_master.")
                break
            if key == 'env_name':
                value = value.split('-v')[1]
            if key == 'num_of_episodes':
                continue                
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
        
        if kwargs['model'] != 'ddpg' and kwargs['model'] != 'td3' and kwargs['model'] != 'sac' and kwargs['model'] != 'dqn':
            model_kwargs['ent_coef'] = kwargs['ent_coef']
            model_kwargs['n_steps'] = kwargs['n_steps']
            model_kwargs['stats_window_size'] = 5
            model_kwargs['policy_kwargs'] = dict(
                                net_arch=dict(
                                    pi=[128, 128, 64],    # 3 layers
                                    vf=[128, 128, 64]
                                )
                            )
            # model_kwargs['use_sde'] = True
            # model_kwargs['sde_sample_freq'] = 9
            
        elif kwargs['model'] == 'td3' or kwargs['model'] == 'sac' or kwargs['model'] == 'ddpg':
            # model_kwargs['policy_kwargs'] = dict(
            # net_arch=[128, 128, 64]
            # )
            # model_kwargs['action_noise'] = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=0.1 * np.ones(env.action_space.shape))
            model_kwargs['train_freq'] = kwargs['n_steps']
            model_kwargs['use_sde'] = True
            model_kwargs['sde_sample_freq'] = 10
        else:
            model_kwargs['policy_kwargs'] = dict(
            net_arch=dict(
                pi=[128, 128, 64],
                qf=[128, 128, 64]  # For DDPG, use 'qf' instead of 'vf'
                )
            )
            
        model_kwargs['gamma'] = kwargs['df']
        model_kwargs['tensorboard_log'] = self.train_tensorboard_dir
        model_kwargs['verbose'] = 1

        return model_object(**model_kwargs)
        
    def train_model(self, **kwargs):
        """Trains a model."""
        self.mode = 'train'
        env, model, callback_list = self.prepare_experiment('train', kwargs)
        name = kwargs['name']
        kwargs['env_name'] = kwargs['env_name'].split('-v')[1]        
        self.current_kwargs = kwargs
        total_steps = env.get_attr('total_steps')[0]
        print(total_steps)

        model.learn(total_timesteps=self.total_steps*kwargs['num_of_episodes'], callback=callback_list, tb_log_name=name)
        
        self.post_experiment(env, model)
        return model
        
    def test_model(self, **kwargs):
        """Evaluates a model."""
        self.mode = 'eval'
        env, model, callback_list = self.prepare_experiment('eval', kwargs)
        self.current_kwargs = kwargs
        episode_rewards = self.custom_evaluate_policy(model, env, callback_list, num_episodes=kwargs['num_of_episodes'])
        self.post_experiment(env, model)
        return episode_rewards
    
    def manual_policy_eval(self, **kwargs):
        """Evaluates a model."""
        self.mode =  'manual_policy_eval'
        env, model, callback_list = self.prepare_experiment('manual_policy_eval', kwargs)
        self.current_kwargs = kwargs
        episode_rewards = self.custom_evaluate_policy(model, env, callback_list, num_episodes=kwargs['num_of_episodes'])
        self.post_experiment(env, model)
        return episode_rewards
    
    def random_policy_eval(self, **kwargs):
        """Evaluates a model."""
        self.mode =  'random_policy_eval'
        env, model, callback_list = self.prepare_experiment('random_policy_eval', kwargs)
        self.current_kwargs = kwargs
        episode_rewards = self.custom_evaluate_policy(model, env, callback_list, num_episodes=kwargs['num_of_episodes'])
        self.post_experiment(env, model)
        return episode_rewards
    
    def get_fake_start_datetime(self, kwargs):
        """Returns the fake start datetime."""
        try:
            train_experiment_name = self.get_train_experiment_name(kwargs)
            return self.train_master[self.train_master['name'] == train_experiment_name]['end_time'].values[0]
        except Exception as e:
            logger.warning(f"Error getting fake_start_datetime: {e}")
            return None
            
    def retrain_model(self, **kwargs):
        """Retrains a model."""
        self.mode = 'retrain'
        if kwargs['fake_start_datetime'] is None:
            kwargs['fake_start_datetime'] = self.get_fake_start_datetime(kwargs)
        env, model, callback_list = self.prepare_experiment('retrain', kwargs)
        logger.info(f"Retraining model from datetime: {kwargs['fake_start_datetime']}")
        name = kwargs['name']
        self.current_kwargs = kwargs
        total_steps = env.get_attr('total_steps')
        model.learn(total_timesteps=total_steps*kwargs['num_of_episodes'], callback=callback_list, tb_log_name=name)        
        self.retrain_master = self.update_master_tables(self.retrain_master, datetime.datetime.strptime(env.time_range[1], '%m/%d/%Y:%H:%M:%S'), kwargs)        
        self.post_experiment(env, model)        
        return model

    def update_master_tables(self, master_table, end_time, kwargs):
        """Updates the master table with the end time and kwargs."""
        kwargs['end_time'] = end_time
        new_row = pd.DataFrame(kwargs, index=[0])
        if new_row['name'].values[0] in master_table['name'].values:
            master_table[master_table['name'] == new_row['name'].values[0]] = new_row
        else:
            master_table = pd.concat([master_table, new_row], ignore_index=True)
        return master_table
    
    def prepare_experiment(self, mode, kwargs):
        """Prepares an experiment."""
        name = f"{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}" 
        self.setup_logging(mode, name)
        logger.info(f"Preparing experiment with env kwargs: {kwargs}")
        env = self.setup_envionment(**kwargs)
        model = self.get_model(kwargs, name, env, mode)
        kwargs['name'] = name
        callback_list = CallbackList([TrainTensorboardCallback(experiment_kwargs=kwargs, verbose=3), HparamsCallback(experiment_kwargs=kwargs, verbose=3), SaveArtifacts(experiment_menager=self)])#SaveModelCallback(save_path=os.path.join(self.models_dir, kwargs['name']))])
        return env, model, callback_list
    
    def post_experiment(self, env, model):
        """Post experiment actions."""
        try:
            time_range = env.get_attr('time_range')
        except:
            time_range = env.time_range
        
        if self.mode == 'train':
            self.train_master = self.update_master_tables(self.train_master, datetime.datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S'), self.current_kwargs)
            model.save(os.path.join(self.models_dir, self.current_kwargs['name']))            
        elif self.mode == 'eval' or self.mode == 'manual_policy_eval':
            self.eval_master = self.update_master_tables(self.eval_master, datetime.datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S'), self.current_kwargs)
        elif self.mode == 'retrain':
            self.retrain_master = self.update_master_tables(self.retrain_master, datetime.datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S'), self.current_kwargs)
            model.save(os.path.join(self.models_dir, self.current_kwargs['name']))
            
        self.save_master_tables()
        # env.reward_calculator.no_agent_values.to_csv(env.reward_calculator.no_agent_table_path, index=False)
    
    def get_model(self, kwargs, name, env, mode):
        if mode == 'train':
            model = self.setup_model(kwargs, env)
        elif mode == 'eval' or mode == 'retrain':
            model_name = self.get_train_experiment_name(kwargs)
            model = self.load_model(kwargs, name, env, model_name, mode)
        elif mode == "manual_policy_eval":
            model = ManualPolicyModel(ManualPolicy(manual_policy_dict_1[kwargs['policy']],env.observation_space, env.action_space), [env])
            tb_logger = configure(os.path.join(self.eval_tensorboard_dir, name), ["stdout", "tensorboard"])
            model.logger = tb_logger
        elif mode == "random_policy_eval":
            model = ManualPolicyModel(RandomPolicy(env.observation_space, env.action_space), [env])
            tb_logger = configure(os.path.join(self.eval_tensorboard_dir, name), ["stdout", "tensorboard"])
            model.logger = tb_logger
        # model.policy.action_dist = NormalizedDiagGaussianDistribution(int(np.prod(env.action_space.shape))) # DirichletDistribution(env.action_space.shape[0])
        return model

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
        dones = [False]
        for episode in range(num_episodes):
            episode_reward = 0
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic=False)
                obs, reward, done, info = env.step(action)
                dones[0] = done
                callbacks.update_locals(locals())
                episode_reward += reward
                # callbacks.update_child_locals(locals())
                callbacks.on_step()
                model.num_timesteps += 1
            episode_rewards.append(episode_reward)
            obs = env.reset()
            callbacks.on_rollout_end()
        callbacks.on_training_end()
        return episode_rewards
    


