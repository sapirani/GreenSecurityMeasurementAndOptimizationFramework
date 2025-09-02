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
from unsused_scripts.measurement import Measurement
from reward_strategy import *

import logging
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import HParam
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

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class ModularTensorboardCallback(BaseCallback):
    def __init__(self, verbose=1, phase="train", experiment_kwargs=None):
        super(ModularTensorboardCallback, self).__init__(verbose)
        self.phase = phase  # This will determine whether it's "train" or "test"
        print("Experiment kwargs: ", experiment_kwargs)
        self.experiment_kwargs = experiment_kwargs
        
    def log_common_metrics(self, env):
        # Record common metrics for both training and evaluation phases
        self.logger.record(f"{self.phase}/distribution_val", env.reward_calculator.reward_values_dict['distributions'][-1])
        self.logger.record(f"{self.phase}/distribution_reward", env.reward_calculator.reward_dict['distributions'][-1])
        
        # Policy logging
        policy_dict = {}
        for i, logtype in enumerate(env.relevant_logtypes):
            for is_trigger in range(2):
                policy_dict[f'{logtype}_{is_trigger}'] = env.current_action[i*2 + is_trigger]
                if i == len(env.relevant_logtypes) - 1:
                    break
        self.logger.record(f"{self.phase}/policy", policy_dict)

    def log_detailed_metrics(self, env, no_agent_last_row):
        # Logging more detailed metrics that are common between train and evaluation
        self.logger.record(f"{self.phase}/alert_reward", env.reward_calculator.reward_dict['alerts'][-1])
        self.logger.record(f"{self.phase}/duration_reward", env.reward_calculator.reward_dict['duration'][-1])
        self.logger.record(f"{self.phase}/total_reward", env.reward_calculator.reward_dict['total'][-1])
        self.logger.record(f"{self.phase}/alert_val", env.reward_calculator.reward_values_dict['alerts'][-1])
        self.logger.record(f"{self.phase}/duration_val", env.reward_calculator.reward_values_dict['duration'][-1])
        self.logger.record(f"{self.phase}/duration_gap", env.reward_calculator.reward_values_dict['duration'][-1] - no_agent_last_row['duration_values'].values[-1])
        self.logger.record(f"{self.phase}/alert_gap", env.reward_calculator.reward_values_dict['alerts'][-1] - no_agent_last_row['alert_values'].values[-1])
        
        # Rule-based metrics
        self.logger.record(f"{self.phase}/rules_duration", {key: env.reward_calculator.time_rules_energy[-1][key] for key in env.reward_calculator.time_rules_energy[-1].keys() if key.startswith('rule_duration')})
        self.logger.record(f"{self.phase}/rules_alerts", {key: env.reward_calculator.time_rules_energy[-1][key] for key in env.reward_calculator.time_rules_energy[-1].keys() if key.startswith('rule_alert')})
        self.logger.record(f"{self.phase}/rules_durations_gap", {key: env.reward_calculator.time_rules_energy[-1][key] - no_agent_last_row[key].values[-1] for key in no_agent_last_row.columns if key.startswith('rule_duration')})
        self.logger.record(f"{self.phase}/rules_alerts_gap", {key: env.reward_calculator.time_rules_energy[-1][key] - no_agent_last_row[key].values[-1] for key in no_agent_last_row.columns if key.startswith('rule_alert')})
        policy_dict = {}
        for i, logtype in enumerate(env.relevant_logtypes):
            for is_trigger in range(2):
                policy_dict[f'{logtype}_{is_trigger}'] = env.action_per_episode[-1][i*2+is_trigger]
                if i == len(env.relevant_logtypes)-1:
                    break 
        self.logger.record("train/episodic_policy", policy_dict)

    def log_no_agent_metrics(self, env, no_agent_last_row):
        # Log metrics related to no-agent scenario
        self.logger.record(f"{self.phase}/no_agent_alert_val", no_agent_last_row['alert_values'].values[-1])
        self.logger.record(f"{self.phase}/no_agent_duration_val", no_agent_last_row['duration_values'].values[-1])
        self.logger.record(f"{self.phase}/no_agent_rules_alerts", {col: no_agent_last_row[col].values[-1] for col in no_agent_last_row.columns if col.startswith('rule_alert')})
        self.logger.record(f"{self.phase}/no_agent_rules_duration", {col: no_agent_last_row[col].values[-1] for col in no_agent_last_row.columns if col.startswith('rule_duration')})
    


class HparamsCallback(BaseCallback):
    def __init__(self, verbose=1, experiment_kwargs=None, phase="train" ):
        super(HparamsCallback, self).__init__(verbose)
        self.experiment_kwargs = experiment_kwargs
        self.phase = phase
    def _on_training_start(self) -> None:
        metric_dict = {f"{self.phase}/{tag}": 0 for tag in ["distribution_val", "distribution_reward", "alert_reward", "duration_reward", "total_reward", "alert_val", "duration_val", "duration_gap", "alert_gap", "rules_duration", "rules_alerts", "rules_durations_gap", "rules_alerts_gap", "no_agent_alert_val", "no_agent_duration_val", "no_agent_rules_alerts", "no_agent_rules_duration"]}
        hparams = HParam(self.experiment_kwargs, metric_dict)
        self.logger.record("hparams", hparams, exclude=("stdout", "log", "json", "csv"))
    
    def _on_step(self) -> bool:
        return True

class TrainTensorboardCallback(ModularTensorboardCallback):
    def __init__(self, verbose=1, experiment_kwargs=None):
        super(TrainTensorboardCallback, self).__init__(verbose, phase="train", experiment_kwargs=experiment_kwargs)

    def _on_rollout_end(self) -> None:
        env = self.training_env.envs[0]
        no_agent_last_row = env.reward_calculator.no_agent_last_row
        self.log_detailed_metrics(env, no_agent_last_row)

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        current_step = env.step_counter
        if current_step == 2:
            no_agent_last_row = env.reward_calculator.no_agent_last_row
            self.log_no_agent_metrics(env, no_agent_last_row)
        self.log_common_metrics(env)
        self.logger.dump(self.num_timesteps)
        return True
    
    # def _on_training_start(self) -> None:
    #     super()._on_training_start()
    #     self.log_hparams(self.experiment_kwargs)
    

        
def eval_tensorboard_callback(locals_dict, globals_dict):
    env = locals_dict["env"].envs[0]
    model = locals_dict["model"]
    tb_logger = model.logger
    current_step = env.step_counter
    done_episodes = len(locals_dict["episode_lengths"])
    total_steps = env.total_steps
    no_agent_last_row = env.reward_calculator.no_agent_last_row

    # Use the shared ModularTensorboardCallback for logging
    modular_callback = ModularTensorboardCallback(phase="test")
    modular_callback.logger = tb_logger
    modular_callback.log_common_metrics(env)
    if current_step == 2:
        modular_callback.log_no_agent_metrics(env, no_agent_last_row)
        tb_logger.dump(done_episodes * total_steps + current_step - 1)
    elif current_step == 1:
        modular_callback.log_detailed_metrics(env, no_agent_last_row)
        tb_logger.dump(done_episodes * total_steps + total_steps)
    else:
        tb_logger.dump(done_episodes * total_steps + current_step - 1)

    return True

        


class Experiment:
    def __init__(self, experiment_dir, model=None):
        self.experiment_dir = experiment_dir

    def setup_logging(self, log_file):
        """Sets up logging to write to the specified log file."""
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file)

    def save_assets(self, path, env):
        # print(env.reward_calculator.reward_values_dict)
        # create a directory for the assets if not exists
        with open(f'{path}/reward_dict.json', 'w') as fp:
            json.dump(env.reward_calculator.reward_dict, fp, cls=NpEncoder)
        with open(f'{path}/reward_values_dict.json', 'w') as fp:
            json.dump(env.reward_calculator.reward_values_dict, fp, cls=NpEncoder)
        with open(f'{path}/time_rules_energy.json', 'w') as fp:
            json.dump(env.reward_calculator.time_rules_energy, fp, cls=NpEncoder)
        with open(f'{path}/action_dict.json', 'w') as fp:
                json.dump(np.array(env.action_per_episode).tolist(), fp, cls=NpEncoder)
        env.reward_calculator.no_agent_values.to_csv(f"{self.experiment_dir}/no_agent_values.csv", index=False)



    
    def setup_environment(self, env_name, **kwargs):     
        alpha, beta, gamma = kwargs['alpha'], kwargs['beta'], kwargs['gamma']
        if "total_additional_logs" in kwargs:
            total_additional_logs = kwargs['total_additional_logs']
        else:
            total_additional_logs = None
        if 'fake_start_datetime' in kwargs: # datetime for test
            fake_start_datetime = kwargs['fake_start_datetime']
            env = gym.make(env_name, fake_start_datetime=fake_start_datetime, total_additional_logs=total_additional_logs)
            # logger.info(f'current parameters:\ntime range:{time_range} \nfake_start_datetime: {fake_start_datetime}\nrule frequency: {rule_frequency}\nsearch_window:{search_window}\nrunning time: {running_time}\nnumber of searches: {num_of_searches}\nalpha {alpha}\nbeta {beta}\n gama {gamma}\nsavedsearches: {savedsearches}\nrelevantlog_types: {relevant_logtypes}')
        else:
            env = gym.make(env_name, total_additional_logs=total_additional_logs)
        
        measurment_tool = Measurement(env.splunk_tools_instance, env.num_of_searches, measure_energy=False)
        RewardCalc = RewardCalc_classes[kwargs['reward_calculator_version']]
        logger.info(f'current parameters:\nalpha {alpha}\nbeta {beta}\n gama {gamma}\nreward_calculator_version: {kwargs["reward_calculator_version"]}')
        reward_calculator = RewardCalc(env.top_logtypes, env.dt_manager, env.splunk_tools_instance, env.rule_frequency, env.num_of_searches, measurment_tool, alpha, beta, gamma, env.env_id)
        env.set_reward_calculator(reward_calculator)
        return env
    
    def load_environment(self, env_name, prefix_path, **kwargs):  
        fake_start_datetime = None
        with open(f'{prefix_path}/train/test_start_fake_time.txt', 'r') as fp:
            fake_start_datetime = fp.read()
        kwargs['fake_start_datetime'] = fake_start_datetime
        kwargs["prefix_path"] = prefix_path
        parameters = self.load_parameters(f"{prefix_path}/train/parameters_train.json")
        kwargs['reward_calculator_version'] = parameters['reward_calculator_version']
        env = self.setup_environment(env_name, **kwargs)
        return env , fake_start_datetime
    
    def save_parameters_to_file(self, parameters, path):
        with open(f"{path}/parameters_train.json", 'w') as fp:
            json.dump(parameters, fp)
        # save reward_calculator.py to the experiment directory
        # with open(f'{path}/reward_calculator.py', 'w') as fp:
        #     with open(r'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/reward_calculator.py', 'r') as fp2:
        #         fp.write(fp2.read())
    
    def load_parameters(self, filename):
        with open(filename, 'r') as fp:
            parameters = json.load(fp)
        return parameters

    def train_model(self, env_name, model, num_of_episodes,**kwargs):
        alpha, beta, gamma, learning_rate, policy = kwargs['alpha'], kwargs['beta'], kwargs['gamma'], kwargs['learning_rate'], kwargs['policy']
        prefix_path = self.set_prefix_path(model, alpha, beta, gamma, learning_rate, policy)            
        path = f'{prefix_path}/train'
        if not os.path.exists(path):
            os.makedirs(path)
        log_file = f'{path}/log.txt'
        self.setup_logging(log_file)
        kwargs["prefix_path"] = prefix_path
        logger.info('train the model')
        env = self.setup_environment(env_name, **kwargs)
        fake_start_datetime = datetime.datetime.strptime(env.time_range[0], '%m/%d/%Y:%H:%M:%S')
        policy_name = kwargs['policy']
        model_object = model_names[model]
        policy_object =  policy_names[policy_name]
        ent_coef = kwargs.get('ent_coef', 0.01)
        df = kwargs.get('df', 0.99)
        model = model_object(policy_object, env, n_steps=env.total_steps, verbose=1, stats_window_size=5, tensorboard_log=f"{path}/tensorboard/", learning_rate=learning_rate, ent_coef=ent_coef, gamma=df)
        self.save_parameters_to_file(kwargs, path)
        callback_list = CallbackList([TrainTensorboardCallback(experiment_kwargs=kwargs), HparamsCallback(experiment_kwargs=kwargs)])
        model.learn(total_timesteps=num_of_episodes*env.total_steps, tb_log_name=logger.name, callback=callback_list)
        model.save(f"{prefix_path}/splunk_attack")
        self.save_test_start_fake_time(path, env)
        self.save_assets(path, env)
        end_time_datetime = datetime.datetime.strptime(env.time_range[1], '%m/%d/%Y:%H:%M:%S')
        clean_env(env.splunk_tools_instance, (fake_start_datetime.timestamp(), end_time_datetime.timestamp()))
        return model
    
    def load_assets(self, path, env):
        with open(f'{path}/reward_dict.json', 'r') as fp:
            env.reward_calculator.reward_dict = json.load(fp)
        with open(f'{path}/reward_values_dict.json', 'r') as fp:
            env.reward_calculator.reward_values_dict = json.load(fp)
        with open(f'{path}/time_rules_energy.json', 'r') as fp:
            env.reward_calculator.time_rules_energy = json.load(fp)
        with open(f'{path}/action_dict.json', 'r') as fp:
            env.action_per_episode = json.load(fp)
    
    def test_model(self, env_name, model, num_of_episodes, **kwargs):
        alpha, beta, gamma, learning_rate, policy, experiment_num = kwargs['alpha'], kwargs['beta'], kwargs['gamma'], kwargs['learning_rate'], kwargs['policy'], kwargs['experiment_num']
        prefix_path = self.get_prefix_path(model, alpha, beta, gamma, learning_rate, policy, experiment_num)     
        env, fake_start_datetime = self.load_environment(env_name, prefix_path, **kwargs) #changed to false
        path = f'{prefix_path}/test_{num_of_episodes}_{str(fake_start_datetime)}'
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        log_file = f'{path}/log.txt'
        self.setup_logging(log_file)
        logger.info('test the model')
        policy_name = kwargs['policy']
        model_object = model_names[model]
        policy_object =  policy_names[policy_name]
        tb_logger = configure(f"{path}/tensorboard/", ['tensorboard']) 
        model = model_object.load(f"{prefix_path}/splunk_attack")
        model.set_logger(tb_logger)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_of_episodes, render=False, callback=eval_tensorboard_callback)
        logger.info(f"mean_reward:{mean_reward}, std_reward:{std_reward}")
        self.save_assets(path, env)
        fake_start_datetime = datetime.datetime.strptime(env.time_range[0], '%m/%d/%Y:%H:%M:%S')
        end_time_datetime = datetime.datetime.strptime(env.time_range[1], '%m/%d/%Y:%H:%M:%S')
        clean_env(env.splunk_tools_instance, (fake_start_datetime.timestamp(), end_time_datetime.timestamp()))
        return env
    
    def test_baseline_agent(self, env_name, model_name, num_of_episodes, **kwargs):
        alpha, beta, gamma, learning_rate, agent_type, policy = kwargs['alpha'], kwargs['beta'], kwargs['gamma'], kwargs['learning_rate'], kwargs['agent_type'], kwargs['policy']
        prefix_path = self.get_prefix_path(model_name, alpha, beta, gamma, learning_rate, policy)   
        env, fake_start_datetime = self.load_environment(env_name, prefix_path, **kwargs)
        path = f'{self.experiment_dir}/baseline_{agent_type}_{num_of_episodes}_{str(fake_start_datetime)}'
        if not os.path.exists(path):
            os.makedirs(path)
        log_file = f'{path}/log.txt'
        self.setup_logging(log_file)
        logger.info(f'test baseline {agent_type} agent')
        for i in range(num_of_episodes):
            env.reset()
            self.run_manual_episode(agent_type, env)
        self.save_assets(path, env)
        return env
            
    def test_no_agent(self, env_name, model_name, num_of_episodes, **kwargs):
        alpha, beta, gamma, learning_rate, policy, experiment_num = kwargs['alpha'], kwargs['beta'], kwargs['gamma'], kwargs['learning_rate'], kwargs['policy'], kwargs['experiment_num']
        prefix_path = self.get_prefix_path(model_name, alpha, beta, gamma, learning_rate, policy, experiment_num)   
        env, fake_start_datetime = self.load_environment(env_name, prefix_path, **kwargs)
        path = f'{self.experiment_dir}/no_agent_{num_of_episodes}_{str(fake_start_datetime)}'
        if not os.path.exists(path):
            os.makedirs(path)
        log_file = f'{path}/log.txt'
        self.setup_logging(log_file)        
        logger.info('test no agent')
        tb_logger = configure(f"{path}/tensorboard/", ['tensorboard']) 
        for i in range(num_of_episodes):
            env.reset()
            env.reward_calculator.get_no_agent_reward(env.time_range)
            env.evaluate_no_agent()
            tb_logger.record("test_no_agent/alert_reward", env.reward_calculator.reward_dict['alerts'][-1])
            tb_logger.record("test_no_agent/duration_reward", env.reward_calculator.reward_dict['duration'][-1])
            tb_logger.record("test_no_agent/total_reward", env.reward_calculator.reward_dict['total'][-1])
            tb_logger.record("test_no_agent/alert_val", env.reward_calculator.reward_values_dict['alerts'][-1])
            tb_logger.record("test_no_agent/duration_val", env.reward_calculator.reward_values_dict['duration'][-1])
            tb_logger.record("test_no_agent/duration_gap", env.reward_calculator.reward_values_dict['duration'][-1] - env.reward_calculator.no_agent_duration_last_val)
            tb_logger.record("test_no_agent/alert_gap", env.reward_calculator.reward_values_dict['alerts'][-1] - env.reward_calculator.no_agent_alert_last_val)
            tb_logger.dump(i)
        self.save_assets(path, env)
        fake_start_datetime = datetime.datetime.strptime(env.time_range[0], '%m/%d/%Y:%H:%M:%S')
        end_time_datetime = datetime.datetime.strptime(env.time_range[1], '%m/%d/%Y:%H:%M:%S')
        clean_env(env.splunk_tools_instance, (fake_start_datetime.timestamp(), end_time_datetime.timestamp()))
        return env
            
    def retrain_model(self, env_name, model, num_of_episodes, **kwargs):
        alpha, beta, gamma, learning_rate, policy, experiment_num = kwargs['alpha'], kwargs['beta'], kwargs['gamma'], kwargs['learning_rate'], kwargs['policy'], kwargs['experiment_num']
        prefix_path = self.get_prefix_path(model, alpha, beta, gamma, learning_rate, policy, experiment_num)    
        path = f'{prefix_path}/train'
        log_file = f'{path}/log.txt'
        self.setup_logging(log_file)
        env, fake_start_datetime = self.load_environment(env_name, prefix_path, **kwargs)
        self.load_assets(f'{prefix_path}/train', env)

        logger.info('retrain the model')
        policy_name = kwargs['policy']
        model_object = model_names[model]
        policy_object =  policy_names[policy_name]
        model = model_object.load(f"{prefix_path}/splunk_attack")
        model.set_env(env)
        # model.set_logger(configure(f"{path}/tensorboard/", ['tensorboard']))
        model.learn(total_timesteps=num_of_episodes*env.total_steps, callback=TrainTensorboardCallback(), tb_log_name=logger.name, reset_num_timesteps=False)
        model.save(f"{prefix_path}/splunk_attack")
        self.save_test_start_fake_time(path, env)
        # self.save_parameters_to_file(kwargs, path)
        self.save_assets(path, env)
        fake_start_datetime = datetime.datetime.strptime(env.time_range[0], '%m/%d/%Y:%H:%M:%S')
        end_time_datetime = datetime.datetime.strptime(env.time_range[1], '%m/%d/%Y:%H:%M:%S')
        clean_env(env.splunk_tools_instance, (fake_start_datetime.timestamp(), end_time_datetime.timestamp()))
        return model
    
    def save_test_start_fake_time(self, path, env):
        with open(f'{path}/test_start_fake_time.txt', 'w') as fp:
            fp.write(str(env.dt_manager.get_fake_current_datetime()))
            
    def set_prefix_path(self, model, alpha, beta, gamma, learning_rate, policy):
        prefix_path = f'{self.experiment_dir}/{policy}_{model}_{alpha}_{beta}_{gamma}__{learning_rate}'
        # check serial number of experiment
        i = 1
        while os.path.exists(prefix_path):
            prefix_path = f'{self.experiment_dir}/{policy}_{model}_{alpha}_{beta}_{gamma}__{learning_rate}/Experiment_{i}'
            i += 1
        return prefix_path
    
    def get_prefix_path(self, model, alpha, beta, gamma, learning_rate, policy, experiment_num):
        if experiment_num == 'last':
            # find the experiment dir with the latest serial number
            i = 1
            # find the last experiment
            prefix_path = f'{self.experiment_dir}/{policy}_{model}_{alpha}_{beta}_{gamma}__{learning_rate}/Experiment_{i}'

            while os.path.exists(prefix_path):
                prefix_path = f'{self.experiment_dir}/{policy}_{model}_{alpha}_{beta}_{gamma}__{learning_rate}/Experiment_{i+1}'
                i += 1
            return f'{self.experiment_dir}/{policy}_{model}_{alpha}_{beta}_{gamma}__{learning_rate}/Experiment_{i-1}'
        else:
            return f'{self.experiment_dir}/{policy}_{model}_{alpha}_{beta}_{gamma}__{learning_rate}/Experiment_{experiment_num}'
        
            
    

    def run_manual_episode(self, agent_type, env):
        done = False
        while not done:
            if agent_type == 'random':
                # action = np.array([random.uniform(0, 1) for i in range((len(env.relevant_logtypes)-1)*2+2)])
                # action = np.random.dirichlet(np.ones(env.action_space.shape))
                action = env.action_space.sample()
            elif agent_type == 'uniform':
                action = 100/len(env.relevant_logtypes)  
            elif agent_type == 'autopic':
                action = np.ones(env.action_space.shape)
                for i in range(len(action)):
                    if i%2 != 0:
                        action[i] = 0
                action[-2] = 0
            obs, reward, done, info = env.step(action)
            env.render()
