import os
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
from stable_baselines3 import A2C, PPO, DQN
urllib3.disable_warnings()

import logging
from stable_baselines3.common.callbacks import BaseCallback
logger = logging.getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model_names = {'a2c': A2C, 'ppo': PPO, 'dqn': DQN}
policy_names = {'mlp': MlpPolicy}





class TrainTensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TrainTensorboardCallback, self).__init__(verbose)


    def _on_rollout_end(self) -> None:
        logger.info("rollout end")
        env = self.training_env.envs[0]  # Accessing the actual environment inside the DummyVecEnv
        self.logger.record("train/alert_reward", env.reward_calculator.reward_dict['alerts'][-1])
        self.logger.record("train/distribution_reward", env.reward_calculator.reward_dict['distributions'][-1])
        self.logger.record("train/duration_reward", env.reward_calculator.reward_dict['duration'][-1])
        self.logger.record("train/total_reward", env.reward_calculator.reward_dict['total'][-1])
        self.logger.record("train/alert_val", env.reward_calculator.reward_values_dict['alerts'][-1])
        self.logger.record("train/distribution_val", env.reward_calculator.reward_values_dict['distributions'][-1])
        self.logger.record("train/duration_val", env.reward_calculator.reward_values_dict['duration'][-1])

        self.logger.record("train/duration_gap", env.reward_calculator.reward_values_dict['duration'][-1] - env.reward_calculator.no_agent_duration_last_val)
        self.logger.record("train/alert_gap", env.reward_calculator.reward_values_dict['alerts'][-1] - env.reward_calculator.no_agent_alert_last_val)
    
    def _on_rollout_start(self) -> None:
        env = self.training_env.envs[0]
        env.reward_calculator.get_no_agent_reward(env.time_range)   
        self.logger.record("train/no_agent_alert_val", env.reward_calculator.no_agent_alert_last_val)
        self.logger.record("train/no_agent_duration_val", env.reward_calculator.no_agent_duration_last_val)  
        
        
    
    def _on_step(self) -> bool:
        env = self.training_env.envs[0]  # Accessing the actual environment inside the DummyVecEnv
        self.logger.record("train/distribution_val", env.reward_calculator.reward_values_dict['distributions'][-1])
        self.logger.record("train/distribution_reward", env.reward_calculator.reward_dict['distributions'][-1])
    
        


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
            json.dump(env.reward_calculator.reward_dict, fp)
        with open(f'{path}/reward_values_dict.json', 'w') as fp:
            json.dump(env.reward_calculator.reward_values_dict, fp)
        with open(f'{path}/time_rules_energy.json', 'w') as fp:
            json.dump(env.reward_calculator.time_rules_energy, fp)   
        with open(f'{path}/action_dict.json', 'w') as fp:
                json.dump(np.array(env.action_per_episode).tolist(), fp)
        env.reward_calculator.no_agent_values.to_csv(f"{self.experiment_dir}/no_agent_values.csv", index=False)



    
    def setup_environment(self, env_name, **kwargs):     
        reward_parameters = {'alpha': kwargs['alpha'], 'beta': kwargs['beta'], 'gamma': kwargs['gamma']}  
        if "total_additional_logs" in kwargs:
            total_additional_logs = kwargs['total_additional_logs']
        else:
            total_additional_logs = None
        tf_log_path = f'{kwargs["prefix_path"]}/tensorboard'
        if 'fake_start_datetime' in kwargs: # datetime for test
            fake_start_datetime = kwargs['fake_start_datetime']
            env = gym.make(env_name, fake_start_datetime=fake_start_datetime, total_additional_logs=total_additional_logs, reward_parameters=reward_parameters, is_measure_energy=None, tf_log_path=tf_log_path)
            # logger.info(f'current parameters:\ntime range:{time_range} \nfake_start_datetime: {fake_start_datetime}\nrule frequency: {rule_frequency}\nsearch_window:{search_window}\nrunning time: {running_time}\nnumber of searches: {num_of_searches}\nalpha {alpha}\nbeta {beta}\n gama {gamma}\nsavedsearches: {savedsearches}\nrelevantlog_types: {relevant_logtypes}')
        else:
            env = gym.make(env_name, total_additional_logs=total_additional_logs, reward_parameters=reward_parameters, is_measure_energy=None, tf_log_path=tf_log_path)
        return env
    
    def load_environment(self, env_name, prefix_path, **kwargs):  
        fake_start_datetime = None
        with open(f'{prefix_path}/train/test_start_fake_time.txt', 'r') as fp:
            fake_start_datetime = fp.read()
        kwargs['fake_start_datetime'] = fake_start_datetime
        kwargs["prefix_path"] = prefix_path
        env = self.setup_environment(env_name, **kwargs)
        return env , fake_start_datetime
    
    def save_parameters_to_file(self, parameters, path):
        with open(f"{path}/parameters_train.json", 'w') as fp:
            json.dump(parameters, fp)
        # save reward_calculator.py to the experiment directory
        with open(f'{path}/reward_calculator.py', 'w') as fp:
            with open(r'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/reward_calculator.py', 'r') as fp2:
                fp.write(fp2.read())
    
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
        policy_name = kwargs['policy']
        model_object = model_names[model]
        policy_object =  policy_names[policy_name]
        model = model_object(policy_object, env, n_steps=env.total_steps, verbose=1, stats_window_size=5, tensorboard_log=f"{path}/tensorboard/", learning_rate=learning_rate)

        model.learn(total_timesteps=num_of_episodes*env.total_steps, tb_log_name=logger.name, callback=TrainTensorboardCallback())
        model.save(f"{prefix_path}/splunk_attack")
        self.save_test_start_fake_time(path, env)
        self.save_parameters_to_file(kwargs, path)
        self.save_assets(path, env)
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
    
    def test_model(self, env_name, model_name, num_of_episodes, **kwargs):
        alpha, beta, gamma, learning_rate, policy = kwargs['alpha'], kwargs['beta'], kwargs['gamma'], kwargs['learning_rate'], kwargs['policy']
        prefix_path = self.get_prefix_path(model_name, alpha, beta, gamma, learning_rate, policy)     
        env, fake_start_datetime = self.load_environment(env_name, prefix_path, **kwargs) #changed to false
        path = f'{prefix_path}/test_{num_of_episodes}'
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        log_file = f'{path}/log.txt'
        self.setup_logging(log_file)
        logger.info('test the model')
        policy_name = kwargs['policy']
        model_object = model_names[model_name]
        policy_object =  policy_names[policy_name]
        model = model_object.load(f"{prefix_path}/splunk_attack")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_of_episodes)
        logger.info(f"mean_reward:{mean_reward}, std_reward:{std_reward}")
        self.save_assets(path, env)
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
        alpha, beta, gamma, learning_rate, policy = kwargs['alpha'], kwargs['beta'], kwargs['gamma'], kwargs['learning_rate'], kwargs['policy']
        prefix_path = self.get_prefix_path(model_name, alpha, beta, gamma, learning_rate, policy)   
        env, fake_start_datetime = self.load_environment(env_name, prefix_path, **kwargs)
        path = f'{self.experiment_dir}/no_agent_{num_of_episodes}_{str(fake_start_datetime)}'
        if not os.path.exists(path):
            os.makedirs(path)
        log_file = f'{path}/log.txt'
        self.setup_logging(log_file)        
        logger.info('test no agent')
        for i in range(num_of_episodes):
            env.reset()
            env.evaluate_no_agent()
        self.save_assets(path, env)
        return env
            
    def retrain_model(self, env_name, model_name, num_of_episodes, **kwargs):
        alpha, beta, gamma, learning_rate, policy = kwargs['alpha'], kwargs['beta'], kwargs['gamma'], kwargs['learning_rate'], kwargs['policy']
        prefix_path = self.get_prefix_path(model_name, alpha, beta, gamma, learning_rate, policy)     
        env, fake_start_datetime = self.load_environment(env_name, prefix_path, **kwargs)
        self.load_assets(f'{prefix_path}/train', env)
        path = f'{prefix_path}/train'
        log_file = f'{path}/log.txt'
        self.setup_logging(log_file)
        logger.info('retrain the model')
        policy_name = kwargs['policy']
        model_object = model_names[model_name]
        policy_object =  policy_names[policy_name]
        model = model_object.load(f"{prefix_path}/splunk_attack")
        model.set_env(env)
        model.learn(total_timesteps=num_of_episodes*env.total_steps, tb_log_name=logger.name, callback=TrainTensorboardCallback())
        model.save(f"{prefix_path}/splunk_attack")
        self.save_test_start_fake_time(path, env)
        self.save_parameters_to_file(kwargs, path)
        self.save_assets(path, env)
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
    
    def get_prefix_path(self, model, alpha, beta, gamma, learning_rate, policy):
        # find the experiment dir with the latest serial number
        i = 1
        # find the last experiment
        prefix_path = f'{self.experiment_dir}/{policy}_{model}_{alpha}_{beta}_{gamma}__{learning_rate}/Experiment_{i}'

        while os.path.exists(prefix_path):
            prefix_path = f'{self.experiment_dir}/{policy}_{model}_{alpha}_{beta}_{gamma}__{learning_rate}/Experiment_{i+1}'
            i += 1
        return f'{self.experiment_dir}/{policy}_{model}_{alpha}_{beta}_{gamma}__{learning_rate}/Experiment_{i-1}'
        
            
    

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
