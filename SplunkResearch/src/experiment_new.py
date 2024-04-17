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
logger = logging.getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model_names = {'a2c': A2C, 'ppo': PPO, 'dqn': DQN}









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


    
    def setup_environment(self, env_name, parameters):       
        if "total_additional_logs" in parameters:
            total_additional_logs = parameters['total_additional_logs']
        else:
            total_additional_logs = None
        # logger.info(f'current parameters:\ntime range:{time_range} \nfake_start_datetime: {fake_start_datetime}\nrule frequency: {rule_frequency}\nsearch_window:{search_window}\nrunning time: {running_time}\nnumber of searches: {num_of_searches}\nalpha {alpha}\nbeta {beta}\n gama {gamma}\nsavedsearches: {savedsearches}\nrelevantlog_types: {relevant_logtypes}')
        env = gym.make(env_name, total_additional_logs=total_additional_logs, reward_parameters=parameters['reward_parameters'], is_measure_energy=parameters['measure_energy'])
        return env
    
    def load_environment(self, env_name, modifed_parameters={}):
        parameters = self.load_parameters(f'{self.experiment_dir}/parameters_train.json') #BUG!!!!!!!!!!!!!!!!!!!!!!!!!
        parameters.update(modifed_parameters)
        env = self.setup_environment(env_name, parameters)
        return env, parameters
    
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

    def train_model(self, parameters, env_name, model, num_of_episodes):
        learning_rate = parameters['learning_rate']
        alpha, beta, gamma = parameters['reward_parameters'].values()
        prefix_path = f'{self.experiment_dir}/{model}_{alpha}_{beta}_{gamma}__{learning_rate}'
        path = f'{prefix_path}/train'
        if not os.path.exists(path):
            os.makedirs(path)
        log_file = f'{path}/log.txt'
        self.setup_logging(log_file)

        logger.info('train the model')
        parameters['measure_energy'] = False
        env = self.setup_environment(env_name, parameters)

        self.save_parameters_to_file(parameters, f'{self.experiment_dir}/parameters_train.json')
        model_object = model_names[model]
        model = model_object(MlpPolicy, env, n_steps=env.total_steps, verbose=1, stats_window_size=5, tensorboard_log=f"{path}/tensorboard/", learning_rate=learning_rate)

        model.learn(total_timesteps=num_of_episodes*env.total_steps, tb_log_name=logger.name)
        model.save(f"{prefix_path}/splunk_attack")
        self.save_assets(path, env)
        return model
    
    # def retrain_model(self, parameters):
    #     logger.info('retrain the model')
    #     parameters['measure_energy'] = False
    #     env = self.load_environment()
    #     model_object = model_names[parameters['model']]
    #     model = model_object.load(f"{self.experiment_dir}/splunk_attack")
    #     model.set_env(env)
    #     model.learn(total_timesteps=parameters['episodes']*env.total_steps)
    #     model.save(f"{self.experiment_dir}/splunk_attack")
    #     self.save_assets(path, env)
    #     return model
    
    def test_model(self, env_name, model, num_of_episodes):
        env, parameters = self.load_environment(env_name, {'measure_energy': False}) #changed to false
        learning_rate = parameters['learning_rate']
        alpha, beta, gamma = parameters['reward_parameters'].values()
        prefix_path = f'{self.experiment_dir}/{model}_{alpha}_{beta}_{gamma}__{learning_rate}'
        path = f'{prefix_path}/test'
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        log_file = f'{path}/log.txt'
        self.setup_logging(log_file)
        logger.info('test the model')
        model_object = model_names[model]
        model = model_object.load(f"{prefix_path}/splunk_attack")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_of_episodes)
        logger.info(f"mean_reward:{mean_reward}, std_reward:{std_reward}")
        self.save_assets(path, env)
        return env
    
    def test_baseline_agent(self, env_name, num_of_episodes, agent_type='random'):
        
        path = f'{self.experiment_dir}/baseline_{agent_type}'
        if not os.path.exists(path):
            os.makedirs(path)
        log_file = f'{path}/log.txt'
        self.setup_logging(log_file)
        
        
        logger.info(f'test baseline {agent_type} agent')
        env, parameters = self.load_environment(env_name,{'measure_energy': False}) #changed to false
        for i in range(num_of_episodes):
            env.reset()
            self.run_manual_episode(agent_type, env)
        self.save_assets(path, env)
        return env

            
    def test_no_agent(self, env_name, num_of_episodes):

        path = f'{self.experiment_dir}/no_agent'
        if not os.path.exists(path):
            os.makedirs(path)
        log_file = f'{path}/log.txt'
        self.setup_logging(log_file)

        
        logger.info('test no agent')
        env, parameters = self.load_environment(env_name,{'measure_energy': False}) #changed to false
        for i in range(num_of_episodes):
            env.reset()
            env.evaluate_no_agent()
        self.save_assets(path, env)
        return env
    
    def test_autopic_agent(self, env_name, num_of_episodes):
        path = f'{self.experiment_dir}/autopic_agent'
        if not os.path.exists(path):
            os.makedirs(path)
        log_file = f'{path}/log.txt'
        self.setup_logging(log_file)
        
        logger.info('test autopic agent')
        env, parameters = self.load_environment(env_name,{'measure_energy': False})
        for i in range(num_of_episodes):
            env.reset()
            env.run_manual_episode('autopic')
        self.save_assets(path, env)
        return env
    
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
