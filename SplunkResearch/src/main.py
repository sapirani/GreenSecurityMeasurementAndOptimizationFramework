import os
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



def update_running_time(running_time, env_file_path):
    command = f'sed -i "s/RUNNING_TIME=.*/RUNNING_TIME={running_time}/" "{env_file_path}"'
    res = subprocess.run(command, shell=True, capture_output=True, text=True)
    self.logger.info(res.stdout)
    self.logger.info(res.stderr)

def choose_random_rules(splunk_tools_instance, num_of_searches, get_only_enabled=True):
    self.logger.info('enable random rules')
    savedsearches = splunk_tools_instance.get_saved_search_names(get_only_enabled=get_only_enabled)
    random_savedsearch = random.sample(savedsearches, num_of_searches)
    for savedsearch in savedsearches:
        if savedsearch not in random_savedsearch:
            splunk_tools_instance.disable_search(savedsearch)
        else:
            splunk_tools_instance.enable_search(savedsearch)
    return random_savedsearch

def update_rules_frequency_and_time_range(splunk_tools_instance, rule_frequency, time_range):
    self.logger.info('update rules frequency')
    splunk_tools_instance.update_all_searches(splunk_tools_instance.update_search_cron_expression, f'*/{rule_frequency} * * * *')
    self.logger.info('update time range of rules')
    splunk_tools_instance.update_all_searches(splunk_tools_instance.update_search_time_range, time_range)   


def run_baseline_method(method, fake_start_datetime, dt_manager, splunk_tools_instance, time_range, log_generator_instance, rule_frequency, search_window, reward_parameter_dict, relevant_logtypes, max_actions_value, num_of_episodes, current_dir):
    dt_manager.set_fake_current_datetime(fake_start_datetime.strftime("%m/%d/%Y:%H:%M:%S"))
    # clean_env(splunk_tools_instance, time_range)
    for j in range(3):
        env = gym.make('splunk_attack-v0', log_generator_instance=log_generator_instance, splunk_tools_instance=splunk_tools_instance, dt_manager=dt_manager, time_range=time_range, rule_frequency=rule_frequency, search_window=search_window, reward_parameter_dict=reward_parameter_dict, relevant_logtypes=relevant_logtypes, max_actions_value=max_actions_value)
        for i in range(num_of_episodes):
            env.reset()
            # done = False
            # while not done:
            #     action = 0 if method == "0" else random.uniform(1, (100-env.sum_of_fractions))
            #     obs, reward, done, info = env.step(action)
            #     env.render()
            env.done = True
            env.update_state()
            env.get_reward()
        save_assets(current_dir, env, mode=f'test_baseline{method}{j}')
        env.update_timerange()#remove
        time_range = env.time_range#remove

def save_assets(current_dir, env, mode='train'):
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

def evaluate_model(rule_frequency, search_window, max_actions_value, num_of_episodes, fake_start_datetime, reward_parameter_dict, current_dir, dt_manager, splunk_tools_instance, relevant_logtypes, time_range, log_generator_instance):
    clean_env(splunk_tools_instance, time_range) 
    dt_manager.set_fake_current_datetime(fake_start_datetime.strftime("%m/%d/%Y:%H:%M:%S"))
    self.logger.info('Testing the model')
    env = gym.make('splunk_attack-v0', log_generator_instance = log_generator_instance, splunk_tools_instance = splunk_tools_instance, dt_manager=dt_manager, time_range=time_range, rule_frequency=rule_frequency, search_window=search_window, reward_parameter_dict=reward_parameter_dict, relevant_logtypes=relevant_logtypes, max_actions_value=max_actions_value)
    self.logger.info('load and run the DRL method')
    model = A2C.load(f"{current_dir}/A2C_splunk_attack")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_of_episodes)
    self.logger.info(f"mean_reward:{mean_reward}, std_reward:{std_reward}")
    save_assets(current_dir, env, mode='test')
 
def clean_env(splunk_tools_instance, time_range):
    date = time_range[1].split(':')[0]
    time_range = (f'{date}:00:00:00', f'{date}:23:59:59')
    splunk_tools_instance.delete_fake_logs(time_range)
    return time_range

if __name__ == "__main__":
    print('##########################################################################start##########################################################################')
    print('##########################################################################\n##########################################################################')
    mode = sys.argv[1]
    rule_frequency = 2
    search_window = 5
    max_actions_value = 30000
    num_of_searches = 28
    running_time="1.5" #in minutes
    num_of_episodes = 5
    test_num_of_episodes = 50
    env_file_path = "/home/shouei/GreenSecurity-FirstExperiment/Scanner/.env"
    fake_start_datetime = datetime.datetime(2023,6,22, 8, 30, 0)
    reward_parameter_dict ={'alpha': 0.6, 'beta': 0.05, 'gamma': 0.15, 'delta': 0.20}

    
    manager = ExperimentManager() 
    global dt_manager
        
    if mode == 'train' or sys.argv[2] == 'baselines':
        # Create a new experiment directory
        current_dir = manager.create_experiment_dir()
        log_file_name = "log_train"
        dt_manager = MockedDatetimeManager(fake_start_datetime=fake_start_datetime, log_file_path=f"{current_dir}/{log_file_name}.txt")
        splunk_tools_instance = SplunkTools(dt_manager=dt_manager)
        print(f"New experiment directory: {current_dir}")
        savedsearches = sorted(choose_random_rules(splunk_tools_instance, num_of_searches, get_only_enabled=True))
        relevant_logtypes =  list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]})
        
    else: 
        if sys.argv[2] == 'last':
            # Get the last experiment directory
            current_dir = manager.get_last_experiment_dir()
        else:
            current_dir = f'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/{sys.argv[2]}'
        print(f"Current directory: {current_dir}")
        with open(f'{current_dir}/parameters_train.json') as f:
            parameters = json.load(f)
        savedsearches = parameters['savedsearches']
        relevant_logtypes = parameters['relevantlog_types']
        if mode == 'test':
            log_file_name = "log_test"
        else:
            log_file_name = "log_retrain"
        dt_manager = MockedDatetimeManager(fake_start_datetime=fake_start_datetime, log_file_path=f"{current_dir}/{log_file_name}.txt")
        splunk_tools_instance = SplunkTools(dt_manager=dt_manager)
            
    
    end_time = dt_manager.get_fake_current_datetime()
    start_time = dt_manager.subtract_time(end_time, minutes=search_window)
    time_range = (start_time, end_time)
    log_generator_instance = LogGenerator(relevant_logtypes, big_replacement_dicts, splunk_tools_instance)
    
    # print all the rules that are running and the current parameters
    self.logger.info(f'current parameters:\nfake_start_datetime: {fake_start_datetime}\nrule frequency: {rule_frequency}\nsearch_window:{search_window}\nrunning time: {running_time}\nnumber of searches: {num_of_searches}\nnumber of episodes: {num_of_episodes}\nmax action value: {max_actions_value}\nreward parameter dict: {reward_parameter_dict}\nsavedsearches: {savedsearches}\nrelevantlog_types: {relevant_logtypes}')
    # save parameters to file
    with open(f'{current_dir}/parameters_{mode}.json', 'w') as fp:
        json.dump({'fake_start_datetime': str(fake_start_datetime), 'rule_frequency': rule_frequency, 'search_window': search_window, 'running_time': running_time, 'num_of_searches': num_of_searches, 'num_of_episodes': num_of_episodes, 'max_actions_value': max_actions_value, 'reward_parameter_dict': reward_parameter_dict, 'savedsearches': savedsearches, 'relevantlog_types': relevant_logtypes}, fp)
    
    update_running_time(running_time, env_file_path)
    update_rules_frequency_and_time_range(splunk_tools_instance, rule_frequency, time_range)
    gym.register(
            id='splunk_attack-v0',
            entry_point='framework:Framework',  # Replace with the appropriate path
        )
    
    if mode == 'test':
        if sys.argv[2] == 'baselines':
            action_values = [0]
            # action_values = [max_actions_value, max_actions_value//2, max_actions_value//4,0]
            for action_value in action_values:
                    run_baseline_method(str(action_value), fake_start_datetime, dt_manager, splunk_tools_instance, time_range, log_generator_instance, rule_frequency, search_window, reward_parameter_dict, relevant_logtypes, action_value, test_num_of_episodes, current_dir)
        else:    
            evaluate_model(rule_frequency, search_window, max_actions_value, test_num_of_episodes, fake_start_datetime, reward_parameter_dict, current_dir, dt_manager, splunk_tools_instance, relevant_logtypes, time_range, log_generator_instance)   
            run_baseline_method("0", fake_start_datetime, dt_manager, splunk_tools_instance, time_range, log_generator_instance, rule_frequency, search_window, reward_parameter_dict, relevant_logtypes, 0, test_num_of_episodes, current_dir) 
            run_baseline_method("1", fake_start_datetime, dt_manager, splunk_tools_instance, time_range, log_generator_instance, rule_frequency, search_window, reward_parameter_dict, relevant_logtypes, max_actions_value, test_num_of_episodes, current_dir)
    else:
        clean_env(splunk_tools_instance, time_range) 
        env = gym.make('splunk_attack-v0', log_generator_instance = log_generator_instance, splunk_tools_instance = splunk_tools_instance, dt_manager=dt_manager, time_range=time_range, rule_frequency=rule_frequency, search_window=search_window, reward_parameter_dict=reward_parameter_dict, relevant_logtypes=relevant_logtypes, max_actions_value=max_actions_value)
        if mode == 'train':
            model = A2C(MlpPolicy, env, verbose=3)
        elif mode == 'retrain':
            model = A2C.load(f"{current_dir}/A2C_splunk_attack", env=env, verbose=3)    
        self.logger.info('start learning')
        model.learn(total_timesteps=len(relevant_logtypes)*num_of_episodes)
        self.logger.info('finish learning, saving model')
        model.save(f"{current_dir}/A2C_splunk_attack")
        save_assets(current_dir, env)

 
    # clean_env(splunk_tools_instance, time_range) 
    self.logger.info('reset the rules frequency')
    splunk_tools_instance.update_all_searches(splunk_tools_instance.update_search_cron_expression,'*/60 * * * *')
    self.logger.info('\n\n\n\n')