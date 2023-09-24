import os
import urllib3
from utils import MockedDatetimeManager
import datetime
import time
import json
import sys
from logtypes import logtypes
from section_logtypes import section_logtypes
from config import replacement_dicts
import logging
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch')
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
from config import replacement_dicts as big_replacement_dicts
urllib3.disable_warnings()



def update_running_time(running_time, env_file_path):
    command = f'sed -i "s/RUNNING_TIME=.*/RUNNING_TIME={running_time}/" "{env_file_path}"'
    res = subprocess.run(command, shell=True, capture_output=True, text=True)
    dt_manager.log(res.stdout)
    dt_manager.log(res.stderr)

def choose_random_rules(splunk_tools_instance, num_of_searches, get_only_enabled=True):
    dt_manager.log('enable random rules')
    savedsearches = splunk_tools_instance.get_saved_search_names(get_only_enabled=get_only_enabled)
    random_savedsearch = random.sample(savedsearches, num_of_searches)
    for savedsearch in savedsearches:
        if savedsearch not in random_savedsearch:
            splunk_tools_instance.disable_search(savedsearch)
        else:
            splunk_tools_instance.enable_search(savedsearch)
    return random_savedsearch

def update_rules_frequency_and_time_range(splunk_tools_instance, rule_frequency, time_range):
    dt_manager.log('update rules frequency')
    splunk_tools_instance.update_all_searches(splunk_tools_instance.update_search_cron_expression, f'*/{rule_frequency} * * * *')
    dt_manager.log('update time range of rules')
    splunk_tools_instance.update_all_searches(splunk_tools_instance.update_search_time_range, time_range)   





def save_assets(current_dir, env, mode='train'):
    print(env.reward_calculator.reward_values_dict)
    # create a directory for the assets if not exists
    if not os.path.exists(f'{current_dir}/{mode}'):
        os.makedirs(f'{current_dir}/{mode}')
    with open(f'{current_dir}/{mode}/reward_dict.json', 'w') as fp:
        json.dump(env.reward_calculator.reward_dict, fp)
    with open(f'{current_dir}/{mode}/reward_values_dict.json', 'w') as fp:
        json.dump(env.reward_calculator.reward_values_dict, fp)
    with open(f'{current_dir}/{mode}/time_rules_energy_dict.json', 'w') as fp:
        json.dump(env.reward_calculator.time_rules_energy_dict, fp)   
    with open(f'{current_dir}/{mode}/action_dict_{mode}.json', 'w') as fp:
            json.dump(env.time_action_dict, fp)

def evaluate_model(rule_frequency, max_actions_value, num_of_episodes, fake_start_datetime, reward_parameter_dict, current_dir, dt_manager, splunk_tools_instance, relevant_logtypes, time_range, log_generator_instance):
    dt_manager.set_fake_current_datetime(fake_start_datetime.strftime("%m/%d/%Y:%H:%M:%S"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dt_manager.log('Testing the model')
    env = gym.make('splunk_attack-v0', log_generator_instance = log_generator_instance, splunk_tools_instance = splunk_tools_instance, dt_manager=dt_manager, time_range=time_range, rule_frequency=rule_frequency, reward_parameter_dict=reward_parameter_dict, relevant_logtypes=relevant_logtypes, max_actions_value=max_actions_value)
    dt_manager.log('load and run the DRL method')
        # load the model
    model = A2C.load(f"{current_dir}/A2C_splunk_attack")
    # dt_manager.log('first measurement')
    # env.reward_calculator.get_full_reward_values(time_range)
        # evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_of_episodes)
    dt_manager.log(f"mean_reward:{mean_reward}, std_reward:{std_reward}")
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
    max_actions_value = 3000
    num_of_searches = 5
    running_time="1" #in minutes
    num_of_episodes = 200
    env_file_path = "/home/shouei/GreenSecurity-FirstExperiment/Scanner/.env"
    fake_start_datetime = datetime.datetime(2023,6,22, 8, 0, 0)
    reward_parameter_dict ={'alpha': 0.6, 'beta': 0.05, 'gamma': 0.15, 'delta': 0.20}
    manager = ExperimentManager() 
    global dt_manager
        
    if mode == 'train':
        # Create a new experiment directory
        current_dir = manager.create_experiment_dir()
        log_file_name = "log_train"
        dt_manager = MockedDatetimeManager(fake_start_datetime=fake_start_datetime, log_file_path=f"{current_dir}/{log_file_name}.txt")
        splunk_tools_instance = SplunkTools(dt_manager=dt_manager)
        print(f"New experiment directory: {current_dir}")
        savedsearches = sorted(choose_random_rules(splunk_tools_instance, num_of_searches, get_only_enabled=False))
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
            
    


    # savedsearches = ['Creation of a New Local User Account', 'Detect Network Connections to Non-Standard Ports', 'Detected Registry Modification', 'Monitor for Administrative and Guest Logon Failures', 'Monitor for Changes to Firewall Rules', 'Monitor for Logon Success', 'Monitor for Registry Changes', 'Multiple Failed Logins from the Same Source', 'User Added to Privileged Group', 'User Login with Local Credentials']
    # relevant_logtypes = [('wineventlog:security', '4657'), ('wineventlog:security', '4625'), ('wineventlog:security', '2005'), ('xmlwineventlog:microsoft-windows-sysmon/operational', '14'), ('xmlwineventlog:microsoft-windows-sysmon/operational', '3'), ('wineventlog:security', '4756'), ('wineventlog:security', '4728'), ('wineventlog:security', '4624'), ('wineventlog:security', '4732'), ('xmlwineventlog:microsoft-windows-sysmon/operational', '12'), ('xmlwineventlog:microsoft-windows-sysmon/operational', '13'), ('wineventlog:security', '4720')]
    end_time = dt_manager.get_fake_current_datetime()
    start_time = dt_manager.subtract_time(end_time, minutes=rule_frequency)
    time_range = (start_time, end_time)
    log_generator_instance = LogGenerator(relevant_logtypes, big_replacement_dicts, splunk_tools_instance)
    
    # print all the rules that are running and the current parameters
    dt_manager.log('current parameters:')
    dt_manager.log(f"fake_start_datetime: {fake_start_datetime}")
    dt_manager.log(f"rule frequency: {rule_frequency}")
    dt_manager.log(f"running time: {running_time}")
    dt_manager.log(f"number of searches: {num_of_searches}")
    dt_manager.log(f"number of episodes: {num_of_episodes}")
    dt_manager.log(f"max action value: {max_actions_value}")
    dt_manager.log(f"reward parameter dict: {reward_parameter_dict}")
    dt_manager.log(f"savedsearches: {savedsearches}")
    
    # save parameters to file
    with open(f'{current_dir}/parameters_{mode}.json', 'w') as fp:
        json.dump({'fake_start_datetime': str(fake_start_datetime), 'rule_frequency': rule_frequency, 'running_time': running_time, 'num_of_searches': num_of_searches, 'num_of_episodes': num_of_episodes, 'max_actions_value': max_actions_value, 'reward_parameter_dict': reward_parameter_dict, 'savedsearches': savedsearches, 'relevantlog_types': relevant_logtypes}, fp)
    
    update_running_time(running_time, env_file_path)
    update_rules_frequency_and_time_range(splunk_tools_instance, rule_frequency, time_range)
    gym.register(
            id='splunk_attack-v0',
            entry_point='framework:Framework',  # Replace with the appropriate path
        )
    clean_env(splunk_tools_instance, time_range) 
    if mode == 'train':
        env = gym.make('splunk_attack-v0', log_generator_instance = log_generator_instance, splunk_tools_instance = splunk_tools_instance, dt_manager=dt_manager, time_range=time_range, rule_frequency=rule_frequency, reward_parameter_dict=reward_parameter_dict, relevant_logtypes=relevant_logtypes, max_actions_value=max_actions_value)
        clean_env(splunk_tools_instance, time_range)         
        # wait till the time is rounded to the next rule frequency
        dt_manager.log('wait till the real time is rounded to the next rule frequency')
        # dt_manager.log('first measurement')
        # env.reward_calculator.get_full_reward_values(time_range)
        model = A2C(MlpPolicy, env, verbose=3)
        dt_manager.log('start learning')
        model.learn(total_timesteps=len(relevant_logtypes)*num_of_episodes)
        dt_manager.log('finish learning, saving model')
        model.save(f"{current_dir}/A2C_splunk_attack")
        save_assets(current_dir, env)
    elif mode == 'retrain':
        env = gym.make('splunk_attack-v0', log_generator_instance = log_generator_instance, splunk_tools_instance = splunk_tools_instance, dt_manager=dt_manager, time_range=time_range, rule_frequency=rule_frequency, reward_parameter_dict=reward_parameter_dict, relevant_logtypes=relevant_logtypes, max_actions_value=max_actions_value)
        # wait till the time is rounded to the next rule frequency
        dt_manager.log('wait till the real time is rounded to the next rule frequency')
        # dt_manager.log('first measurement')
        # env.reward_calculator.get_full_reward_values(time_range)
        model = A2C.load(f"{current_dir}/A2C_splunk_attack", env=env, verbose=3)
        dt_manager.log('start learning')
        model.learn(total_timesteps=len(relevant_logtypes)*num_of_episodes)
        dt_manager.log('finish learning, saving model')
        model.save(f"{current_dir}/A2C_splunk_attack")
        save_assets(current_dir, env)
    else:
        num_of_episodes = 20
        evaluate_model(rule_frequency, max_actions_value, num_of_episodes, fake_start_datetime, reward_parameter_dict, current_dir, dt_manager, splunk_tools_instance, relevant_logtypes, time_range, log_generator_instance)   

        # run the baseline method, which is that instead of using the model to predict the next action, we use the random action
        dt_manager.set_fake_current_datetime(fake_start_datetime.strftime("%m/%d/%Y:%H:%M:%S"))
        dt_manager.log('\n\nrun the baseline method\n')
        clean_env(splunk_tools_instance, time_range)         
        env = gym.make('splunk_attack-v0', log_generator_instance = log_generator_instance, splunk_tools_instance = splunk_tools_instance, dt_manager=dt_manager, time_range=time_range, rule_frequency=rule_frequency, reward_parameter_dict=reward_parameter_dict,  relevant_logtypes=relevant_logtypes, max_actions_value=max_actions_value)
        # dt_manager.log('first measurement')
        # env.reward_calculator.get_full_reward_values(time_range)
        for i in range(num_of_episodes):
            env.reset()
            done = False
            while not done:
                action = [random.uniform(0, (1-env.sum_of_fractions))]
                obs, reward, done, info = env.step(action)
                env.render()
        save_assets(current_dir, env, mode='test_baseline1')   

        
        # run the second baseline method, which is to not perform actions at all
        max_actions_value = 0
        dt_manager.set_fake_current_datetime(fake_start_datetime.strftime("%m/%d/%Y:%H:%M:%S"))
        dt_manager.log('\n\nrun the second baseline method\n')
        clean_env(splunk_tools_instance, time_range)         
        env = gym.make('splunk_attack-v0', log_generator_instance = log_generator_instance, splunk_tools_instance = splunk_tools_instance, dt_manager=dt_manager, time_range=time_range, rule_frequency=rule_frequency, reward_parameter_dict=reward_parameter_dict,  relevant_logtypes=relevant_logtypes, max_actions_value=max_actions_value)
        # dt_manager.log('first measurement')
        # env.reward_calculator.get_full_reward_values(time_range)
        for i in range(num_of_episodes):
            env.reset()
            done = False
            while not done:
                action = [0]
                obs, reward, done, info = env.step(action)
                env.render()
        save_assets(current_dir, env, mode='test_baseline2')   
    # clean_env(splunk_tools_instance, time_range) 
    dt_manager.log('reset the rules frequency')
    env.splunk_tools.update_all_searches(env.splunk_tools.update_search_cron_expression,'*/60 * * * *')
    dt_manager.log('\n\n\n\n')