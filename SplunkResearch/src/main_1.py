import os
import urllib3
import datetime
import time
import json
import sys
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch')
from src.datetime_manager import MockedDatetimeManager

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
    # random_savedsearch = random.sample(savedsearches, num_of_searches)
    random_savedsearch = ['Monitor for Additions to Firewall Rules', 'Detected Registry Modification', 'Detect Network Connections to Non-Standard Ports', 'Detect Network Connections from Non-Browser or Non-Email Client', 'Disabled Security Tool', 'Multiple Network Connections to Same Port on External Hosts', 'Process Opened a Network Connection', 'Suspicious Remote Thread Creation', 'Modification of Executable File']
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


def permutation_experiment():
    # run the experiment for all the permutations of the parameters
    # search_window = [5, 15, 30, 60]
    # max actions value = [1000, 5000, 20000, 50000, 100000, 200000]
    # rule_frequency = 3
    # saved searches = [Monitor for Additions to Firewall Rules, Detected Registry Modification, Detect Network Connections to Non-Standard Ports, Detect Network Connections from Non-Browser or Non-Email Client, Disabled Security Tool, Multiple Network Connections to Same Port on External Hosts, Process Opened a Network Connection, Suspicious Remote Thread Creation, Modification of Executable File]
    reward_parameter_dict = {}
    rule_frequency = 2
    running_time="1.3"
    env_file_path = "/home/shouei/GreenSecurity-FirstExperiment/Scanner/.env"
    fake_start_datetime = datetime.datetime(2023,6,22, 8, 30, 0)
    manager = ExperimentManager() 
    global dt_manager
     # Create a new experiment directory
    current_dir = manager.create_experiment_dir()
    log_file_name = "log_train"
    dt_manager = MockedDatetimeManager(fake_start_datetime=fake_start_datetime, log_file_path=f"{current_dir}/{log_file_name}.txt")
    splunk_tools_instance = SplunkTools(dt_manager=dt_manager)
    print(f"New experiment directory: {current_dir}")
    num_of_searches = 9
    savedsearches = sorted(choose_random_rules(splunk_tools_instance, num_of_searches, get_only_enabled=True))
    relevant_logtypes =  list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]})
    log_generator_instance = LogGenerator(relevant_logtypes, big_replacement_dicts, splunk_tools_instance)
    num_of_episodes = 5  
    mode = 'test'
    action_values = [0, 1000, 20000, 50000, 200000][::-1]
    for max_actions_value in action_values:      
        is_first_episode = False   
        if max_actions_value != 200000:
            clean_env(splunk_tools_instance)
            is_first_episode = True     
        for search_window in [5, 15, 30, 60]:
                print('##########################################################################start##########################################################################')
                print('##########################################################################\n##########################################################################')
                dt_manager.set_fake_current_datetime(fake_start_datetime.strftime("%m/%d/%Y:%H:%M:%S"))              
                end_time = dt_manager.get_fake_current_datetime()
                start_time = dt_manager.subtract_time(end_time, minutes=search_window)
                time_range = (start_time, end_time)  
                self.logger.info(f'current time range: {time_range}')              
                # print all the rules that are running and the current parameters
                self.logger.info(f'current parameters:\ntime range:{time_range} \nfake_start_datetime: {fake_start_datetime}\nrule frequency: {rule_frequency}\nsearch_window:{search_window}\nrunning time: {running_time}\nnumber of searches: {num_of_searches}\nnumber of episodes: {num_of_episodes}\nmax action value: {max_actions_value}\nreward parameter dict: {reward_parameter_dict}\nsavedsearches: {savedsearches}\nrelevantlog_types: {relevant_logtypes}')
                # save parameters to file
                with open(f'{current_dir}/parameters_{mode}.json', 'w') as fp:
                    json.dump({'fake_start_datetime': str(fake_start_datetime), 'rule_frequency': rule_frequency, 'search_window': search_window, 'running_time': running_time, 'num_of_searches': num_of_searches, 'num_of_episodes': num_of_episodes, 'max_actions_value': max_actions_value, 'reward_parameter_dict': reward_parameter_dict, 'savedsearches': savedsearches, 'relevantlog_types': relevant_logtypes}, fp)
                
                update_running_time(running_time, env_file_path)
                update_rules_frequency_and_time_range(splunk_tools_instance, rule_frequency, time_range)
                gym.register(
                        id='splunk_attack-v0',
                        entry_point='framework:Framework',  # Replace with the appropriate path
                    )
                # clean_env(splunk_tools_instance, time_range)
                env = gym.make('splunk_attack-v0', log_generator_instance=log_generator_instance, splunk_tools_instance=splunk_tools_instance, dt_manager=dt_manager, time_range=time_range, rule_frequency=rule_frequency, search_window=search_window, reward_parameter_dict=reward_parameter_dict, relevant_logtypes=relevant_logtypes, max_actions_value=max_actions_value)
                env.time_action_dict[str(time_range)] = {}
                
                for i in range(num_of_episodes):
                    env.reset()
                    # clean_env(splunk_tools_instance, time_range) 
                    done = False
                    if not is_first_episode:
                        env.set_max_actions_value(0)
                    while not done:
                        action = 100/len(relevant_logtypes)
                        # inserted_logs += action*max_actions_value
                        obs, reward, done, info = env.step(action)
                        env.render()
                    if is_first_episode:
                        is_first_episode = False
                # inserted_logs = min(inserted_logs, max(action_values))
                save_assets(current_dir, env, mode=f'test_baseline_{search_window}_{max_actions_value}')               
                # clean_env(splunk_tools_instance, time_range) 
                self.logger.info('reset the rules frequency')
                splunk_tools_instance.update_all_searches(splunk_tools_instance.update_search_cron_expression,'*/60 * * * *')
                self.logger.info('\n\n\n\n')

                
def clean_env(splunk_tools_instance, time_range=('06/22/2023:00:00:00', '06/22/2023:23:59:59')):
    date = time_range[1].split(':')[0]
    time_range = (f'{date}:00:00:00', f'{date}:23:59:59')
    splunk_tools_instance.delete_fake_logs(time_range)
    return time_range

if __name__ == "__main__":
    permutation_experiment()