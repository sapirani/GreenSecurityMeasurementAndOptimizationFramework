import urllib3
from utils import MockedDatetimeManager
import datetime
import time
import json
import sys
from logtypes import logtypes
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

def choose_random_rules(splunk_tools_instance, num_of_searches):
    dt_manager.log('enable random rules')
    savedsearches = splunk_tools_instance.get_saved_search_names(get_only_enabled=True)
    random_savedsearch = random.sample(savedsearches, num_of_searches)
    for savedsearch in savedsearches:
        if savedsearch not in random_savedsearch:
            splunk_tools_instance.disable_search(savedsearch)
        else:
            splunk_tools_instance.enable_search(savedsearch)
    dt_manager.log(f'current running rules: {random_savedsearch}')
    return random_savedsearch

def update_rules_frequency_and_time_range(splunk_tools_instance, rule_frequency, time_range):
    dt_manager.log('update rules frequency')
    splunk_tools_instance.update_all_searches(splunk_tools_instance.update_search_cron_expression, f'*/{rule_frequency} * * * *')
    dt_manager.log('update time range of rules')
    splunk_tools_instance.update_all_searches(splunk_tools_instance.update_search_time_range, time_range)   





if __name__ == "__main__":
    print('##########################################################################start##########################################################################')
    mode = sys.argv[1]
    rule_frequency = 5
    max_actions_value = 400
    num_of_searches = 5
    running_time="1.5" #in minutes
    num_of_episodes = 50
    baseline = False
    env_file_path = "/home/shouei/GreenSecurity-FirstExperiment/Scanner/.env"
    fake_start_datetime = datetime.datetime(2023,6,22, 8, 0, 0)
    
    manager = ExperimentManager()
    if mode == 'train':
        # Create a new experiment directory
        current_dir = manager.create_experiment_dir()
        print(f"New experiment directory: {current_dir}")
        log_file_name = "log_train"
    else:
        current_dir = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/exp_20230823_154412'
        print(f"Current directory: {current_dir}")
        log_file_name = "log_test"
    
    
    global dt_manager
    dt_manager = MockedDatetimeManager(fake_start_datetime=fake_start_datetime, log_file_path=f"{current_dir}/{log_file_name}.txt")
    update_running_time(running_time, env_file_path)
    
    splunk_tools_instance = SplunkTools(dt_manager=dt_manager)
    # splunk_tools_instance.update_all_searches(splunk_tools_instance._update_search, {"disabled": 1})
    savedsearches = choose_random_rules(splunk_tools_instance, num_of_searches)
    log_generator_instance = LogGenerator(logtypes, big_replacement_dicts, splunk_tools_instance)
        

    
    end_time = dt_manager.get_fake_current_datetime()
    start_time = dt_manager.subtract_time(end_time, minutes=rule_frequency)
    time_range = (start_time, end_time)
    update_rules_frequency_and_time_range(splunk_tools_instance, rule_frequency, time_range)
    
    
    
    # print all the rules that are running and the current parameters
    dt_manager.log('current parameters:')
    dt_manager.log(f"No DRL experiment") if baseline else dt_manager.log(f"DRL experiment")
    dt_manager.log(f"fake_start_datetime: {fake_start_datetime}")
    dt_manager.log(f"rule frequency: {rule_frequency}")
    dt_manager.log(f"running time: {running_time}")
    dt_manager.log(f"number of searches: {num_of_searches}")
    dt_manager.log(f"number of episodes: {num_of_episodes}")
    dt_manager.log(f"max action value: {max_actions_value}")
    

    gym.register(
            id='splunk_attack-v0',
            entry_point='framework:Framework',  # Replace with the appropriate path
        )
    env = gym.make('splunk_attack-v0', log_generator_instance = log_generator_instance, splunk_tools_instance = splunk_tools_instance, dt_manager=dt_manager, time_range=time_range, rule_frequency=rule_frequency, baseline=baseline, max_actions_value=max_actions_value)
    
    # wait till the time is rounded to the next rule frequency
    dt_manager.log('wait till the real time is rounded to the next rule frequency')
    dt_manager.wait_til_next_rule_frequency(rule_frequency)
    env.before_first_step()
    
    # env = Framework(replacement_dicts, time_range, rule_frequency, max_actions_value)
    if mode == 'train':
        model = A2C(MlpPolicy, env, verbose=3)
        dt_manager.log('start learning')
        model.learn(total_timesteps=len(logtypes)*num_of_episodes)
        dt_manager.log('finish learning, saving model')
        model.save(f"{current_dir}/A2C_splunk_attack")
        with open(f'{current_dir}/reward_dict.json', 'w') as fp:
            json.dump(env.reward_dict, fp)
        with open(f'{current_dir}/reward_values_dict_train.json', 'w') as fp:
            json.dump(env.reward_values_dict, fp)
    
    else:
        dt_manager.log('Testing the model')
        dt_manager.log('load and run the DRL method')
        # load the model
        model = A2C.load(f"{current_dir}/A2C_splunk_attack")
        # evaluate the model
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        dt_manager.log(f"mean_reward:{mean_reward}, std_reward:{std_reward}")
        with open(f'{current_dir}/reward_values_dict_model_test.json', 'w') as fp:
                json.dump(env.reward_values_dict, fp)
        
        # run the baseline method, which is that instead of using the model to predict the next action, we use the random action
        dt_manager.log('run the baseline method')
        env = gym.make('splunk_attack-v0', log_generator_instance = log_generator_instance, splunk_tools_instance = splunk_tools_instance, dt_manager=dt_manager, time_range=time_range, rule_frequency=rule_frequency, baseline=baseline, max_actions_value=max_actions_value)
        env.before_first_step()
        for i in range(10):
            env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                env.render()
                time.sleep(1)     
        with open(f'{current_dir}/reward_values_dict_baseline1.json', 'w') as fp:
            json.dump(env.reward_values_dict, fp)
        
        # run the second baseline method, which is to perform actions at all
        dt_manager.log('run the second baseline method')
        env = gym.make('splunk_attack-v0', log_generator_instance = log_generator_instance, splunk_tools_instance = splunk_tools_instance, dt_manager=dt_manager, time_range=time_range, rule_frequency=rule_frequency, baseline=baseline, max_actions_value=max_actions_value)
        env.before_first_step()
        for i in range(10):
            env.reset()
            done = False
            while not done:
                action = [0]
                obs, reward, done, info = env.step(action)
                env.render()
                time.sleep(1)     
        with open(f'{current_dir}/reward_values_dict_baseline2.json', 'w') as fp:
            json.dump(env.reward_values_dict, fp)

    dt_manager.log('reset the rules frequency')
    env.splunk_tools.update_all_searches(env.splunk_tools.update_search_cron_expression,'*/60 * * * *')
    dt_manager.log('\n\n\n\n')