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

now = datetime.datetime.now()
fake_start_datetime = datetime.datetime(2023,6,22, now.hour, now.minute, now.second)
dt_manager = MockedDatetimeManager(fake_start_datetime=fake_start_datetime)

def update_running_time(running_time, env_file_path):
    command = f'sed -i "s/RUNNING_TIME=.*/RUNNING_TIME={running_time}/" "{env_file_path}"'
    res = subprocess.run(command, shell=True, capture_output=True, text=True)
    dt_manager.log(res.stdout)
    dt_manager.log(res.stderr)

def choose_random_rules(splunk_tools_instance, num_of_searches):
    dt_manager.log('enable random rules')
    savedsearches = splunk_tools_instance.get_saved_search_names(True)
    random_savedsearch = random.sample(savedsearches, num_of_searches)
    for savedsearch in random_savedsearch:
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
    rule_frequency = 5
    max_actions_value = 400
    num_of_searches = 5
    running_time="1" #in minutes
    num_of_experiments = 1000
    baseline = False
    env_file_path = "/home/shouei/GreenSecurity-FirstExperiment/Scanner/.env"
    
    
    
    
    
    # print(dt_manager.get_current_datetime())  # Mocked datetime
    # time.sleep(5)  # Wait for 5 seconds
    # print(dt_manager.get_current_datetime())  # Mocked datetime, 5 seconds later
    # dt_manager.log("This should use the real datetime.")  # Real datetime
    
    splunk_tools_instance = SplunkTools()
    # splunk_tools_instance.update_all_searches(splunk_tools_instance._update_search, {"disabled": 1})
    update_running_time(running_time, env_file_path)
    savedsearches = choose_random_rules(splunk_tools_instance, num_of_searches)
    
    log_generator_instance = LogGenerator(logtypes, big_replacement_dicts, splunk_tools_instance)
        
    # wait till the time is rounded to the next rule frequency
    dt_manager.log('wait till the time is rounded to the next rule frequency')
    dt_manager.wait_til_next_rule_frequency(rule_frequency)
    fake_now = dt_manager.get_current_datetime()
    time_range = (dt_manager.add_time(fake_now, seconds=20), dt_manager.add_time(fake_now, minutes=rule_frequency))
    update_rules_frequency_and_time_range(splunk_tools_instance, rule_frequency, time_range)
    gym.register(
            id='splunk_attack-v0',
            entry_point='framework:Framework',  # Replace with the appropriate path
        )
    env = gym.make('splunk_attack-v0', log_generator_instance = log_generator_instance, splunk_tools_instance = splunk_tools_instance, dt_manager=dt_manager, time_range=time_range, rule_frequency=rule_frequency, baseline=baseline, max_actions_value=max_actions_value)
    # print all the rules that are running and the current parameters
    dt_manager.log('current parameters:')
    dt_manager.log(f"No DRL experiment") if baseline else dt_manager.log(f"DRL experiment")
    dt_manager.log(f"time range: {time_range}")
    dt_manager.log(f"rule frequency: {rule_frequency}")
    dt_manager.log(f"running time: {running_time}")
    dt_manager.log(f"number of searches: {num_of_searches}")
    dt_manager.log(f"number of experiments: {num_of_experiments}")
    dt_manager.log(f"max action value: {max_actions_value}")
    
    env.before_first_step()
    
    # env = Framework(replacement_dicts, time_range, rule_frequency, max_actions_value)
    if not baseline:
        model = A2C(MlpPolicy, env, verbose=3)
        # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        # dt_manager.log(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        dt_manager.log('start learning')
        model.learn(total_timesteps=num_of_experiments)
        # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        # dt_manager.log(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        dt_manager.log('finish learning, saving model')
        model.save("ppo_splunk_attack")
        with open('reward_dict.json', 'w') as fp:
            json.dump(env.reward_dict, fp)
    else:
        env.reset()
        no_action = [0 for _ in range(len(logtypes))]
        for _ in range(num_of_experiments):
            random_num = random.randint(0,3)
            env.replacement_dicts = {field.lower():{key: random.choice(value) for key, value in replacement_dicts[field].items()} for field in replacement_dicts}
            if random_num == 0:
                dt_manager.log('no action first')
                state, reward, done, _ = env.step(no_action)
                random_action = env.action_space.sample()
                state, reward, done, _ = env.step(random_action)
                env.reset()
            else:
                dt_manager.log('random action first')
                random_action = env.action_space.sample()
                state, reward, done, _ = env.step(random_action)
                env.reset()
                state, reward, done, _ = env.step(no_action)

    dt_manager.log('reset the rules frequency')
    env.splunk_tools.update_all_searches(env.splunk_tools.update_search_cron_expression,'*/60 * * * *')
    dt_manager.log('\n\n\n\n')