
import json
import sys
from logtypes import logtypes
from config import replacement_dicts
import logging
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch')
from splunk_tools import SplunkTools
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
import random
import subprocess
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import gym
from stable_baselines3 import A2C
from log_generator import LogGenerator
from config import replacement_dicts as big_replacement_dicts



def update_running_time(running_time, env_file_path):
    command = f'sed -i "s/RUNNING_TIME=.*/RUNNING_TIME={running_time}/" "{env_file_path}"'
    res = subprocess.run(command, shell=True, capture_output=True, text=True)
    logging.info(res.stdout)
    logging.info(res.stderr)

def choose_random_rules(env, num_of_searches):
    logging.info('enable random rules')
    savedsearches = env.splunk_tools.get_saved_search_names(True)
    random_savedsearch = random.sample(savedsearches, num_of_searches)
    for savedsearch in random_savedsearch:
        env.splunk_tools.enable_search(savedsearch)
    logging.info(f'current running rules: {random_savedsearch}')
    return random_savedsearch

def update_rules_frequency_and_time_range(env, rule_frequency, time_range):
    logging.info('update rules frequency')
    env.splunk_tools.update_all_searches(env.splunk_tools.update_search_cron_expression, f'*/{rule_frequency} * * * *')
    logging.info('update time range of rules')
    env.splunk_tools.update_all_searches(env.splunk_tools.update_search_time_range, time_range)   





if __name__ == "__main__":
    logging.info('##########################################################################start##########################################################################')
    rule_frequency = 5
    date = '06/22/2023'
    time_range = (f'{date}:08:00:00', f'{date}:08:05:00')
    max_actions_value = 400
    num_of_searches = 5
    running_time="1" #in minutes
    num_of_experiments = 1000
    baseline = False
    env_file_path = "/home/shouei/GreenSecurity-FirstExperiment/Scanner/.env"
    
    # print all the rules that are running and the current parameters
    logging.info('current parameters:')
    logging.info(f"No DRL experiment") if baseline else logging.info(f"DRL experiment")
    logging.info(f"time range: {time_range}")
    logging.info(f"rule frequency: {rule_frequency}")
    logging.info(f"running time: {running_time}")
    logging.info(f"number of searches: {num_of_searches}")
    logging.info(f"number of experiments: {num_of_experiments}")
    logging.info(f"max action value: {max_actions_value}")
    
    
    splunk_tools_instance = SplunkTools()
    log_generator_instance = LogGenerator(logtypes, big_replacement_dicts, splunk_tools_instance)
        
    gym.register(
            id='splunk_attack-v0',
            entry_point='framework:Framework',  # Replace with the appropriate path
        )
    env = gym.make('splunk_attack-v0', log_generator_instance = log_generator_instance, splunk_tools_instance = splunk_tools_instance, time_range=time_range, rule_frequency=rule_frequency, baseline=baseline, max_actions_value=max_actions_value)
    
    # env.splunk_tools.update_all_searches(env.splunk_tools._update_search, {"disabled": 1})
    update_running_time(running_time, env_file_path)
    savedsearches = choose_random_rules(env, num_of_searches)
    update_rules_frequency_and_time_range(env, rule_frequency, time_range)
    
    # env = Framework(replacement_dicts, time_range, rule_frequency, max_actions_value)
    if not baseline:
        model = A2C(MlpPolicy, env, verbose=3)
        # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        # logging.info(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        logging.info('start learning')
        model.learn(total_timesteps=num_of_experiments)
        # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        # logging.info(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        logging.info('finish learning, saving model')
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
                logging.info('no action first')
                state, reward, done, _ = env.step(no_action)
                random_action = env.action_space.sample()
                state, reward, done, _ = env.step(random_action)
                env.reset()
            else:
                logging.info('random action first')
                random_action = env.action_space.sample()
                state, reward, done, _ = env.step(random_action)
                env.reset()
                state, reward, done, _ = env.step(no_action)

    logging.info('reset the rules frequency')
    env.splunk_tools.update_all_searches(env.splunk_tools.update_search_cron_expression,'*/60 * * * *')
    logging.info('\n\n\n\n')