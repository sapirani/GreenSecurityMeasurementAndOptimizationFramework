
import random
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import datetime
import os
import inspect
import numpy as np
import urllib3
import json
import sys

from strategies.action_strategy import ActionStrategy8
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch')
import logging
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import custom_splunk #dont remove!!!
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C, PPO, DQN
import stable_baselines3 as sb3
urllib3.disable_warnings()
from stable_baselines3.common.logger import configure
from env_utils import *
from measurement import Measurement
from strategies.reward_strategy import *

import logging
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import HParam
logger = logging.getLogger(__name__)

class SaveModelCallback(BaseCallback):
    def __init__(self, verbose=1, save_freq=500, save_path=None):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
        return True
    
class SaveArtifacts(BaseCallback):
    def __init__(self, verbose=1, save_freq=500, experiment_menager=None):
        super(SaveArtifacts, self).__init__(verbose)
        self.save_freq = save_freq
        self.experiment_menager = experiment_menager

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            env = self.training_env.envs[0]
            self.experiment_menager.post_experiment(env, self.model)
        return True

class ModularTensorboardCallback(BaseCallback):
    def __init__(self, verbose=1, phase="train", experiment_kwargs=None):
        super(ModularTensorboardCallback, self).__init__(verbose)
        self.phase = phase  # This will determine whether it's "train" or "test"
        print("Experiment kwargs: ", experiment_kwargs)
        self.experiment_kwargs = experiment_kwargs
        self.episodic_metrics = ["alert", "duration", "std_duration", "cpu", "std_cpu", "read_bytes", "read_count", "write_bytes", "write_count", "total_cpu_usage"]
        self.reward_values = [ "p_values", "t_values", "degrees_of_freedom"] 
        self.started_measurements = False
        self.time_rules_energy_current_len = 0
        
    def _safe_log(self, key, value_list):
        # Check if value_list is a pandas Series or a list
        if isinstance(value_list, pd.Series):
            # For pandas Series, check if it's empty
            if not value_list.empty:
                self.logger.record(key, value_list.iloc[-1])
                return True
        elif isinstance(value_list, list) and len(value_list) > 0:
            # For lists, check if it's non-empty
            self.logger.record(key, value_list[-1])
            return True
        return False

    def log_common_metrics(self, env):

        # if self.started_measurements or random.randint(0, 100) == 2:
            # Record common metrics for both training and evaluation phases
        self._safe_log(f"{self.phase}/distribution_val", env.reward_calculator.reward_values_dict.get('distributions', []))
        self._safe_log(f"{self.phase}/distribution_reward", env.reward_calculator.reward_dict.get('distributions', []))
        self._safe_log(f"{self.phase}/quotas", env.action_strategy.action_quotas)

    def log_detailed_metrics(self, env, no_agent_last_row):
        # Log detailed metrics while safely checking list indices
        self._safe_log(f"{self.phase}/total_reward", env.reward_calculator.reward_dict.get('total', []))
        if len(env.reward_calculator.time_rules_energy) > self.time_rules_energy_current_len:
            self.time_rules_energy_current_len = len(env.reward_calculator.time_rules_energy)
            print(f"Time rules energy length: {self.time_rules_energy_current_len}")
            self._safe_log(f"{self.phase}/alert_reward", env.reward_calculator.reward_dict.get('alerts', []))
            self._safe_log(f"{self.phase}/duration_reward", env.reward_calculator.reward_dict.get('duration', []))
            self._safe_log(f"{self.phase}/final_distribution", env.reward_calculator.reward_dict.get('final_distribution', []))
            for metric in self.episodic_metrics:
                if len(env.reward_calculator.time_rules_energy):
                    success = self._safe_log(f"{self.phase}/{metric}", [env.reward_calculator.time_rules_energy[-1].get(metric, [])])
                    if success:
                        self.started_measurements = True
                    if success and no_agent_last_row is not None:
                        self._safe_log(f"{self.phase}/{metric}_gap", [env.reward_calculator.time_rules_energy[-1].get(metric, []) - no_agent_last_row[metric].values[-1]])
            for metric in self.reward_values:
                self._safe_log(f"{self.phase}/{metric}", env.reward_calculator.reward_values_dict.get(metric, []))


            # Rule-based metrics
            for metric in self.episodic_metrics:
                for i, rule in enumerate(env.splunk_tools_instance.active_saved_searches):
                    rule = rule['title']
                    if len(env.reward_calculator.time_rules_energy):
                        self.logger.record(f"{self.phase}/rules_{metric}", 
                                    {key.split(f'rule_{metric}_')[1]: env.reward_calculator.time_rules_energy[-1][key]
                                        for key in env.reward_calculator.time_rules_energy[-1].keys() if key.startswith(f'rule_{metric}')})
                        if no_agent_last_row is not None:
                            self.logger.record(f"{self.phase}/rules_{metric}_gap", 
                                    {key.split(f'rule_{metric}_')[1]: env.reward_calculator.time_rules_energy[-1][key] - no_agent_last_row[key].values[-1]
                                        for key in env.reward_calculator.time_rules_energy[-1].keys() if key.startswith(f'rule_{metric}')})
                        
            # self.logger.record(f"{self.phase}/rules_alert_reward", {rule['title']: env.reward_calculator.current_rules_alert_reward.get(rule['title'], []) for rule in env.splunk_tools_instance.active_saved_searches})                    
            # log episodic policy
            policy_dict = {}
            for i, logtype in enumerate(env.relevant_logtypes):
                for is_trigger in range(2):
                    if isinstance(env.action_strategy, ActionStrategy8):
                        policy_dict[f'{logtype}_0'] = env.action_per_episode[-1][i]
                    else:
                        policy_dict[f'{logtype}_{is_trigger}'] = env.action_per_episode[-1][i*2+is_trigger]
                    if i == len(env.relevant_logtypes)-1:
                        break 
            self.logger.record(f"{self.phase}/episodic_policy", policy_dict)

    def log_no_agent_metrics(self, env, no_agent_last_row):
        if len(env.reward_calculator.time_rules_energy) > self.time_rules_energy_current_len:
            # Log metrics related to no-agent scenario while safely checking list indices
            if no_agent_last_row is None:
                return
            self._safe_log(f"{self.phase}/no_agent_alert_val", no_agent_last_row.get('alert', []))
            self._safe_log(f"{self.phase}/no_agent_duration_val", no_agent_last_row.get('duration', []))
            self._safe_log(f"{self.phase}/no_agent_cpu", no_agent_last_row.get('cpu', []))
            self._safe_log(f"{self.phase}/no_agent_total_cpu_usage", no_agent_last_row.get('total_cpu_usage', []))

            self.logger.record(f"{self.phase}/no_agent_rules_alerts", {col.split('rule_alert_')[1]: no_agent_last_row[col].values[-1] for col in no_agent_last_row.columns if col.startswith('rule_alert')})
            self.logger.record(f"{self.phase}/no_agent_rules_duration", {col.split('rule_duration_')[1]: no_agent_last_row[col].values[-1] for col in no_agent_last_row.columns if col.startswith('rule_duration')})
            self.logger.record(f"{self.phase}/no_agent_rules_std_duration", {col.split('rule_std_duration')[1]: no_agent_last_row[col].values[-1] for col in no_agent_last_row.columns if col.startswith('rule_std_duration')})
            self.logger.record(f"{self.phase}/logs_amount", env.time_range_logs_amount[-1])
        
    


class HparamsCallback(BaseCallback):
    def __init__(self, verbose=1, experiment_kwargs=None, phase="train" ):
        super(HparamsCallback, self).__init__(verbose)
        self.experiment_kwargs = experiment_kwargs
        self.phase = phase
    def _on_training_start(self) -> None:
        metric_dict = {f"{self.phase}/{tag}": 0 for tag in ["distribution_val", "distribution_reward", "alert_reward", "duration_reward", "total_reward", "alert_val", "duration_val", "duration_gap", "alert_gap", "no_agent_alert_val", "no_agent_duration_val"]}
        # add to metric_dict the default metrics
        metric_dict.update({f"rollout/{tag}": 0 for tag in ["ep_rew_mean", "ep_rew_std", "ep_len_mean", "ep_len_std"]})
        hparams = HParam(self.experiment_kwargs, metric_dict)
        self.logger.record("hparams", hparams, exclude=("log", "json", "csv"))
    
    def _on_step(self) -> bool:
        return True

class TrainTensorboardCallback(ModularTensorboardCallback):
    def __init__(self, verbose=1, experiment_kwargs=None, phase="train"):
        super(TrainTensorboardCallback, self).__init__(verbose, phase=phase, experiment_kwargs=experiment_kwargs)

    # def _on_rollout_end(self) -> None:
    #     env = self.training_env.envs[0]
    #     no_agent_last_row = env.reward_calculator.no_agent_last_row
    #     self.log_no_agent_metrics(env, no_agent_last_row)
    #     self.log_detailed_metrics(env, no_agent_last_row)

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        current_step = env.all_steps_counter
        # if current_step == 2:
        #     no_agent_last_row = env.reward_calculator.no_agent_last_row
        #     self.log_no_agent_metrics(env, no_agent_last_row)
        self.log_common_metrics(env)
        if self.locals.get('dones')[0]:#env.done:#
            no_agent_last_row = env.reward_calculator.no_agent_last_row
            # no_agent_last_row = None
            self.log_no_agent_metrics(env, no_agent_last_row)
            self.log_detailed_metrics(env, no_agent_last_row)
        self.logger.dump(current_step)
        return True

    def _on_training_end(self) -> None:
        env = self.training_env.envs[0]
        start_time_datetime = datetime.datetime.strptime(env.fake_start_datetime, '%m/%d/%Y:%H:%M:%S')#datetime.datetime.strptime(kwargs['fake_start_datetime'], '%m/%d/%Y:%H:%M:%S')
        end_time_datetime = datetime.datetime.strptime(env.time_range[1], '%m/%d/%Y:%H:%M:%S')
        clean_env(env.splunk_tools_instance, (start_time_datetime.timestamp(), end_time_datetime.timestamp()))
    
    # def _on_training_start(self) -> None:
    #     super()._on_training_start()
    #     self.log_hparams(self.experiment_kwargs)
    

        
# def eval_tensorboard_callback(locals_dict, globals_dict):
#     env = locals_dict["env"].envs[0]
#     model = locals_dict["model"]
#     tb_logger = model.logger
#     current_step = env.step_counter
#     done_episodes = len(locals_dict["episode_lengths"])
#     total_steps = env.total_steps
#     no_agent_last_row = env.reward_calculator.no_agent_last_row

#     # Use the shared ModularTensorboardCallback for logging
#     modular_callback = ModularTensorboardCallback(phase="test")
#     modular_callback.logger = tb_logger
#     modular_callback.log_common_metrics(env)
#     if current_step == 2:
#         modular_callback.log_no_agent_metrics(env, no_agent_last_row)
#         tb_logger.dump(done_episodes * total_steps + current_step - 1)
#     elif current_step == 1:
#         modular_callback.log_detailed_metrics(env, no_agent_last_row)
#         tb_logger.dump(done_episodes * total_steps + total_steps)
#     else:
#         tb_logger.dump(done_episodes * total_steps + current_step - 1)

#     return True