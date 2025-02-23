import datetime
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import numpy as np
import logging
from stable_baselines3.common.logger import HParam
from env_utils import *
logger = logging.getLogger(__name__)

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1, phase="train", experiment_kwargs=None):
        super().__init__(verbose)
        self.phase = phase
        self.experiment_kwargs = experiment_kwargs
        self.episodic_metrics = [
            "alert", "duration", "cpu", "std_cpu", "std_duration", 
            "read_bytes", "read_count", "write_bytes", "write_count",
            "total_cpu_usage"
        ]
        self.started_measurements = False

    def _log_metrics(self, key: str, value, exclude_from_csv=True):
        """Safely log metrics to tensorboard"""
        if value is not None:
            self.logger.record(f"{self.phase}/{key}", value)


    def _on_step(self) -> bool:
        """Log metrics at each step"""
        info = self.locals['infos'][0]  # Get info from environment step
        
        # Log basic step information
        step_counter = info.get('step', 0)
        all_steps = info.get('all_steps_counter', 0)
        action_window = info.get('action_window')
        if action_window:
            self._log_metrics('action_window', f"{action_window}")

        # Log reward components
        for reward_type in ['distribution_reward', 'energy_reward', 'alert_reward']:
            if reward_type in info:
                self._log_metrics(reward_type, info[reward_type])
        
        # # Log distributions
        # real_dist = info.get('real_distribution')
        # fake_dist = info.get('fake_distribution')
        
        # self._log_metrics('real_distribution', real_dist, exclude_from_csv=True)
        # self._log_metrics('fake_distribution', fake_dist, exclude_from_csv=True)
        
        self._log_metrics('distribution_value', info.get('distribution_value'))

        # Log quota information
        if 'inserted_logs' in info:
            self._log_metrics('inserted_logs', info['inserted_logs'])
            
        if 'total_current_logs' in info:
            self._log_metrics('total_current_logs', info['total_current_logs'])
        
        if 'inserted_logs' in info and 'total_current_logs' in info:
            self._log_metrics('actual_quota', info['inserted_logs']/info['total_current_logs'])

        # Log episode end metrics
        if info.get('done', False):
            self._log_episode_metrics(info) 
        self.logger.dump(all_steps)
        return True

    def _log_episode_metrics(self, info):
        """Log metrics at episode end"""
        
        # Log rules execution metrics
        current_metrics = info.get('combined_metrics', {})
        baseline_metrics = info.get('combined_baseline_metrics', {})
        
        current_window = info.get('current_window')
        if current_window:
            self._log_metrics('current_window', f"{current_window}", exclude_from_csv=True)
            
        # compute and log total reward
        env = self.training_env.envs[0]
        total_reward = env.alpha*info.get('energy_reward', 0) + env.beta*info.get('alert_reward', 0) + env.gamma*info.get('distribution_reward', 0)
        self._log_metrics('total_reward', total_reward)
        
        
        
        if current_metrics and baseline_metrics:
            # Log basic metrics
            for metric in self.episodic_metrics:
                current_val = current_metrics.get(metric)
                baseline_val = baseline_metrics.get(metric)
                if current_val is not None and baseline_val is not None:
                    self._log_metrics(f'{metric}', current_val)
                    self._log_metrics(f'baseline_{metric}', baseline_val)
                    self._log_metrics(f'{metric}_gap', current_val - baseline_val)

            raw_current_metrics = info.get('raw_metrics', {})
            raw_baseline_metrics = info.get('raw_baseline_metrics', {})
            baseline_dict = {metric: {} for metric in self.episodic_metrics}
            current_dict = {metric: {} for metric in self.episodic_metrics}
            gap_dict = {metric: {} for metric in self.episodic_metrics}
            for search_name in raw_current_metrics.keys():
                for metric in raw_current_metrics[search_name].keys():
                    current_val = raw_current_metrics[search_name].get(metric)
                    baseline_val = raw_baseline_metrics[search_name].get(metric)
                    if current_val is not None and baseline_val is not None:
                        current_dict[metric][search_name] = current_val
                        baseline_dict[metric][search_name] = baseline_val
                        gap_dict[metric][search_name] = current_val - baseline_val
            for metric in self.episodic_metrics:
                self._log_metrics(f'{metric}_rules_metrics', current_dict[metric], exclude_from_csv=True)
                self._log_metrics(f'baseline_{metric}_rules_metrics', baseline_dict[metric], exclude_from_csv=True)
                self._log_metrics(f'{metric}_gap_rules_metrics', gap_dict[metric], exclude_from_csv=True)


            # log episodic policy (injected logs)
            if 'episode_logs' in info:
                self._log_metrics('episodic_policy', info['episode_logs'], exclude_from_csv=True)
            # # Log alert rewards
            # rules_alert_reward = info.get('rules_alert_reward', {})
            # if rules_alert_reward:
            #     self._log_metrics('rules_alert_rewards', rules_alert_reward, exclude_from_csv=True)
    
    def _on_training_end(self) -> None:
        env = self.training_env.envs[0]
        start_time_datetime = datetime.datetime.strptime(env.fake_start_datetime, '%m/%d/%Y:%H:%M:%S')#datetime.datetime.strptime(kwargs['fake_start_datetime'], '%m/%d/%Y:%H:%M:%S')
        end_time_datetime = datetime.datetime.strptime(env.time_range[1], '%m/%d/%Y:%H:%M:%S')
        clean_env(env.splunk_tools_instance, (start_time_datetime.timestamp(), end_time_datetime.timestamp()))
    


class HParamsCallback(BaseCallback):
    def __init__(self, verbose=1, experiment_kwargs=None, phase="train"):
        super().__init__(verbose)
        self.experiment_kwargs = experiment_kwargs
        self.phase = phase
    def _on_step(self) -> bool:
        return True
    def _on_training_start(self) -> None:
        """Setup hyperparameter logging"""
        metric_dict = {
            f"{self.phase}/{tag}": 0 for tag in [
                "distribution_reward", "energy_reward", "alert_reward",
                "duration", "cpu", "total_cpu_usage"
            ]
        }
        
        # Add rollout metrics
        metric_dict.update({
            f"rollout/{tag}": 0 for tag in [
                "ep_rew_mean", "ep_rew_std", "ep_len_mean", "ep_len_std"
            ]
        })
        
        hparams = HParam(self.experiment_kwargs, metric_dict)
        self.logger.record("hparams", hparams, exclude=("stdout", "log", "json", "csv"))
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, eval_log_dir=None, phase="eval", **kwargs):
        super().__init__(eval_env, **kwargs)
        self.phase = phase
        self.episodic_metrics = [
            "alert", "duration", "cpu", "std_cpu", "std_duration",
            "read_bytes", "read_count", "write_bytes", "write_count",
            "total_cpu_usage"
        ]

        # Set up separate TensorBoard logger for evaluation
        if eval_log_dir:
            self.eval_logger = configure(eval_log_dir, ["tensorboard"])
        else:
            self.eval_logger = None  # Use default logger


    def _log_metrics(self, key: str, value):
        if value is not None:
            if self.eval_logger:
                self.eval_logger.record(f"{self.phase}/{key}", value)
            else:
                self.logger.record(f"{self.phase}/{key}", value)  # Fallback to default logger

    def _on_step(self) -> bool:
        """Log per-step and episodic evaluation metrics"""
        result = super()._on_step()  # Run standard EvalCallback logic
        eval_infos = self.locals['infos'][0] 
        self._log_metrics('distribution_value', eval_infos.get('distribution_value'))
        # Log reward components
        for reward_type in ['distribution_reward', 'energy_reward', 'alert_reward']:
            if reward_type in eval_infos:
                self._log_metrics(reward_type, eval_infos[reward_type])
        if eval_infos.get('done', False):
            step = self.num_timesteps


            # Log episodic metrics at episode end
            current_metrics = eval_infos.get('combined_metrics', {})
            baseline_metrics = eval_infos.get('combined_baseline_metrics', {})

            # Compute total reward
            env = self.eval_env
            total_reward = (
                env.envs[0].alpha * eval_infos.get("energy_reward", 0)
                + env.envs[0].beta * eval_infos.get("alert_reward", 0)
                + env.envs[0].gamma * eval_infos.get("distribution_reward", 0)
            )
            self._log_metrics("total_reward", total_reward)

            # Log episodic metrics
            for metric in self.episodic_metrics:
                current_val = current_metrics.get(metric, None)
                baseline_val = baseline_metrics.get(metric, None)

                if current_val is not None and baseline_val is not None:
                    self._log_metrics(f"{metric}", current_val)
                    self._log_metrics(f"baseline_{metric}", baseline_val)
                    self._log_metrics(f"{metric}_gap", current_val - baseline_val)

            raw_current_metrics = eval_infos.get('raw_metrics', {})
            raw_baseline_metrics = eval_infos.get('raw_baseline_metrics', {})
            baseline_dict = {metric: {} for metric in self.episodic_metrics}
            current_dict = {metric: {} for metric in self.episodic_metrics}
            gap_dict = {metric: {} for metric in self.episodic_metrics}
            for search_name in raw_current_metrics.keys():
                for metric in raw_current_metrics[search_name].keys():
                    current_val = raw_current_metrics[search_name].get(metric)
                    baseline_val = raw_baseline_metrics[search_name].get(metric)
                    if current_val is not None and baseline_val is not None:
                        current_dict[metric][search_name] = current_val
                        baseline_dict[metric][search_name] = baseline_val
                        gap_dict[metric][search_name] = current_val - baseline_val
            for metric in self.episodic_metrics:
                self._log_metrics(f'{metric}_rules_metrics', current_dict[metric])
                self._log_metrics(f'baseline_{metric}_rules_metrics', baseline_dict[metric])
                self._log_metrics(f'{metric}_gap_rules_metrics', gap_dict[metric])

            
            # Log injected logs (policy actions)
            if "episode_logs" in eval_infos:
                self._log_metrics("episodic_policy", eval_infos["episode_logs"])
        

            self.eval_logger.dump(step)

        return result
