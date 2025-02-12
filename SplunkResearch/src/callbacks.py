from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import numpy as np
import logging
from stable_baselines3.common.logger import HParam

logger = logging.getLogger(__name__)

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1, phase="train", experiment_kwargs=None):
        super().__init__(verbose)
        self.phase = phase
        self.experiment_kwargs = experiment_kwargs
        self.episodic_metrics = [
            "alert", "duration", "std_duration", "cpu", "std_cpu",
            "read_bytes", "read_count", "write_bytes", "write_count",
            "total_cpu_usage"
        ]
        self.started_measurements = False

    def _log_metrics(self, key: str, value, exclude_from_csv=False):
        """Safely log metrics to tensorboard"""
        if value is not None:
            self.logger.record(f"{self.phase}/{key}", value, exclude=("csv",) if exclude_from_csv else None)

    def _on_step(self) -> bool:
        """Log metrics at each step"""
        info = self.locals['infos'][0]  # Get info from environment step
        
        # Log basic step information
        step_counter = info.get('step', 0)
        all_steps = info.get('all_steps_counter', 0)
        current_time = info.get('current_time')
        if current_time:
            self._log_metrics('current_time', current_time, exclude_from_csv=True)

        # Log reward components
        for reward_type in ['distribution_reward', 'energy_reward', 'alert_reward']:
            if reward_type in info:
                self._log_metrics(reward_type, info[reward_type])

        # Log distributions
        real_dist = info.get('real_distribution')
        fake_dist = info.get('fake_distribution')
        if real_dist is not None and fake_dist is not None:
            self._log_metrics('distribution_distance', info.get('distribution_distance'))

        # Log quota information
        if 'remaining_quota' in info:
            self._log_metrics('quota_remaining', info['remaining_quota'])

        # Log episode end metrics
        if info.get('done', False):
            self._log_episode_metrics(info) 
        self.logger.dump(all_steps)
        return True

    def _log_episode_metrics(self, info):
        """Log metrics at episode end"""
        # Log rules execution metrics
        current_metrics = info.get('current_metrics', {})
        baseline_metrics = info.get('baseline_metrics', {})
        
        if current_metrics and baseline_metrics:
            # Log basic metrics
            for metric in self.episodic_metrics:
                current_val = current_metrics.get(metric)
                baseline_val = baseline_metrics.get(metric)
                if current_val is not None and baseline_val is not None:
                    self._log_metrics(f'{metric}', current_val)
                    self._log_metrics(f'baseline_{metric}', baseline_val)
                    self._log_metrics(f'{metric}_gap', current_val - baseline_val)

            # Log rule-specific metrics
            rules_dict = {}
            for rule in info.get('active_rules', []):
                rule_name = rule['title']
                for metric in self.episodic_metrics:
                    key = f'rule_{metric}_{rule_name}'
                    if key in current_metrics:
                        rules_dict[key] = current_metrics[key]
            if rules_dict:
                self._log_metrics('rules_metrics', rules_dict, exclude_from_csv=True)

            # # Log alert rewards
            # rules_alert_reward = info.get('rules_alert_reward', {})
            # if rules_alert_reward:
            #     self._log_metrics('rules_alert_rewards', rules_alert_reward, exclude_from_csv=True)

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

class SaveArtifactsCallback(BaseCallback):
    def __init__(self, experiment_manager, save_freq=200):
        super().__init__(verbose=1)
        self.save_freq = save_freq
        self.experiment_manager = experiment_manager

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            env = self.training_env.envs[0]
            # self.experiment_manager.post_experiment(env, self.model)
            
            # Save additional metrics if available
            info = self.locals['infos'][0]
            if 'metrics_to_save' in info:
                metrics_df = pd.DataFrame(info['metrics_to_save'])
                metrics_df.to_csv(f"{self.experiment_manager.metrics_path}.csv", index=False)
                
        return True

def create_callbacks(experiment_manager, config):
    """Create all necessary callbacks"""
    return [
        TensorboardCallback(
            phase=config.get('phase', 'train'),
            experiment_kwargs=config
        ),
        # HParamsCallback(
        #     experiment_kwargs=config,
        #     phase=config.get('phase', 'train')
        # ),
        SaveArtifactsCallback(
            experiment_manager=experiment_manager
        )
    ]
