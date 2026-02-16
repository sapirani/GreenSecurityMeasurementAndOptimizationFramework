
import os
import logging
import datetime
import random
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from typing import Dict, Any
import numpy as np
from collections import defaultdict
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

logger = logging.getLogger(__name__)


class MetricsLoggerCallback:
    """Base class containing shared logging logic"""

    def __init__(self, phase="train", log_dir=None, rules=None, event_types=None, writers=None):
        self.phase = phase
        self.episodic_metrics = [
            "alert", "duration", "cpu",
            "read_bytes", "read_count", "write_bytes", "write_count", 'memory_mb']
        self.log_dir = log_dir
        self.rules = rules
        self.event_types = event_types

        self.writers = writers if writers else {}

    def _log_metrics(self, key: str, value, exclude_from_csv=True):
        """Safely log metrics to tensorboard"""
        if value is not None:
            self.logger.record(f"{self.phase}/{key}", value)

    def log_step_metrics(self, info: Dict):
        """Log metrics available at each step"""
        # Log basic step information
        action_window = info.get('action_window')
        if action_window:
            self._log_metrics('action_window', f"{action_window}")

        # Log reward components
        for reward_type in ['distribution_reward']:
            if reward_type in info:
                self._log_metrics(reward_type, info[reward_type])

        self._log_metrics('distribution_value', info.get('distribution_value'))

    def log_episode_metrics(self, info: Dict, env):
        """Log metrics at episode end"""
        # Log current window
        current_window = info.get('current_window')
        if current_window:
            self._log_metrics('current_window', f"{current_window}", exclude_from_csv=True)

        # Get the actual environment from the vectorized env wrapper
        actual_env = env.envs[0] if hasattr(env, 'envs') else env

        # Log metrics for current and baseline
        current_metrics = info.get('combined_metrics', {})
        baseline_metrics = info.get('combined_baseline_metrics', {})
        self._log_metrics('final_distribution_value', info.get('distribution_value'))
        self._log_metrics('ac_distribution_value', info.get('ac_distribution_value'))
        self._log_metrics('full_ac_distribution_value', info.get('full_ac_distribution_value'))
        self._log_metrics('ac_distribution_reward', info.get('ac_distribution_reward'))
        self._log_metrics('energy_reward', info.get('energy_reward'))
        self._log_metrics('norm_energy_reward', info.get('norm_energy_reward'))
        self._log_metrics('alert_reward', info.get('alert_reward'))
        self._log_metrics('norm_alert_reward', info.get('norm_alert_reward'))
        self._log_metrics('constrained_reward', info.get('constrained_reward'))
        self._log_metrics('constrained_energy_term', info.get('constrained_energy_term'))
        self._log_metrics('constrained_alert_metric', info.get('constrained_alert_metric'))
        self._log_metrics('constrained_distribution_metric', info.get('constrained_distribution_metric'))
        self._log_metrics('constrained_quota_metric', info.get('constrained_quota_metric'))
        self._log_metrics('constrained_alert_penalty', info.get('constrained_alert_penalty'))
        self._log_metrics('constrained_distribution_penalty', info.get('constrained_distribution_penalty'))
        self._log_metrics('constrained_quota_penalty', info.get('constrained_quota_penalty'))
        self._log_metrics('lambda_alert', info.get('lambda_alert'))
        self._log_metrics('lambda_distribution', info.get('lambda_distribution'))
        self._log_metrics('lambda_quota', info.get('lambda_quota'))

        self._log_metrics('total_episode_logs', info.get('total_episode_logs'))

        if current_metrics and baseline_metrics:
            # Log basic metrics
            for metric in self.episodic_metrics:
                current_val = current_metrics.get(metric)
                baseline_val = baseline_metrics.get(metric)
                if current_val is not None and baseline_val is not None:

                    self._log_metrics(f'{metric}', current_val)
                    self._log_metrics(f'baseline_{metric}', baseline_val)
                    self._log_metrics(f'{metric}_gap', current_val - baseline_val)
            if 'real_cpu' in current_metrics:
                    self._log_metrics('measured_cpu', current_metrics.get('real_cpu', 0))
                    self._log_metrics('cpu_error', (current_metrics.get('cpu', 0) - current_metrics.get('real_cpu', 0))/current_metrics.get('real_cpu', 0))
            # Log detailed rules metrics
            raw_current_metrics = info.get('raw_metrics', {})
            raw_baseline_metrics = info.get('raw_baseline_metrics', {})

            for search_name in self.rules:
                for metric in self.episodic_metrics:
                    raw_current_metrics_search = raw_current_metrics.get(search_name, None)
                    raw_baseline_metrics_search = raw_baseline_metrics.get(search_name, None)
                    if raw_current_metrics_search is None or raw_baseline_metrics_search is None:
                        continue
                    current_val = raw_current_metrics_search.get(metric, None)
                    baseline_val = raw_baseline_metrics_search.get(metric, None)
                    if current_val is not None and baseline_val is not None:
                        self.writers[search_name].add_scalar(f'{self.phase}/rules_baseline_{metric}', baseline_val, global_step=info['n_calls'])
                        self.writers[search_name].add_scalar(f'{self.phase}/rules_{metric}', current_val, global_step=info['n_calls'])
                        self.writers[search_name].add_scalar(f'{self.phase}/rules_{metric}_gap', current_val - baseline_val, global_step=info['n_calls'])

        # Log episodic policy
        for event_type in info.get('episode_logs', {}):
            self.writers[f"{event_type}"].add_scalar(f'{self.phase}/episodic_policy', info['episode_logs'][f"{event_type}"], global_step=info['n_calls'])
            self.writers[f"{event_type}"].add_scalar(f'{self.phase}/diversity_policy', info['diversity_episode_logs'].get(f"{event_type}", 0), global_step=info['n_calls'])

        if 'episodic_inserted_logs' in info:
            self._log_metrics('episodic_inserted_logs', info['episodic_inserted_logs'], exclude_from_csv=True)

        if 'episodic_inserted_logs' in info and 'episode_logs' in info:
            self._log_metrics('actual_quota', info['episodic_inserted_logs']/(info['total_episode_logs']+1e-8), exclude_from_csv=True)

        for event_type in self.event_types:
            self.writers[event_type].add_scalar(f'{self.phase}/real_relevant_distribution', info['real_relevant_distribution'].get(event_type, 0), global_step=info['n_calls'])
            self.writers[event_type].add_scalar(f'{self.phase}/fake_relevant_distribution', info['fake_relevant_distribution'].get(event_type, 0), global_step=info['n_calls'])

        for writer in self.writers.values():
            writer.flush()


class CustomTensorboardCallback(MetricsLoggerCallback, BaseCallback):
    def __init__(self, log_dir, rules, event_types, verbose=1, writers=None):
        BaseCallback.__init__(self, verbose)
        MetricsLoggerCallback.__init__(self, "train", log_dir, rules, event_types, writers=writers)


    def _on_step(self) -> bool:
        """Log metrics at each step"""
        info = self.locals['infos'][0]  # Get info from environment step
        info['n_calls'] = self.n_calls  # Add current step count to info
        self.log_step_metrics(info)

        ep_step = info.get('step', 0)
        global_step = info.get('all_steps_counter', 0)
        logger.info(f"[Step] episode_step={ep_step}, global_step={global_step}, n_calls={self.n_calls}")

        # Log episode end metrics
        if info.get('done', False):
            self.log_episode_metrics(info, self.training_env)

        self.logger.dump(self.n_calls)
        return True


class HParamsCallback(BaseCallback):
    """Log hyperparameters to TensorBoard at training start."""

    def __init__(self, hparam_dict: dict, log_dir: str, verbose=0):
        super().__init__(verbose)
        self.hparam_dict = hparam_dict
        self.log_dir = log_dir
        self._logged = False

    def _on_step(self) -> bool:
        if not self._logged:
            writer = SummaryWriter(log_dir=self.log_dir)
            # Flatten and sanitize hparams for TensorBoard
            flat_hparams = {}
            for k, v in self.hparam_dict.items():
                if isinstance(v, (int, float, str, bool)):
                    flat_hparams[k] = v
                else:
                    flat_hparams[k] = str(v)
            writer.add_hparams(flat_hparams, {'dummy': 0})
            writer.close()
            self._logged = True
        return True


class CustomEvalCallback3( MetricsLoggerCallback, EvalCallback):
    def __init__(self,
                 eval_env,
                 log_dir, rules, event_types,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 3000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1,
                 writers=None,
                 full_eval_env=None,
                 additional_percentage: float = 1.0,
                 hosts_num: int = 100,
                 is_random_agent: bool = False,
                 results_dir: str = None):

        EvalCallback.__init__(
            self,
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose
        )
        MetricsLoggerCallback.__init__(self, "eval", log_dir, rules, event_types, writers=writers)
        self.full_eval_env = full_eval_env
        self.additional_percentage = additional_percentage
        self.hosts_number = hosts_num
        self.is_random_agent = is_random_agent
        self.results_dir = results_dir
        self.all_actions_list = []

    def evaluate_policy(self, *args, **kwargs):
        """Override evaluate_policy to collect info during evaluation"""
        self.eval_infos = []

        def _store_actions_callback(locals_, globals_):
            """
            Callback function to store the action at each step.

            :param locals_: A dictionary containing local variables.
            :param globals_: A dictionary containing global variables.
            """
            # 'actions' is the variable name used inside evaluate_policy
            # It will be a numpy array, e.g., array([1])
            self.all_actions_list.append(locals_['actions'][0])

        def _log_info_callback(locals_, globals_):
            info = locals_['info']
            info['n_calls'] = self.n_calls  # Add current step count to info
            self.eval_infos.append(info)

        def _log_info_store_actions_callback(locals_, globals_):
            _log_info_callback(locals_, globals_)
            _store_actions_callback(locals_, globals_)

        kwargs['callback'] = _log_info_store_actions_callback
        return evaluate_policy(*args, **kwargs)

    def _aggregate_eval_metrics(self, infos):
        """Aggregate metrics from multiple evaluation episodes"""
        # Get the last info which should contain final episode metrics
        last_info = infos[-1] if infos else None
        if not last_info:
            return

        # Calculate mean of step-wise metrics across all infos
        mean_metrics = defaultdict(list)
        for info in infos:
            for metric in ['distribution_reward', 'energy_reward', 'alert_reward',
                         'distribution_value', 'inserted_logs', 'total_current_logs',
                         'constrained_reward', 'constrained_energy_term',
                         'constrained_alert_metric', 'constrained_distribution_metric',
                         'constrained_quota_metric']:
                if metric in info:
                    mean_metrics[metric].append(info[metric])

        # Log mean metrics
        for metric, values in mean_metrics.items():
            try:
                self._log_metrics(f'mean_{metric}', np.mean(values))
            except Exception as e:
                logger.warning(f"Error logging metric {metric}: {e}")

        # Log the final episode metrics using the base class method
        self.log_episode_metrics(last_info, self.eval_env)

    def _on_step(self) -> bool:
        """Evaluate the agent and log metrics"""
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Reset eval info collection
            self.eval_infos = []

            # Run evaluation
            episode_rewards, episode_lengths = self.evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn
            )
            # use stored actions to evaluate full eval env if provided
            if self.full_eval_env is not None:
                obs = self.full_eval_env.reset()
                for action in self.all_actions_list:
                    obs, reward, terminated, truncated, info = self.full_eval_env.step(action)
                    done = terminated or truncated
                    if done:
                        if 'ac_distribution_value' in info:
                            last_real_eval_info = self.eval_infos[-1]
                            logger.debug(f"Last eval info: {last_real_eval_info}")
                            last_real_eval_info["full_ac_distribution_value"] = info['ac_distribution_value']
                            # add all metrics to results_for_csv
                            # Extract time range from eval info
                            current_window = last_real_eval_info.get('current_window', (None, None))
                            results_for_csv = [{
                                "eval_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "time_range_start": current_window[0] if current_window else None,
                                "time_range_end": current_window[1] if current_window else None,
                                "additional_percentage": self.additional_percentage,
                                "hosts_number": self.hosts_number,
                                "full_ac_distribution_value": last_real_eval_info["full_ac_distribution_value"],
                                "cpu": last_real_eval_info['combined_metrics'].get('cpu', None),
                                "baseline_cpu": last_real_eval_info['combined_baseline_metrics'].get('cpu', None),
                                "duration": last_real_eval_info['combined_metrics'].get('duration', None),
                                "baseline_duration": last_real_eval_info['combined_baseline_metrics'].get('duration', None),
                                "read_bytes": last_real_eval_info['combined_metrics'].get('read_bytes', None),
                                "baseline_read_bytes": last_real_eval_info['combined_baseline_metrics'].get('read_bytes', None),
                                "write_bytes": last_real_eval_info['combined_metrics'].get('write_bytes', None),
                                "baseline_write_bytes": last_real_eval_info['combined_baseline_metrics'].get('write_bytes', None),
                                "alert": last_real_eval_info['combined_metrics'].get('alert', None),
                                "baseline_alert": last_real_eval_info['combined_baseline_metrics'].get('alert', None),
                                "memory_mb": last_real_eval_info['combined_metrics'].get('memory_mb', None),
                                "baseline_memory_mb": last_real_eval_info['combined_baseline_metrics'].get('memory_mb', None),
                            }]
                            # dump to csv in results directory
                            if self.results_dir:
                                csv_path = os.path.join(self.results_dir, f"full_eval_results_random_{self.is_random_agent}.csv")
                            else:
                                experiments_dir = self.log_dir.split("tensorboard")[0]
                                csv_path = f"{experiments_dir}/full_eval_results_random_{self.is_random_agent}.csv"
                            pd.DataFrame(results_for_csv).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

                        obs = self.full_eval_env.reset()
                self.all_actions_list = []

            # Log aggregated evaluation metrics
            self._aggregate_eval_metrics(self.eval_infos)

            # Log standard evaluation metrics
            self._log_metrics('mean_reward', np.mean(episode_rewards))
            self._log_metrics('mean_ep_length', np.mean(episode_lengths))

            self.logger.dump(self.n_calls)

        return True


class SplunkLincenceCheckCallback(BaseCallback):
    def __init__(self):
        super(SplunkLincenceCheckCallback, self).__init__()
        self.check_interval = 1000

    def _on_step(self) -> bool:
        """Check Splunk license usage at each step"""
        if self.n_calls % self.check_interval == 0:
            env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
            res = env.splunk_tools.check_license_usage()
            remaining_mb = res['remaining_mb']
            quota_mb = res['quota_mb']
            if remaining_mb < 1000:
                logger.warning(f"Splunk license usage is low: {remaining_mb} MB remaining out of {quota_mb} MB")
                self.logger.record('splunk/remaining_mb', remaining_mb)
                self.logger.record('splunk/quota_mb', quota_mb)
                self.logger.dump(self.n_calls)
                # stop training
                return False
            else:
                logger.info(f"Splunk license usage is sufficient: {remaining_mb} MB remaining out of {quota_mb} MB")
                self.logger.record('splunk/license_usage', remaining_mb)
                self.logger.record('splunk/quota_mb', quota_mb)
                self.logger.dump(self.n_calls)
        return True
