
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

        # Dedicated writer for custom env metrics (energy_reward, alert_reward, etc.).
        # By writing directly via add_scalar() at local step (n_calls) we avoid
        # touching the SB3 logger buffer, which means SB3 can dump fps / loss /
        # ep_rew_mean at num_timesteps without any interference from our dumps.
        self.custom_writer = SummaryWriter(
            log_dir=os.path.join(log_dir, f"{phase}_metrics")
        ) if log_dir else None

    def _log_metrics(self, key: str, value, step: int = 0, exclude_from_csv=True):
        """Write a scalar directly to the custom_writer at the given local step.
        Non-numeric values (e.g. TimeWindow string repr) are silently skipped —
        add_scalar() only accepts floats/ints."""
        if value is not None and self.custom_writer is not None and isinstance(value, (int, float)):
            self.custom_writer.add_scalar(f"{self.phase}/{key}", value, global_step=step)

    def log_step_metrics(self, info: Dict, step: int = 0):
        """Log metrics available at each step"""
        action_window = info.get('action_window')
        if action_window:
            self._log_metrics('action_window', f"{action_window}", step=step)

        for reward_type in ['distribution_reward']:
            if reward_type in info:
                self._log_metrics(reward_type, info[reward_type], step=step)

        self._log_metrics('distribution_value', info.get('distribution_value'), step=step)

    def log_episode_metrics(self, info: Dict, env, step: int = 0):
        """Log metrics at episode end"""
        current_window = info.get('current_window')
        if current_window:
            self._log_metrics('current_window', f"{current_window}", step=step)

        current_metrics = info.get('combined_metrics', {})
        baseline_metrics = info.get('combined_baseline_metrics', {})
        self._log_metrics('final_distribution_value', info.get('distribution_value'), step=step)
        self._log_metrics('ac_distribution_value', info.get('ac_distribution_value'), step=step)
        self._log_metrics('full_ac_distribution_value', info.get('full_ac_distribution_value'), step=step)
        self._log_metrics('ac_distribution_reward', info.get('ac_distribution_reward'), step=step)
        self._log_metrics('energy_reward', info.get('energy_reward'), step=step)
        self._log_metrics('norm_energy_reward', info.get('norm_energy_reward'), step=step)
        self._log_metrics('alert_reward', info.get('alert_reward'), step=step)
        self._log_metrics('norm_alert_reward', info.get('norm_alert_reward'), step=step)
        self._log_metrics('constrained_reward', info.get('constrained_reward'), step=step)
        self._log_metrics('constrained_energy_term', info.get('constrained_energy_term'), step=step)
        self._log_metrics('constrained_alert_metric', info.get('constrained_alert_metric'), step=step)
        self._log_metrics('constrained_distribution_metric', info.get('constrained_distribution_metric'), step=step)
        self._log_metrics('constrained_quota_metric', info.get('constrained_quota_metric'), step=step)
        self._log_metrics('constrained_alert_penalty', info.get('constrained_alert_penalty'), step=step)
        self._log_metrics('constrained_distribution_penalty', info.get('constrained_distribution_penalty'), step=step)
        self._log_metrics('constrained_quota_penalty', info.get('constrained_quota_penalty'), step=step)
        self._log_metrics('lambda_alert', info.get('lambda_alert'), step=step)
        self._log_metrics('lambda_distribution', info.get('lambda_distribution'), step=step)
        self._log_metrics('lambda_quota', info.get('lambda_quota'), step=step)
        self._log_metrics('tau_alert_effective', info.get('tau_alert_effective'), step=step)
        self._log_metrics('tau_kl_effective', info.get('tau_kl_effective'), step=step)
        self._log_metrics('tau_quota_effective', info.get('tau_quota_effective'), step=step)
        self._log_metrics('total_episode_logs', info.get('total_episode_logs'), step=step)

        if current_metrics and baseline_metrics:
            for metric in self.episodic_metrics:
                current_val = current_metrics.get(metric)
                baseline_val = baseline_metrics.get(metric)
                if current_val is not None and baseline_val is not None:
                    self._log_metrics(f'{metric}', current_val, step=step)
                    self._log_metrics(f'baseline_{metric}', baseline_val, step=step)
                    self._log_metrics(f'{metric}_gap', current_val - baseline_val, step=step)
            if 'real_cpu' in current_metrics:
                self._log_metrics('measured_cpu', current_metrics.get('real_cpu', 0), step=step)
                real_cpu = current_metrics.get('real_cpu', 0)
                if real_cpu != 0:
                    self._log_metrics('cpu_error', (current_metrics.get('cpu', 0) - real_cpu) / real_cpu, step=step)

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
                        self.writers[search_name].add_scalar(f'{self.phase}/rules_baseline_{metric}', baseline_val, global_step=step)
                        self.writers[search_name].add_scalar(f'{self.phase}/rules_{metric}', current_val, global_step=step)
                        self.writers[search_name].add_scalar(f'{self.phase}/rules_{metric}_gap', current_val - baseline_val, global_step=step)

        for event_type in info.get('episode_logs', {}):
            self.writers[f"{event_type}"].add_scalar(f'{self.phase}/episodic_policy', info['episode_logs'][f"{event_type}"], global_step=step)
            self.writers[f"{event_type}"].add_scalar(f'{self.phase}/diversity_policy', info['diversity_episode_logs'].get(f"{event_type}", 0), global_step=step)

        if 'episodic_inserted_logs' in info:
            self._log_metrics('episodic_inserted_logs', info['episodic_inserted_logs'], step=step)

        if 'episodic_inserted_logs' in info and 'episode_logs' in info:
            self._log_metrics('actual_quota', info['episodic_inserted_logs'] / (info['total_episode_logs'] + 1e-8), step=step)

        for event_type in self.event_types:
            self.writers[event_type].add_scalar(f'{self.phase}/real_relevant_distribution', info['real_relevant_distribution'].get(event_type, 0), global_step=step)
            self.writers[event_type].add_scalar(f'{self.phase}/fake_relevant_distribution', info['fake_relevant_distribution'].get(event_type, 0), global_step=step)

        # Flush all writers (rules, event_types, custom)
        for writer in self.writers.values():
            writer.flush()
        if self.custom_writer is not None:
            self.custom_writer.flush()

    def _close_custom_writer(self):
        if self.custom_writer is not None:
            try:
                self.custom_writer.close()
            except Exception as e:
                logger.warning(f"Error closing custom_writer: {e}")
            self.custom_writer = None


class CustomTensorboardCallback(MetricsLoggerCallback, BaseCallback):
    def __init__(self, log_dir, rules, event_types, verbose=1, writers=None):
        BaseCallback.__init__(self, verbose)
        MetricsLoggerCallback.__init__(self, "train", log_dir, rules, event_types, writers=writers)

    def _on_step(self) -> bool:
        """Log metrics at each step"""
        info = self.locals['infos'][0]
        info['n_calls'] = self.n_calls
        # Step-level env metrics at local step (n_calls)
        self.log_step_metrics(info, step=self.n_calls)

        ep_step = info.get('step', 0)
        global_step = info.get('all_steps_counter', 0)
        logger.info(f"[Step] episode_step={ep_step}, global_step={global_step}, n_calls={self.n_calls}")

        # Episode-level env metrics written directly to custom_writer at local step.
        # We do NOT call self.logger.dump() here so that SB3's internal metrics
        # (fps, loss, ep_rew_mean) are written exclusively by SB3's own dump at
        # num_timesteps, with no interference from our buffer usage.
        done = self.locals.get('dones', [False])[0] or info.get('done', False)
        if done:
            self.log_episode_metrics(info, self.training_env, step=self.n_calls)

        return True

    def _on_training_end(self) -> None:
        self._close_custom_writer()


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


class CustomEvalCallback3(MetricsLoggerCallback, EvalCallback):
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
            self.all_actions_list.append(locals_['actions'][0])

        def _log_info_callback(locals_, globals_):
            info = locals_['info']
            info['n_calls'] = self.n_calls
            self.eval_infos.append(info)

        def _log_info_store_actions_callback(locals_, globals_):
            _log_info_callback(locals_, globals_)
            _store_actions_callback(locals_, globals_)

        kwargs['callback'] = _log_info_store_actions_callback
        return evaluate_policy(*args, **kwargs)

    def _aggregate_eval_metrics(self, infos, step: int = 0):
        """Aggregate metrics from multiple evaluation episodes"""
        last_info = infos[-1] if infos else None
        if not last_info:
            return

        mean_metrics = defaultdict(list)
        for info in infos:
            for metric in ['distribution_reward', 'energy_reward', 'alert_reward',
                         'distribution_value', 'inserted_logs', 'total_current_logs',
                         'constrained_reward', 'constrained_energy_term',
                         'constrained_alert_metric', 'constrained_distribution_metric',
                         'constrained_quota_metric']:
                if metric in info:
                    mean_metrics[metric].append(info[metric])

        for metric, values in mean_metrics.items():
            try:
                self._log_metrics(f'mean_{metric}', np.mean(values), step=step)
            except Exception as e:
                logger.warning(f"Error logging metric {metric}: {e}")

        self.log_episode_metrics(last_info, self.eval_env, step=step)

    def _on_step(self) -> bool:
        """Evaluate the agent and log metrics"""
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_infos = []

            episode_rewards, episode_lengths = self.evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn
            )
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
                            if self.results_dir:
                                csv_path = os.path.join(self.results_dir, f"full_eval_results_random_{self.is_random_agent}.csv")
                            else:
                                experiments_dir = self.log_dir.split("tensorboard")[0]
                                csv_path = f"{experiments_dir}/full_eval_results_random_{self.is_random_agent}.csv"
                            pd.DataFrame(results_for_csv).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

                        obs = self.full_eval_env.reset()
                self.all_actions_list = []

            # All eval metrics written to custom_writer at local step (n_calls).
            # No self.logger.dump() — SB3's own dump handles agent metrics.
            self._aggregate_eval_metrics(self.eval_infos, step=self.n_calls)
            self._log_metrics('mean_reward', np.mean(episode_rewards), step=self.n_calls)
            self._log_metrics('mean_ep_length', np.mean(episode_lengths), step=self.n_calls)

        return True

    def _on_training_end(self) -> None:
        self._close_custom_writer()


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
                return False
            else:
                logger.info(f"Splunk license usage is sufficient: {remaining_mb} MB remaining out of {quota_mb} MB")
                self.logger.record('splunk/license_usage', remaining_mb)
                self.logger.record('splunk/quota_mb', quota_mb)
                self.logger.dump(self.n_calls)
        return True
