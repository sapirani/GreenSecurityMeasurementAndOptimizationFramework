
import random
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from typing import Dict, Any
import numpy as np
from collections import defaultdict
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter

class MetricsLoggerCallback:
    """Base class containing shared logging logic"""
    
    def __init__(self, phase="train", log_dir=None, rules=None, event_types=None, writers=None):
        self.phase = phase
        self.episodic_metrics = [
            "alert", "duration", "cpu"]
        #     , "std_cpu", "std_duration", 
        #     "read_bytes", "read_count", "write_bytes", "write_count",
        #     "total_cpu_usage"
        # ]
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
        for reward_type in ['distribution_reward', 'energy_reward', 'alert_reward']:
            if reward_type in info:
                self._log_metrics(reward_type, info[reward_type])
        
        self._log_metrics('distribution_value', info.get('distribution_value'))

        # # Log quota information
        # if 'inserted_logs' in info:
        #     self._log_metrics('inserted_logs', info['inserted_logs'])
            
        # if 'total_current_logs' in info:
        #     self._log_metrics('total_current_logs', info['total_current_logs'])
        
        # if 'inserted_logs' in info and 'total_current_logs' in info:
        #     self._log_metrics('actual_quota', info['inserted_logs']/info['total_current_logs'])

    def log_episode_metrics(self, info: Dict, env):
        """Log metrics at episode end"""
        # Log current window
        current_window = info.get('current_window')
        if current_window:
            self._log_metrics('current_window', f"{current_window}", exclude_from_csv=True)
            
        # Get the actual environment from the vectorized env wrapper
        actual_env = env.envs[0] if hasattr(env, 'envs') else env
            
        # # Compute and log total reward
        # total_reward = (getattr(actual_env, 'alpha', 1.0) * info.get('energy_reward', 0) + 
        #                getattr(actual_env, 'beta', 1.0) * info.get('alert_reward', 0) + 
        #                getattr(actual_env, 'gamma', 1.0) * info.get('distribution_reward', 0))
        # self._log_metrics('total_reward', total_reward)
        
        # Log metrics for current and baseline
        current_metrics = info.get('combined_metrics', {})
        baseline_metrics = info.get('combined_baseline_metrics', {})
        self._log_metrics('final_distribution_value', info.get('distribution_value'))
        self._log_metrics('ac_distribution_value', info.get('ac_distribution_value'))
        self._log_metrics('ac_distribution_reward', info.get('ac_distribution_reward'))
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

            # Log detailed rules metrics
            raw_current_metrics = info.get('raw_metrics', {})
            raw_baseline_metrics = info.get('raw_baseline_metrics', {})
            
            # baseline_dict = {metric: {} for metric in self.episodic_metrics}
            # current_dict = {metric: {} for metric in self.episodic_metrics}
            # gap_dict = {metric: {} for metric in self.episodic_metrics}
            
            for search_name in self.rules:
                for metric in self.episodic_metrics:
                    raw_current_metrics_search = raw_current_metrics.get(search_name, None)
                    raw_baseline_metrics_search = raw_baseline_metrics.get(search_name, None)
                    if raw_current_metrics_search is None or raw_baseline_metrics_search is None:
                        continue
                    current_val = raw_current_metrics_search.get(metric, None)
                    baseline_val = raw_baseline_metrics_search.get(metric, None)
                    if current_val is not None and baseline_val is not None:
                        self.writers[search_name].add_scalar(f'{self.phase}/rules_{metric}', current_val, global_step=info['n_calls'])
                        self.writers[search_name].add_scalar(f'{self.phase}/rules_baseline_{metric}', baseline_val, global_step=info['n_calls'])
                        self.writers[search_name].add_scalar(f'{self.phase}/rules_{metric}_gap', current_val - baseline_val, global_step=info['n_calls'])
                        # current_dict[metric][search_name] = current_val
                        # baseline_dict[metric][search_name] = baseline_val
                        # gap_dict[metric][search_name] = current_val - baseline_val
                        
            # for metric in self.episodic_metrics:
            #     self._log_metrics(f'{metric}_rules_metrics', current_dict[metric], exclude_from_csv=True)
            #     self._log_metrics(f'baseline_{metric}_rules_metrics', baseline_dict[metric], exclude_from_csv=True)
            #     self._log_metrics(f'{metric}_gap_rules_metrics', gap_dict[metric], exclude_from_csv=True)

        # Log episodic policy
        for event_type in info.get('episode_logs', {}):
            self.writers[f"{event_type}"].add_scalar(f'{self.phase}/episodic_policy', info['episode_logs'][f"{event_type}"], global_step=info['n_calls'])
            self.writers[f"{event_type}"].add_scalar(f'{self.phase}/diversity_policy', info['diversity_episode_logs'][f"{event_type}"], global_step=info['n_calls'])
        
        # if 'episode_logs' in info:
        #     self._log_metrics('episodic_policy', info['episode_logs'], exclude_from_csv=True)
        
        if 'episodic_inserted_logs' in info:
            self._log_metrics('episodic_inserted_logs', info['episodic_inserted_logs'], exclude_from_csv=True)
            
        if 'episodic_inserted_logs' in info and 'episode_logs' in info:
            self._log_metrics('actual_quota', info['episodic_inserted_logs']/info['total_episode_logs'])
        
        # if 'diversity_episode_logs' in info:
        #     self._log_metrics('diversity_episode_logs', info['diversity_episode_logs'], exclude_from_csv=True)
        
        for event_type in self.event_types:
            self.writers[event_type].add_scalar(f'{self.phase}/real_relevant_distribution', info['real_relevant_distribution'].get(event_type, 0), global_step=info['n_calls'])
            self.writers[event_type].add_scalar(f'{self.phase}/fake_relevant_distribution', info['fake_relevant_distribution'].get(event_type, 0), global_step=info['n_calls'])
        
        # if 'real_relevant_distribution' in info:
        #     current_sum = np.sum(list(info['real_relevant_distribution'].values()))
        #     for k,v in info['real_relevant_distribution'].items():
        #         info['real_relevant_distribution'][k] = v / current_sum
        #     self._log_metrics('real_relevant_distribution', info['real_relevant_distribution'], exclude_from_csv=True)
        
        # if 'fake_relevant_distribution' in info:
        #     current_sum = np.sum(list(info['fake_relevant_distribution'].values()))
        #     for k,v in info['fake_relevant_distribution'].items():
        #         info['fake_relevant_distribution'][k] = v / current_sum
        #     self._log_metrics('fake_relevant_distribution', info['fake_relevant_distribution'], exclude_from_csv=True)
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

        # Log episode end metrics
        if info.get('done', False):
            self.log_episode_metrics(info, self.training_env)
            
        all_steps = info.get('all_steps_counter', 0)
        self.logger.dump(self.n_calls)
        return True


class CustomEvalCallback3( MetricsLoggerCallback, EvalCallback):
    def __init__(self, 
                 eval_env,
                 log_dir, rules, event_types,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1,
                 writers=None):
        
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
        
    def evaluate_policy(self, *args, **kwargs):
        """Override evaluate_policy to collect info during evaluation"""
        self.eval_infos = []
        
        def _log_info_callback(locals_, globals_):
            info = locals_['info']
            info['n_calls'] = self.n_calls  # Add current step count to info
            self.eval_infos.append(info)
            return True
            
        kwargs['callback'] = _log_info_callback
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
                         'distribution_value', 'inserted_logs', 'total_current_logs']:
                if metric in info:
                    mean_metrics[metric].append(info[metric])
                    
        # Log mean metrics
        for metric, values in mean_metrics.items():
            try:
                self._log_metrics(f'mean_{metric}', np.mean(values))
            except Exception as e:
                print(f"Error logging metric {metric}: {e}")
        
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
                print(f"Splunk license usage is low: {remaining_mb} MB remaining out of {quota_mb} MB")
                self.logger.record('splunk/remaining_mb', remaining_mb)
                self.logger.record('splunk/quota_mb', quota_mb)
                self.logger.dump(self.n_calls)
                # stop training 
                return False
            else:
                print(f"Splunk license usage is sufficient: {remaining_mb} MB remaining out of {quota_mb} MB")
                self.logger.record('splunk/license_usage', remaining_mb)
                self.logger.record('splunk/quota_mb', quota_mb)
                self.logger.dump(self.n_calls)
        return True