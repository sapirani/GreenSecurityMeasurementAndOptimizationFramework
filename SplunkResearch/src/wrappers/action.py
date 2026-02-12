import datetime
import logging
import subprocess
import sys
from time import sleep
from gymnasium.core import ActionWrapper
from gymnasium import make, spaces
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import sys
import urllib3

# Suppress insecure request warnings (equivalent to curl -k silent mode)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sys.path.insert(1, '/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch/src')
from config import config
from wrappers.action_interpreters import (
    ActionContext,
    ActionInterpreter,
    LearnableVolumeDecorator,
    SoftmaxDistributionInterpreter,
    SmoothTriggerInterpreter,
)

logger = logging.getLogger(__name__)
SPLUNK_BINARY_PATH = config.get('splunk.binary_path', '/opt/splunk/bin/splunk')


class Action(ActionWrapper):
    """Wrapper for managing log injection actions.

    Delegates action interpretation to an :class:`ActionInterpreter` strategy
    object.  Side-effects (episode tracking, distribution updates, Splunk I/O)
    remain here so interpreters stay pure.
    """

    def __init__(self, env, interpreter: ActionInterpreter, test_random=False):
        super().__init__(env)

        self.interpreter = interpreter

        # Build action space from interpreter
        self.action_space = interpreter.get_action_space(
            self.unwrapped.top_logtypes,
            self.unwrapped.relevant_logtypes,
        )

        self.episodic_logs_to_inject = []
        self.current_logs = {}

        self.episode_logs = {
            f"{key[0]}_{key[1]}_{istrigger}": 0
            for key in self.unwrapped.top_logtypes
            for istrigger in [0, 1]
        }
        self.remaining_quota = 0
        self.inserted_logs = 0
        self.diversity_factor = config.get('action.diversity_factor', 30)
        self.diversity_episode_logs = {
            f"{key[0]}_{key[1]}_{istrigger}": 0
            for key in self.unwrapped.top_logtypes
            for istrigger in ([0, 1] if key in self.unwrapped.relevant_logtypes else [0])
        }
        self.info = {}
        self._disable_injection = False
        self.test_random = test_random
        self.current_real_quantity = 0

        self.fake_storage_state = []
        self.logs_to_delete = []
        self.action_time_indexer = {}
        self.log_type_indexer = {}
        for i, logtype in enumerate(self.unwrapped.top_logtypes):
            self.log_type_indexer[f"{logtype[0]}_{logtype[1]}_0"] = i
            if logtype in self.unwrapped.relevant_logtypes:
                self.log_type_indexer[f"{logtype[0]}_{logtype[1]}_1"] = i + len(self.unwrapped.top_logtypes)

    # ------------------------------------------------------------------
    # Context builder (read-only snapshot passed to interpreter)
    # ------------------------------------------------------------------

    def _build_context(self) -> ActionContext:
        rel_index = {
            lt: self.unwrapped.relevant_logtypes.index(lt)
            for lt in self.unwrapped.relevant_logtypes
        }
        return ActionContext(
            top_logtypes=self.unwrapped.top_logtypes,
            relevant_logtypes=self.unwrapped.relevant_logtypes,
            relevant_logtypes_index=rel_index,
            additional_percentage=self.unwrapped.config.additional_percentage,
            base_log_count=config.get('action.base_log_count', 20000),
            diversity_factor=config.get('action.diversity_factor', 30),
            softmax_temperature=config.get('action.softmax_temperature', 20),
            sigmoid_sharpness=config.get('action.sigmoid_sharpness', 10),
        )

    # ------------------------------------------------------------------
    # Core action → side-effects
    # ------------------------------------------------------------------

    def action(self, raw_action):
        """Interpret raw NN output and apply side-effects."""
        logger.debug(f"Raw action: {raw_action}")

        ctx = self._build_context()
        plan = self.interpreter.interpret(raw_action, ctx)

        # Apply side-effects from the plan
        self.inserted_logs = 0
        self.current_logs = {}

        for key, entry in plan.logs_to_inject.items():
            log_count = entry['count']
            self.inserted_logs += log_count
            self.unwrapped.episodic_inserted_logs += log_count
            self.unwrapped.episodic_fake_logs_qnt += log_count

            self.current_logs[key] = log_count
            self.episode_logs[key] += log_count
            self.diversity_episode_logs[key] = max(
                entry['diversity'],
                self.diversity_episode_logs.get(key, 0),
            )

        return plan.logs_to_inject

    # ------------------------------------------------------------------
    # Quota
    # ------------------------------------------------------------------

    def _calculate_quota(self) -> None:
        """Calculate injection quotas"""
        step_size_multiplier = config.get('action.step_size_multiplier', 2000)
        self.step_size = int(
            (self.unwrapped.time_manager.step_size // 3600)
            * step_size_multiplier
            * self.unwrapped.config.additional_percentage
        )
        self.remaining_quota = self.step_size

    # ------------------------------------------------------------------
    # Distribution tracking
    # ------------------------------------------------------------------

    def update_fake_distribution(self):
        """Update fake distribution with injected logs"""
        for logtype, count in self.current_logs.items():
            formated_logtype = (*logtype.split('_')[:-1],)
            if formated_logtype in self.unwrapped.top_logtypes:
                self.unwrapped.fake_distribution[formated_logtype] += count
                self.unwrapped.ac_fake_distribution[formated_logtype] += count

        self.unwrapped.fake_state = np.array([
            self.unwrapped.fake_distribution[k] / (sum(self.unwrapped.fake_distribution.values()) + 1e-8)
            for k in self.unwrapped.top_logtypes if k != 'other'
        ])
        self.unwrapped.ac_fake_state = np.array([
            self.unwrapped.ac_fake_distribution[k] / (sum(self.unwrapped.ac_fake_distribution.values()) + 1e-8)
            for k in self.unwrapped.top_logtypes if k != 'other'
        ])
        cache = self.unwrapped.logtype_key_cache
        self.unwrapped.fake_relevant_distribution = {
            cache[logtype]: self.unwrapped.ac_fake_state[self.unwrapped.top_logtypes_indices[logtype]]
            for logtype in self.unwrapped.top_logtypes
        }

    # ------------------------------------------------------------------
    # Injection / deletion helpers (unchanged)
    # ------------------------------------------------------------------

    def disable_injection(self):
        """Disable log injection"""
        self._disable_injection = True
        logger.info("Log injection disabled")

    def collect_configs(self, logs_to_inject, time_range, injection_id):
        """Inject logs into environment"""
        logger.info(f"Action time range: {time_range}")
        if self._disable_injection:
            logger.info("Log injection disabled, not injecting logs")
            return
        configs = []
        for logtype, log_info in logs_to_inject.items():
            logsource, eventcode, is_trigger = logtype.split('_')
            count, diversity = log_info['count'], log_info['diversity']
            if count == 0:
                continue
            configs.append({
                'logsource': logsource,
                'eventcode': eventcode,
                'istrigger': is_trigger,
                'time_range': time_range,
                'num_logs': count,
                'diversity': diversity,
                'injection_id': injection_id
            })
        return configs

    def save_mixed_batch(self, batch_of_tuples):
        """Takes a batch of (log, source) tuples, groups them,
        and writes to separate files efficiently."""
        buckets = {}
        for log_entry, source in batch_of_tuples:
            if source not in buckets:
                buckets[source] = []
            buckets[source].append(log_entry)

        for source, logs in buckets.items():
            self.unwrapped.splunk_tools.write_logs_to_monitor(logs, source)

    def inject_episodic_logs(self, injection_id):
        """Inject episodic logs into environment"""
        if self._disable_injection:
            logger.info("Log injection disabled, not injecting episodic logs")
            return
        configs = []
        for logs_to_inject, time_window in self.episodic_logs_to_inject:
            time_range = time_window.to_tuple()
            logger.info(f"Injecting episodic logs: {logs_to_inject} at time range {time_range}")
            configs.extend(self.collect_configs(logs_to_inject, time_range, injection_id=injection_id))

        security_log_file_path = self.unwrapped.splunk_tools.log_file_prefix + "/wineventlog:security.txt"
        system_log_file_path = self.unwrapped.splunk_tools.log_file_prefix + "/wineventlog:system.txt"

        for batch in self.unwrapped.log_generator.generate_massive_stream(configs, batch_size=50000):
            self.save_mixed_batch(batch)

        self.flush_logs(security_log_file_path, log_type="Security")
        self.flush_logs(system_log_file_path, log_type="System")

        results = 0
        dt_all_start_date = self.episodic_logs_to_inject[0][1].start_dt
        dt_all_end_date = self.episodic_logs_to_inject[-1][1].end_dt
        all_start_date = dt_all_start_date.strftime("%Y-%m-%dT%H:%M:%S")
        all_end_date = dt_all_end_date.strftime("%Y-%m-%dT%H:%M:%S")

        logs_count = sum([sum([logs[x]['count'] for x in logs]) for logs, _ in self.episodic_logs_to_inject]) + 0.0000001
        logger.info(f"Total episodic logs to inject: {logs_count}")
        attempts = 0
        while (logs_count - results) / logs_count > 0.01:
            sleep(2)
            default_host = config.get('splunk.default_host', 'dt-splunk')
            secondary_host = config.get('hosts.secondary', '132.72.81.150') # No hosts.secondary in yaml
            query = f'index={self.unwrapped.splunk_tools.index_name} host IN ("{default_host}", {secondary_host}) | stats count'
            results = self.unwrapped.splunk_tools.run_search(query, all_start_date, all_end_date)
            results = int(results[0]['count'])
            logger.info(f"Waiting for logs to be indexed: {results}/{logs_count}, {(logs_count - results) / logs_count * 100:.2f}% remaining")
            attempts += 1
            if attempts > 10:
                logger.warning("Max attempts reached while waiting for logs to be indexed.")
                break

        self.unwrapped.log_generator.logs_to_delete = {}

    def flush_logs(self, log_file_path, log_type="Security"):
        splunk_port = config.get('splunk.port', 8089)
        splunk_mgmt_uri = f"https://{self.unwrapped.splunk_tools.splunk_host}:{splunk_port}"
        endpoint = f"{splunk_mgmt_uri}/services/receivers/stream"
        print(endpoint)
        username = self.unwrapped.splunk_tools.splunk_username
        password = self.unwrapped.splunk_tools.splunk_password
        index = self.unwrapped.splunk_tools.index_name
        sourcetype = "WinEventLog:" + log_type

        secondary_host = config.get('hosts.secondary', '132.72.81.150')
        params = {
            "index": index,
            "sourcetype": sourcetype,
            "host": secondary_host
        }

        headers = {
            "x-splunk-input-mode": "streaming"
        }

        try:
            with open(log_file_path, 'rb') as f:
                response = requests.post(
                    endpoint,
                    auth=(username, password),
                    params=params,
                    headers=headers,
                    data=f,
                    verify=False,
                    timeout=30
                )
            response.raise_for_status()
            if response.text:
                logger.info(f"Splunk Response: {response.text}")

        except requests.exceptions.HTTPError as e:
            print(f"Splunk API Error: {e}", file=sys.stderr)
            print(f"Response Body: {e.response.text}", file=sys.stderr)

        except requests.exceptions.ConnectionError:
            print(f"Connection Failed: Could not reach {splunk_mgmt_uri}", file=sys.stderr)
            print("Check if Splunk is running and port 8089 is open.", file=sys.stderr)

        except FileNotFoundError:
            print(f"Error: Log file not found at {log_file_path}", file=sys.stderr)

        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step / reset
    # ------------------------------------------------------------------

    def step(self, action):
        """Inject logs and step environment"""
        if self.test_random:
            action = self.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._calculate_quota()
        logs_to_inject = self.action(action)
        logger.debug(f"Action window: {self.unwrapped.time_manager.action_window.to_tuple()}")

        self.episodic_logs_to_inject.append((logs_to_inject, self.unwrapped.time_manager.action_window))
        self.update_fake_distribution()

        info.update(self.get_injection_info())
        return obs, reward, terminated, truncated, info

    def get_injection_info(self):
        """Get information about current injections"""
        return {
            'current_logs': self.current_logs,
            'episode_logs': self.episode_logs,
            'diversity_episode_logs': self.diversity_episode_logs,
            'remaining_quota': self.remaining_quota,
            'inserted_logs': self.inserted_logs,
            'episodic_inserted_logs': self.unwrapped.episodic_inserted_logs,
            'fake_relevant_distribution': self.unwrapped.fake_relevant_distribution,
        }

    def reset(self, **kwargs):
        """Reset tracking on environment reset"""
        self.current_logs = {}
        self.episode_logs = {
            f"{key[0]}_{key[1]}_{istrigger}": 0
            for key in self.unwrapped.top_logtypes
            for istrigger in [0, 1]
        }
        self._calculate_quota()
        self.diversity_episode_logs = {
            f"{key[0]}_{key[1]}_{istrigger}": 0
            for key in self.unwrapped.top_logtypes
            for istrigger in ([0, 1] if key in self.unwrapped.relevant_logtypes else [0])
        }
        self.episodic_logs_to_inject = []
        self.logs_to_delete = {}
        self.info = kwargs["options"]
        self.unwrapped.episodic_inserted_logs = 0

        obs, info = self.env.reset(**kwargs)
        self.unwrapped.log_generator.fake_splunk_state = {}
        self.unwrapped.log_generator.logs_to_delete = {}
        return obs, info


# ======================================================================
# Registry & factory
# ======================================================================

ACTION_INTERPRETER_REGISTRY = {
    # Backward-compatible names
    'Action8': SoftmaxDistributionInterpreter,
    'Action12': lambda: LearnableVolumeDecorator(SmoothTriggerInterpreter()),
    # Descriptive names
    'SoftmaxDistribution': SoftmaxDistributionInterpreter,
    'SmoothTrigger': SmoothTriggerInterpreter,
    'SmoothTriggerVolume': lambda: LearnableVolumeDecorator(SmoothTriggerInterpreter()),
    'SoftmaxDistributionVolume': lambda: LearnableVolumeDecorator(SoftmaxDistributionInterpreter()),
}


def create_action_wrapper(env, action_type: str, test_random: bool = False) -> Action:
    """Create an Action wrapper with the specified interpreter.

    Args:
        env: The gymnasium environment to wrap.
        action_type: Key into ACTION_INTERPRETER_REGISTRY.
        test_random: If True, sample random actions instead of using model output.

    Returns:
        Action wrapper instance.
    """
    if action_type not in ACTION_INTERPRETER_REGISTRY:
        raise ValueError(
            f"Unknown action type '{action_type}'. "
            f"Available: {list(ACTION_INTERPRETER_REGISTRY.keys())}"
        )
    entry = ACTION_INTERPRETER_REGISTRY[action_type]
    interpreter = entry()
    return Action(env, interpreter, test_random)
