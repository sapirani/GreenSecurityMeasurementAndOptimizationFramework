import datetime
import logging
import subprocess
import sys
from time import sleep
from unittest import result
from gymnasium.core import ActionWrapper
from gymnasium import make, spaces
import numpy as np

logger = logging.getLogger(__name__)
SPLUNK_BINARY_PATH = "/opt/splunk/bin/splunk"
class Action(ActionWrapper):
    """Wrapper for managing log injection actions"""
    
    def __init__(self, env, test_random=False):
        super().__init__(env)
        
        # Create action space:
        # First value is quota percentage (0-1)
        # then values are distribution across log types
        # then values are triggering levels for each log type
        # then values are diversity levels for each log type
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(1 + len(self.unwrapped.relevant_logtypes)*3,),
            dtype=np.float32
        )
        self.episodic_logs_to_inject = []
        # Track injected logs
        self.current_logs = {}
        
        self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.relevant_logtypes for istrigger in [0, 1]}
        self.remaining_quota = 0
        self.inserted_logs = 0
        self.diversity_factor = 30
        self.diversity_episode_logs = {f"{key[0]}_{key[1]}_1":0 for key in self.unwrapped.relevant_logtypes}
        self.info = {}
        self._disable_injection = False
        self.test_random = test_random
    # def _calculate_quota(self) -> None:
    #     """Calculate injection quotas"""
    #     self.total_additional_logs = (self.unwrapped.config.additional_percentage * 
    #                                 self.unwrapped.config.search_window * 
    #                                 self.unwrapped.config.logs_per_minute)
        
    #     self.step_size = int((self.total_additional_logs // self.unwrapped.config.search_window) * 
    #                         self.unwrapped.config.action_duration // 60)
    #     self.remaining_quota = self.step_size
    
        self.fake_storage_state = []
        self.logs_to_delete = []
        self.action_time_indexer = {}
        self.log_type_indexer = {}
        for i, logtype in enumerate(self.unwrapped.top_logtypes):
            self.log_type_indexer[f"{logtype[0]}_{logtype[1]}_0"] = i
            if logtype in self.unwrapped.relevant_logtypes:
                self.log_type_indexer[f"{logtype[0]}_{logtype[1]}_1"] = i + len(self.unwrapped.top_logtypes)

    def _calculate_quota(self) -> None:
        """Calculate injection quotas"""

        self.step_size = int((self.unwrapped.time_manager.step_size//3600) * 2000 * self.unwrapped.config.additional_percentage)
        self.remaining_quota = self.step_size
    
            
    def update_fake_distribution(self): 
        """Update fake distribution with injected logs"""
        # Add injected logs to existing real distribution
        # self.fake_distribution = self.real_distribution.copy()#BUG!!

        for logtype, count in self.current_logs.items():
            formated_logtype = (*logtype.split('_')[:-1],)
            if formated_logtype in self.unwrapped.top_logtypes:
                self.unwrapped.fake_distribution[formated_logtype] += count
                self.unwrapped.ac_fake_distribution[formated_logtype] += count
                
            # else:
            #     self.fake_distribution['other'] += count

            # else:
            #     self.ac_fake_distribution['other'] += count
            
        # Normalize fake distribution
        self.unwrapped.fake_state = np.array([self.unwrapped.fake_distribution[k]/(sum(self.unwrapped.fake_distribution.values())+1e-8) for k in self.unwrapped.top_logtypes if k != 'other'])
        self.unwrapped.ac_fake_state = np.array([self.unwrapped.ac_fake_distribution[k]/(sum(self.unwrapped.ac_fake_distribution.values())+1e-8) for k in self.unwrapped.top_logtypes if k != 'other'])
        self.unwrapped.fake_relevant_distribution = {"_".join(logtype): self.unwrapped.ac_fake_state[self.unwrapped.relevant_logtypes_indices[logtype]] for logtype in self.unwrapped.top_logtypes}
        return
        
    def action(self, action):
        """Convert raw action to log injection dictionary"""
        # Split action into quota and distribution
        quota_pct = action[0]
        distribution = action[1:1+len(self.unwrapped.relevant_logtypes)]
        trigger_levels = action[1+len(self.unwrapped.relevant_logtypes):2*len(self.unwrapped.relevant_logtypes)+1]
        diversity_levels = action[2*len(self.unwrapped.relevant_logtypes)+1:] * self.diversity_factor
        
        # Normalize distribution
        distribution = distribution / (np.sum(distribution) + 1e-8)
        
        # Calculate number of logs to inject
        num_logs = int(quota_pct * self.remaining_quota)
        self.inserted_logs = num_logs
        # self.remaining_quota = self.quota - num_logs
        
        # Distribute logs among types
        logs_to_inject = {}
        for i, logtype in enumerate(self.unwrapped.relevant_logtypes):
            for is_trigger in [False, True]:
                log_count = int(distribution[i] * num_logs * (is_trigger * trigger_levels[i] + (1-is_trigger) * (1 - trigger_levels[i]))) 
                key = f"{logtype[0]}_{logtype[1]}_{int(is_trigger)}"
                if log_count > 0:
                    logs_to_inject[key] = {
                        'count': log_count,
                        'diversity': round(diversity_levels[i])
                    }
                    
                    # Track logs
                    self.current_logs[key] = log_count


                    self.episode_logs[key] += log_count
                    self.diversity_episode_logs[key] = max(round(diversity_levels[i]), self.diversity_episode_logs[key])
        
        return logs_to_inject
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
            # fake_logs = self.unwrapped.log_generator.prepare_taskes(
            #     logsource, eventcode, is_trigger,
            #     time_range, count, diversity, injection_id
            #     )
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

          
            
    def delete_episodic_logs(self, deletion_plan):
        """Delete episodic logs from environment according to logs_to_delete"""
        if self._disable_injection:
            logger.info("Log deletion disabled, not deleting episodic logs")
            return
        for time_range, logs in deletion_plan.items():
            by_log_type_deletion_list = []
            for logtype, variations in logs.items():
                if "RemoveAll" in variations:
                    by_log_type_deletion_list.append((logtype, variations.split('_')[1]))
                    deletion_plan[time_range][logtype] = {}
                    continue
            
            if by_log_type_deletion_list:   
                condition = " OR ".join([f"(source={logtype.split('_')[0]} EventCode={logtype.split('_')[1]} var_id>={var_id})" for logtype, var_id in by_log_type_deletion_list])
                condition = condition.replace("var_id>=0", "")
                self.unwrapped.splunk_tools.delete_fake_logs(
                    time_range, f'{condition} ', None)
                logger.info(f"Deleting all episodic logs at time range {time_range} for logtypes {by_log_type_deletion_list}")
            
            
            
            
            # for logtype, variations in logs.items():
            #     if variations == "RemoveAll_0": # maybe for all the time range together?????????????
            #         by_log_type_deletion_list.append(logtype)
            #         deletion_plan[time_range][logtype] = {}
            #         continue
                    
            #     logsource, eventcode, is_trigger = logtype.split('_')
            #     # max_var_id = max(variations.keys())
            #     for variation_id, count in variations.items():
            #         if count > 0:
            #             condition = f"source={logsource} EventCode={eventcode} var_id >= {variation_id}"                        
            #             self.unwrapped.splunk_tools.delete_fake_logs(
            #                 time_range, f'{condition}') 
            #                 # time_range, f'{condition} [search index=main host="dt-splunk" {condition}| head {count}| eval t=_time | return {count} _time=t ]', count) 
            #             logger.info(f"Deleting episodic logs at time range {time_range} for logtype {logtype} variation>={variation_id}")
            #             break
            # if by_log_type_deletion_list:   
            #     condition = " OR ".join([f"(source={logtype.split('_')[0]} EventCode={logtype.split('_')[1]})" for logtype in by_log_type_deletion_list])
            #     self.unwrapped.splunk_tools.delete_fake_logs(
            #         time_range, f'{condition} ', None)
            #     logger.info(f"Deleting all episodic logs at time range {time_range} for logtypes {by_log_type_deletion_list}")
                        
        # TBD
        # for time_index in self.logs_to_delete:
        #     time_range = list(self.action_time_indexer.keys())[list(self.action_time_indexer.values()).index(time_index)]
        #     for key in 
        #     logger.info(f"Deleting episodic logs at time range {time_range}")
        #     for logtype, count in logs.items():
        #         if count > 0:
        #             self.unwrapped.splunk_tools.delete_fake_logs(

    def save_mixed_batch(self, batch_of_tuples):
            """
            Takes a batch of (log, source) tuples, groups them, 
            and writes to separate files efficiently.
            """
            buckets = {}
            
            # 1. Sort the logs into buckets (in memory)
            for log_entry, source in batch_of_tuples:
                if source not in buckets:
                    buckets[source] = []
                buckets[source].append(log_entry)
                
            # 2. Write each bucket to its own file
            for source, logs in buckets.items():
                # We reuse your existing function!
                self.unwrapped.splunk_tools.write_logs_to_monitor(logs, source)
                
    def inject_episodic_logs(self, injection_id):
        """Inject episodic logs into environment"""
        if self._disable_injection:
            logger.info("Log injection disabled, not injecting episodic logs")
            return
        configs = []
        for logs_to_inject, time_range in self.episodic_logs_to_inject:
            logger.info(f"Injecting episodic logs: {logs_to_inject} at time range {time_range}")
            configs.extend(self.collect_configs(logs_to_inject, time_range, injection_id=injection_id))
        
        # logger.info(f"Waiting for {sum(self.episode_logs.values())} logs to be written")
        # sleep(sum(self.episode_logs.values())/4500)  # wait for logs to be written
        security_log_file_path = self.unwrapped.splunk_tools.log_file_prefix + "wineventlog:security.txt"
        system_log_file_path = self.unwrapped.splunk_tools.log_file_prefix + "wineventlog:system.txt"
        
        for batch in self.unwrapped.log_generator.generate_massive_stream(configs, batch_size=50000):
            self.save_mixed_batch(batch)
            
        self.delete_episodic_logs(self.unwrapped.log_generator.logs_to_delete)
        
        self.flush_logs(security_log_file_path, log_type="Security")
        self.flush_logs(system_log_file_path, log_type="System")
        # check if all logs with injection id in splunk
        results = 0
        start_date = datetime.datetime.strptime(self.episodic_logs_to_inject[0][1][0], '%m/%d/%Y:%H:%M:%S').strftime("%Y-%m-%dT%H:%M:%S")
        end_date = datetime.datetime.strptime(self.episodic_logs_to_inject[-1][1][1], '%m/%d/%Y:%H:%M:%S').strftime("%Y-%m-%dT%H:%M:%S")
        
        logs_len = sum([sum([logs[x]['count'] for x in logs]) for logs, _ in self.episodic_logs_to_inject]) - sum([sum([info[log_type][var_id] for log_type in info for var_id in info[log_type]]) for time_range, info in self.unwrapped.log_generator.logs_to_delete.items()])
        logger.info(f"Total episodic logs to inject: {logs_len}")
        
        while logs_len > results + 50:
            sleep(2)
            query = f'index=main host="dt-splunk" | stats count'
            # query = f'index=main host="dt-splunk" Injection_id={injection_id} | stats count'
            results = self.unwrapped.splunk_tools.run_search(query, start_date, end_date)            
            results = int(results[0]['count'])
            logger.info(f"Waiting for logs to be indexed: {results}/{logs_len}")
        
        # delete the logs according to self.unwrapped.log_generator.logs_to_delete
        # reset logs to delete
        self.unwrapped.log_generator.logs_to_delete = {}
            
        # waiting_time = logs_len/ 5000  # 30000 logs per second 
        # sleep(waiting_time)  # wait for logs to be indexed


    def flush_logs(self, log_file_path, log_type="Security"):
        cmd_list = [
            'sudo', '-S',
            SPLUNK_BINARY_PATH,
            "add",
            "oneshot",
            log_file_path,
            "-index", "main",
            "-sourcetype", "WinEventLog:" + log_type,
            "-auth", "shouei:123456789" # Add this if you need to authenticate
        ]

        # print(f"Injecting {log_file_path} into Splunk. This will block until complete...")
        # 4. Run the command
        try:
            # input: Sends the password string to the command's stdin
            # text=True: Encodes the input and decodes stdout/stderr as text (using UTF-8)
            # capture_output=True: Captures stdout and stderr
            # check=True: Raises an error if the command returns a non-zero exit code
            result = subprocess.run(
                cmd_list,
                input=" \n",
                text=True,
                capture_output=True,
                check=True
            )
            
            # If the command was successful
            # print("Command executed successfully:")
            if result.stdout:
                logger.info("\n--- STDOUT ---")
                logger.info(result.stdout)
            if result.stderr:
                logger.info("\n--- STDERR ---")
                logger.info(result.stderr)

        except subprocess.CalledProcessError as e:
            # This block runs if check=True and the command fails
            print("Command failed:", file=sys.stderr)
            print(f"Return Code: {e.returncode}", file=sys.stderr)
            
            if e.stdout:
                print("\n--- STDOUT ---", file=sys.stderr)
                print(e.stdout, file=sys.stderr)
            
            if e.stderr:
                print("\n--- STDERR ---", file=sys.stderr)
                # Check for the specific sudo error
                if "no password was provided" in e.stderr or "incorrect password" in e.stderr:
                    print("Hint: The password may have been incorrect.", file=sys.stderr)
                else:
                    print(e.stderr, file=sys.stderr)

        except FileNotFoundError:
            # This block runs if "sudo" or the splunk path is not found
            print(f"Error: Command not found.", file=sys.stderr)
            print(f"Please check the path: {cmd_list[2]}", file=sys.stderr)
        
        
    def step(self, action):
        """Inject logs and step environment"""
        if self.test_random:
            action = self.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        # logger.info(f"Raw action: {action}")
        self._calculate_quota()
        logs_to_inject = self.action(action)
        # logger.info(f"Action: {logs_to_inject}")
        logger.info(f"Action window: {self.unwrapped.time_manager.action_window.to_tuple()}")
        # self.inject_logs(logs_to_inject, self.env.time_manager.action_window.to_tuple())

        



        self.episodic_logs_to_inject.append((logs_to_inject, self.unwrapped.time_manager.action_window.to_tuple()))
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
            
            
            # 'quota_used_pct': (self.quota - self.remaining_quota) / self.quota
        }

    def reset(self, **kwargs):
        """Reset tracking on environment reset"""
        self.current_logs = {}
        self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.relevant_logtypes for istrigger in [0, 1]}
        self._calculate_quota()
        self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.relevant_logtypes for istrigger in [0, 1]}
        self.episodic_logs_to_inject = []

        # self.remaining_quota = self.quota
        self.info = kwargs["options"]
        self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
        
        obs, info = self.env.reset(**kwargs)
        return obs, info


class Action2(Action):
    
    def __init__(self, env, test_random=False):
        super().__init__(env, test_random)
    
    def action(self, action):
        """Convert raw action to log injection dictionary"""
        # Split action into quota and distribution
        quota_pct = action[0]
        distribution = action[1:1+len(self.unwrapped.relevant_logtypes)]
        trigger_levels = action[1+len(self.unwrapped.relevant_logtypes):2*len(self.unwrapped.relevant_logtypes)+1]
        diversity_levels = action[2*len(self.unwrapped.relevant_logtypes)+1:] * self.diversity_factor
        
        # Normalize distribution
        distribution = distribution / (np.sum(distribution) + 1e-8)
        
        # Calculate number of logs to inject
        num_logs = int(quota_pct * self.remaining_quota)
        self.inserted_logs = num_logs
        # self.remaining_quota = self.quota - num_logs
        
        # Distribute logs among types
        logs_to_inject = {}
        for i, logtype in enumerate(self.unwrapped.relevant_logtypes):
            is_trigger = round(trigger_levels[i])
            log_count = int(distribution[i] * num_logs)
            key = f"{logtype[0]}_{logtype[1]}_{int(is_trigger)}"
            if log_count > 0:
                logs_to_inject[key] = {
                    'count': log_count,
                    'diversity': round(diversity_levels[i])
                }
                
                # Track logs
                self.current_logs[key] = log_count


                self.episode_logs[key] += log_count
                self.diversity_episode_logs[key] = max(round(diversity_levels[i]), self.diversity_episode_logs[key])
        
        return logs_to_inject

class Action3(Action):
    
    def __init__(self, env, test_random=False):
        super().__init__(env, test_random)
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(1 + len(self.unwrapped.relevant_logtypes)*2,),
            dtype=np.float32
        )
    
    def action(self, action):
        """Convert raw action to log injection dictionary"""
        # Split action into quota and distribution
        quota_pct = action[0]
        distribution = action[1:1+len(self.unwrapped.relevant_logtypes)]
        trigger_levels = action[1+len(self.unwrapped.relevant_logtypes):2*len(self.unwrapped.relevant_logtypes)+1]* self.diversity_factor

        # Normalize distribution
        distribution = distribution / (np.sum(distribution) + 1e-8)
        
        # Calculate number of logs to inject
        num_logs = int(quota_pct * self.remaining_quota)
        self.inserted_logs = num_logs
        # self.remaining_quota = self.quota - num_logs
        
        # Distribute logs among types
        logs_to_inject = {}
        for i, logtype in enumerate(self.unwrapped.relevant_logtypes):
            trigger_level_i = round(trigger_levels[i])
            is_trigger = min(trigger_level_i, 1)
            log_count = int(distribution[i] * num_logs)
            key = f"{logtype[0]}_{logtype[1]}_{int(is_trigger)}"
            if log_count > 0:
                logs_to_inject[key] = {
                    'count': log_count,
                    'diversity': trigger_level_i
                }
                
                # Track logs
                self.current_logs[key] = log_count


                self.episode_logs[key] += log_count
                self.diversity_episode_logs[key] = max(trigger_level_i, self.diversity_episode_logs[key])
        
        return logs_to_inject

class Action4(Action):
        
        def __init__(self, env, test_random=False):
            super().__init__(env, test_random)
            self.action_space = spaces.Box(
                low=0,
                high=1,
                shape=(1 + len(self.unwrapped.relevant_logtypes),),
                dtype=np.float32
            )
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}

        def action(self, action):
            """Convert raw action to log injection dictionary"""
            # Split action into quota and distribution
            distribution = action

            
            # Normalize distribution
            distribution = distribution / (np.sum(distribution) + 1e-8) #TODO try softmax
            
            # Calculate number of logs to inject
            num_logs = self.remaining_quota
            self.inserted_logs = num_logs
            # self.remaining_quota = self.quota - num_logs
            
            # Distribute logs among types
            logs_to_inject = {}
            allocation = self.minimize_distribution_distance(self.remaining_quota*distribution[-1])

            for i, logtype in enumerate(self.unwrapped.top_logtypes):
                trigger_level_i = 0
                is_trigger = min(trigger_level_i, 1)
                if logtype in self.unwrapped.relevant_logtypes:
                    index = self.unwrapped.relevant_logtypes.index(logtype)
                    log_count = int(distribution[index] * num_logs)
                    key = f"{logtype[0]}_{logtype[1]}_{int(is_trigger)}"
                else:
                    log_count = int(allocation[i])
                    key = f"{logtype[0]}_{logtype[1]}_{int(is_trigger)}"
                if log_count > 0:
                    logs_to_inject[key] = {
                        'count': log_count,
                        'diversity': 0
                    }
                    
                    # Track logs
                    self.current_logs[key] = log_count
    
    
                    self.episode_logs[key] += log_count
                    self.diversity_episode_logs[key] = 0
       
            return logs_to_inject
        
        def minimize_distribution_distance(self, quota):
            """Minimize distance between distributions"""
            real_distribution = self.obs[:len(self.unwrapped.top_logtypes)]
            fake_distribution = self.obs[len(self.unwrapped.top_logtypes):2*len(self.unwrapped.top_logtypes)]
            delta_distribution = real_distribution - fake_distribution
            delta_distribution = np.clip(delta_distribution, 0, 1)
            # take top 5 deltas
            delta_distribution_indices = np.argsort(delta_distribution)[-5:]
            top_5_normalized_delta_distribution = delta_distribution[delta_distribution_indices] / (np.sum(delta_distribution[delta_distribution_indices]) + 1e-8)
            allocation = []
            for i in range(len(self.unwrapped.top_logtypes)):
                if i in delta_distribution_indices:
                    allocation.append(top_5_normalized_delta_distribution[np.where(delta_distribution_indices == i)[0][0]] * quota)
                else:
                    allocation.append(0)

            return allocation

        def reset(self, **kwargs):
            """Reset tracking on environment reset"""
            self.current_logs = {}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}

            # self.remaining_quota = self.quota
            self.info = kwargs["options"]
            
            obs, info = self.env.reset(**kwargs)
            return obs, info
        
class Action5(Action):
        
        def __init__(self, env, test_random=False):
            super().__init__(env, test_random)
            self.action_space = spaces.Box(
                low=0,
                high=1,
                shape=(len(self.unwrapped.top_logtypes),),
                dtype=np.float32
            )
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}

        def action(self, action):
            """Convert raw action to log injection dictionary"""
            # Split action into quota and distribution
            distribution = action

            
            # Normalize distribution
            distribution = distribution / (np.sum(distribution) + 1e-8) #TODO try softmax
            logger.info(f"Action distribution: {distribution}")
            # Calculate number of logs to inject
            num_logs = self.remaining_quota
            self.inserted_logs = 0
            # self.remaining_quota = self.quota - num_logs
            
            # Distribute logs among types
            logs_to_inject = {}
            # max(0, (F_target_i * (N_real + Q_t) - R_t_i * N_real))
            current_real_quantity = self.info['total_episode_logs']
            real_distribution = self.obs[:len(self.unwrapped.top_logtypes)]
            action = (distribution * (current_real_quantity + self.remaining_quota) - real_distribution * current_real_quantity)
            action = [max(0, a) for a in action]
            for i, logtype in enumerate(self.unwrapped.top_logtypes):
                trigger_level_i = 0
                is_trigger = min(trigger_level_i, 1)
                log_count = int(action[i])
                key = f"{logtype[0]}_{logtype[1]}_{int(is_trigger)}"
                if log_count > 0:
                    logs_to_inject[key] = {
                        'count': log_count,
                        'diversity': 0
                    }
                    
                    # Track logs
                    self.current_logs[key] = log_count
    
    
                    self.episode_logs[key] += log_count
                    self.diversity_episode_logs[key] = 0
                    self.inserted_logs += log_count
            
            return logs_to_inject

    
        def reset(self, **kwargs):
            """Reset tracking on environment reset"""
            self.current_logs = {}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}

            # self.remaining_quota = self.quota
            self.info = kwargs["options"]
            
            obs, info = self.env.reset(**kwargs)
            return obs, info
        
class Action6(Action):
        """relevant logs = top logtypes"""
        
        def __init__(self, env, test_random=False):
            super().__init__(env, test_random)
            self.action_space = spaces.Box(
                low=0,
                high=1,
                shape=(len(self.unwrapped.top_logtypes),),
                dtype=np.float32
            )
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}

        def action(self, action):
            """Convert raw action to log injection dictionary"""
            # Split action into quota and distribution
            distribution = action

            
            # Normalize distribution
            distribution = distribution / (np.sum(distribution) + 1e-8) #TODO try softmax
            current_real_quantity = self.info['total_episode_logs']
            
            # Calculate number of logs to inject
            # num_logs = self.unwrapped.config.additional_percentage * current_real_quantity
            num_logs = self.remaining_quota
            self.inserted_logs = num_logs
            # self.remaining_quota = self.quota - num_logs
            
            # Distribute logs among types
            logs_to_inject = {}

            for i, logtype in enumerate(self.unwrapped.top_logtypes):
                trigger_level_i = 0
                is_trigger = min(trigger_level_i, 1)
                log_count = int(distribution[i] * num_logs)
                key = f"{logtype[0]}_{logtype[1]}_{int(is_trigger)}"
                if log_count > 0:
                    logs_to_inject[key] = {
                        'count': log_count,
                        'diversity': 0
                    }
                    
                    # Track logs
                    self.current_logs[key] = log_count
    
    
                    self.episode_logs[key] += log_count
                    self.diversity_episode_logs[key] = 0
            return logs_to_inject

    
        def reset(self, **kwargs):
            """Reset tracking on environment reset"""
            self.current_logs = {}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}

            # self.remaining_quota = self.quota
            self.info = kwargs["options"]
            
            obs, info = self.env.reset(**kwargs)
            return obs, info
        
class Action7(Action): # working!!!! 21/04/25
        """relevant logs = top logtypes"""
        
        def __init__(self, env, test_random=False):
            super().__init__(env, test_random)
            self.action_space = spaces.Box(
                low=0,
                high=.01,
                shape=(len(self.unwrapped.top_logtypes),),
                dtype=np.float32
            )
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0]}


        def action(self, action):
            """Convert raw action to log injection dictionary"""
            # Split action into quota and distribution
            distribution = action
            
            distribution /= (np.sum(distribution) + 1e-8) #TODO try softmax

            self.inserted_logs = 0
            
            # self.remaining_quota = self.quota - num_logs
            
            # Distribute logs among types
            logs_to_inject = {}

            for i, logtype in enumerate(self.unwrapped.top_logtypes):

                log_count = int(distribution[i] * 3000)
                # log_count = int(distribution[i] * 0.10 * self.unwrapped._normalize_factor)
                
                # log_count = int(distribution[i] * 0.005 * self.unwrapped._normalize_factor)
                self.inserted_logs += log_count
                self.unwrapped.episodic_inserted_logs += log_count
 
                if log_count > 0:
                    diversity = 0

                    is_trigger = int(np.ceil(diversity))
                    key = f"{logtype[0]}_{logtype[1]}_{int(is_trigger)}"
                    logs_to_inject[key] = {
                        'count': log_count,
                        'diversity': int(diversity * self.diversity_factor)
                    }
                    
                    # Track logs
                    self.current_logs[key] = log_count
    
    
                    self.episode_logs[key] += log_count
                    self.diversity_episode_logs[key] = max(diversity, self.diversity_episode_logs[key])
            return logs_to_inject

    
        def reset(self, **kwargs):
            """Reset tracking on environment reset"""
            self.current_logs = {}
            # self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}

            # self.remaining_quota = self.quota
            self.info = kwargs["options"]
            self.unwrapped.episodic_inserted_logs = 0
            
            obs, info = self.env.reset(**kwargs)
            return obs, info
        
class Action8(Action):
        """relevant logs = top logtypes"""
        
        def __init__(self, env, test_random=False):
            super().__init__(env, test_random)
            low_bounds = np.zeros(len(self.unwrapped.top_logtypes) + len(self.unwrapped.relevant_logtypes))
            # low_bounds[-1] = 0.1
            # self.action_space = spaces.MultiDiscrete(
            #     [10] * len(self.unwrapped.top_logtypes) + [31] * len(self.unwrapped.relevant_logtypes)
            # )
            self.action_space = spaces.Box(
                low=low_bounds,
                high=np.ones(len(self.unwrapped.top_logtypes) + len(self.unwrapped.relevant_logtypes)),
                shape=(len(self.unwrapped.top_logtypes)+ len(self.unwrapped.relevant_logtypes),),
                dtype=np.float32
            )
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_1":0 for key in self.unwrapped.top_logtypes}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self.diversity_factor = 30
            self.current_real_quantity = 0
            
        def action(self, action):
            """Convert raw action to log injection dictionary"""
            # Split action into quota and distribution
            # check zero action 
            logger.info(f"Raw action: {action}")
            distribution = action[:len(self.unwrapped.top_logtypes)]
            scaled_logits = 20 * distribution
            # softmax normalization
            exp_vals = np.exp(scaled_logits - np.max(scaled_logits))
            distribution = exp_vals / exp_vals.sum()
            # distribution /= (np.sum(distribution) + 1e-8) 
            
            diversity_list = action[len(self.unwrapped.top_logtypes):]
            # current_quota = action[-1]
            # num_logs = 20000
            num_logs = self.unwrapped.config.additional_percentage * self.current_real_quantity
            # num_logs = 1000
            self.inserted_logs = 0
            self.current_logs = {}
            # self.remaining_quota = self.quota - num_logs
            
            # Distribute logs among types
            logs_to_inject = {}

            for i, logtype in enumerate(self.unwrapped.top_logtypes):

                log_count = int(distribution[i] * num_logs)
                # log_count = int(distribution[i] * 0.10 * self.unwrapped._normalize_factor)
                
                # log_count = int(distribution[i] * 0.005 * self.unwrapped._normalize_factor)
                self.inserted_logs += log_count
                self.unwrapped.episodic_inserted_logs += log_count
                diversity = 0
                # if log_count > 0:
                is_trigger = 0
                
                if logtype in self.unwrapped.relevant_logtypes:
                    diversity = float(diversity_list[self.unwrapped.relevant_logtypes.index(logtype)])
                    if log_count == 0:
                        diversity = 0
                    # is_trigger = int(np.ceil(diversity/self.diversity_factor ))
                    is_trigger = int(np.ceil(diversity ))
                    
                    # diversity = int(diversity)
                    diversity = int(diversity * self.diversity_factor)
                diversity = max(1, min(diversity, log_count))
                    
                    
                key = f"{logtype[0]}_{logtype[1]}_{is_trigger}"
                # if self.unwrapped.time_manager.action_window.to_tuple() not in self.action_time_indexer:
                #      self.action_time_indexer[self.unwrapped.time_manager.action_window.to_tuple()] = self.unwrapped.step_counter
                # if (key,self.unwrapped.time_manager.action_window.to_tuple()) in self.fake_storage_state:
                #      diff = log_count - self.fake_storage_state[self.action_time_indexer[self.unwrapped.time_manager.action_window.to_tuple()]][self.log_type_indexer[key]]
                logs_to_inject[key] = {
                    # 'count': max(0, -diff),
                    'count': log_count,
                    'diversity': diversity
                }
                    #  self.logs_to_delete[self.action_time_indexer[self.unwrapped.time_manager.action_window.to_tuple()]][self.log_type_indexer[key]] = max(0, diff)
                    #  self.fake_storage_state[(key,self.unwrapped.time_manager.action_window.to_tuple())] -= diff
                
                # Track logs
                self.current_logs[key] = log_count

                self.episode_logs[key] += log_count
                self.diversity_episode_logs[key] = max(logs_to_inject[key]['diversity'], self.diversity_episode_logs[key])
                # if logtype in self.unwrapped.relevant_logtypes and is_trigger == 1:
                #     self.unwrapped.rules_rel_diff_alerts[logtype] =  self.diversity_episode_logs[key]/ (self.diversity_factor)
                self.unwrapped.episodic_fake_logs_qnt += log_count
                
            return logs_to_inject

    
        def reset(self, **kwargs):
            """Reset tracking on environment reset"""
            self.current_logs = {}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self.episodic_logs_to_inject = []
            self.logs_to_delete = {}
            # self.remaining_quota = self.quota
            self.info = kwargs["options"]
            self.unwrapped.episodic_inserted_logs = 0
            self.unwrapped.episodic_fake_logs_qnt = 0
            obs, info = self.env.reset(**kwargs)
            
            return obs, info

class SngleAction(Action):
        """relevant logs = top logtypes"""
        
        def __init__(self, env, test_random=False):
            super().__init__(env, test_random)
            self.action_space = spaces.Box(
                low=0,
                high=1,
                shape=(3,),
                dtype=np.float32
            )
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self.diversity_factor = 31
            self.unwrapped.episodic_inserted_logs = 0
            self.current_real_quantity = 0
            
        def action(self, action):
            """Convert raw action to log injection dictionary"""
            # Split action into quota and distribution
            log_index = action[0]
            diversity = action[1]
            num_logs = 0.5 * self.current_real_quantity * action[2]
            self.inserted_logs = 0
            
            # self.remaining_quota = self.quota - num_logs
            
            # Distribute logs among types
            logs_to_inject = {}


            log_count = int(num_logs)
            # log_count = int(distribution[i] * 0.10 * self.unwrapped._normalize_factor)
            
            # log_count = int(distribution[i] * 0.005 * self.unwrapped._normalize_factor)
            self.inserted_logs += log_count
            self.unwrapped.episodic_inserted_logs += log_count
            logtype = self.unwrapped.top_logtypes[int(log_index*len(self.unwrapped.top_logtypes))-1]
            if log_count > 0:
                if not logtype in self.unwrapped.relevant_logtypes:
                    diversity = 0
                is_trigger = int(np.ceil(diversity))
                key = f"{logtype[0]}_{logtype[1]}_{int(is_trigger)}"
                logs_to_inject[key] = {
                    'count': log_count,
                    'diversity': int(diversity * self.diversity_factor) + 1
                }
                
                # Track logs
                self.current_logs[key] = log_count


                self.episode_logs[key] += log_count
                self.diversity_episode_logs[key] = max(logs_to_inject[key]['diversity'], self.diversity_episode_logs[key])

            return logs_to_inject

    
        def reset(self, **kwargs):
            """Reset tracking on environment reset"""
            self.current_logs = {}
            # self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}

            # self.remaining_quota = self.quota
            self.info = kwargs["options"]
            self.unwrapped.episodic_inserted_logs = 0
            
            obs, info = self.env.reset(**kwargs)
            return obs, info
        
class Action9(Action):
        """relevant logs = top logtypes"""
        
        def __init__(self, env, test_random=False):
            super().__init__(env, test_random)
            self.action_space = spaces.Box(
                low=0,
                high=1,
                shape=(len(self.unwrapped.top_logtypes) + len(self.unwrapped.relevant_logtypes),),
                dtype=np.float32
            )
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self.diversity_factor = 31
            self.unwrapped.episodic_inserted_logs = 0
            
        def action(self, action):
            """Convert raw action to log injection dictionary"""
            # Split action into quota and distribution


            num_logs = self.remaining_quota
            self.inserted_logs = 0
            
            # self.remaining_quota = self.quota - num_logs
            
            # Distribute logs among types
            logs_to_inject = {}

            for i, logtype in enumerate(self.unwrapped.top_logtypes):


                subtypes = 1
                diversity = 0
                if logtype in self.unwrapped.relevant_logtypes:
                    subtypes = 2
                    
                for j in range(subtypes):
                    is_trigger = j
                    if subtypes == 2:
                        log_count = int(action[(1-is_trigger)*i - is_trigger*self.unwrapped.relevant_logtypes.index(logtype)] * 300)
                    else:
                        log_count = int(action[i] * 300)
                    self.inserted_logs += log_count
                    self.unwrapped.episodic_inserted_logs += log_count
                    key = f"{logtype[0]}_{logtype[1]}_{int(is_trigger)}"                    
                    logs_to_inject[key] = {
                        'count': log_count,
                        'diversity': int(diversity * self.diversity_factor) + 1
                    }
                    
                    # Track logs
                    self.current_logs[key] = log_count
    
    
                    self.episode_logs[key] += log_count
                    self.diversity_episode_logs[key] = max(logs_to_inject[key]['diversity'], self.diversity_episode_logs[key])
            return logs_to_inject

    
        def reset(self, **kwargs):
            """Reset tracking on environment reset"""
            self.current_logs = {}
            # self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}

            # self.remaining_quota = self.quota
            self.info = kwargs["options"]
            self.unwrapped.episodic_inserted_logs = 0
            
            obs, info = self.env.reset(**kwargs)
            return obs, info
        
class Action10(Action8):
    """relevant logs = top logtypes"""
    
    def __init__(self, env, test_random=False):
        super().__init__(env, test_random)
        self.action_space = spaces.Box(
            low=0.0000000001,
            high=1,
            shape=(len(self.unwrapped.top_logtypes) + len(self.unwrapped.relevant_logtypes)+1,),
            dtype=np.float32
        )

        
    def action(self, action):
        """Convert raw action to log injection dictionary"""
        # Split action into quota and distribution
        distribution = action[:len(self.unwrapped.top_logtypes)]
        # softmax normalization
        distribution = np.exp(distribution) / np.sum(np.exp(distribution))
        # distribution /= (np.sum(distribution) + 1e-8) 
        
        diversity_list = action[len(self.unwrapped.top_logtypes):-1]
        quota_pct = action[-1]
        num_logs = quota_pct * self.current_real_quantity
        self.inserted_logs = 0
        
        # self.remaining_quota = self.quota - num_logs
        
        # Distribute logs among types
        logs_to_inject = {}

        for i, logtype in enumerate(self.unwrapped.top_logtypes):

            log_count = int(distribution[i] * num_logs)
            # log_count = int(distribution[i] * 0.10 * self.unwrapped._normalize_factor)
            
            # log_count = int(distribution[i] * 0.005 * self.unwrapped._normalize_factor)
            self.inserted_logs += log_count
            self.unwrapped.episodic_inserted_logs += log_count

            if log_count > 0:
                diversity = 0
                if logtype in self.unwrapped.relevant_logtypes:
                    diversity = float(diversity_list[self.unwrapped.relevant_logtypes.index(logtype)])
                is_trigger = int(np.ceil(diversity))
                key = f"{logtype[0]}_{logtype[1]}_{int(is_trigger)}"
                logs_to_inject[key] = {
                    'count': log_count,
                    'diversity': int(diversity * self.diversity_factor) + 1
                }
                
                # Track logs
                self.current_logs[key] = log_count


                self.episode_logs[key] += log_count
                self.diversity_episode_logs[key] = max(logs_to_inject[key]['diversity'], self.diversity_episode_logs[key])

        return logs_to_inject
    
class Action11(Action):
        """relevant logs = top logtypes"""
        
        def __init__(self, env, test_random=False):
            super().__init__(env, test_random)
            self.action_space = spaces.Box(
                low=0,
                high=1,
                shape=(len(self.unwrapped.top_logtypes)+ len(self.unwrapped.relevant_logtypes),),
                dtype=np.float32
            )
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self.diversity_factor = 31
            self.unwrapped.episodic_inserted_logs = 0
            self.current_real_quantity = 0
            
        def action(self, action):
            """Convert raw action to log injection dictionary"""
            # Split action into quota and distribution
            distribution = action[:len(self.unwrapped.top_logtypes)]
            # softmax normalization
            distribution = np.exp(distribution) / np.sum(np.exp(distribution))
            # distribution /= (np.sum(distribution) + 1e-8) 
            
            diversity_list = action[len(self.unwrapped.top_logtypes):]
            num_logs =  5000
            # num_logs = self.unwrapped.config.additional_percentage * self.current_real_quantity
            self.inserted_logs = 0
            self.current_logs = {}
            # self.remaining_quota = self.quota - num_logs
            
            # Distribute logs among types
            logs_to_inject = {}

            for i, logtype in enumerate(self.unwrapped.top_logtypes):

                log_count = int(distribution[i] * num_logs)
                # log_count = int(distribution[i] * 0.10 * self.unwrapped._normalize_factor)
                
                # log_count = int(distribution[i] * 0.005 * self.unwrapped._normalize_factor)
                self.inserted_logs += log_count
                self.unwrapped.episodic_inserted_logs += log_count
                diversity = 0
                if log_count > 0:
                    if logtype in self.unwrapped.relevant_logtypes:
                        diversity = 0
                        diversity = float(diversity_list[self.unwrapped.relevant_logtypes.index(logtype)])
                        
                    is_trigger = int(np.ceil(diversity))
                    key = f"{logtype[0]}_{logtype[1]}_{int(is_trigger)}"
                    logs_to_inject[key] = {
                        'count': log_count,
                        'diversity': int(diversity * self.diversity_factor) + 1
                    }
                    
                    # Track logs
                    self.current_logs[key] = log_count
    
    
                    self.episode_logs[key] += log_count
                    self.diversity_episode_logs[key] = max(logs_to_inject[key]['diversity'], self.diversity_episode_logs[key])
                    if logtype in self.unwrapped.relevant_logtypes:
                        self.unwrapped.rules_rel_diff_alerts[logtype] = self.diversity_episode_logs[key]/ (self.diversity_factor+ 1)
            return logs_to_inject

    
        def reset(self, **kwargs):
            """Reset tracking on environment reset"""
            self.current_logs = {}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}

            # self.remaining_quota = self.quota
            self.info = kwargs["options"]
            self.unwrapped.episodic_inserted_logs = 0
            
            obs, info = self.env.reset(**kwargs)
            return obs, info   
# class RandomAction(Action8):
        
#     def __init__(self, env, test_random=False):
#         super().__init__(env, test_random)
#         # Store the original step method
#         self._original_step = env.step
    
#     def step(self, action):
#         """
#         Override step to use random actions instead of provided ones.
#         This works even if the wrapped environment has a custom step implementation.
#         """
#         # Generate random action
#         random_action = self.action_space.sample()
        
#         # Call the original step with our random action
#         return self._original_step(random_action)


    
# Usage example:
if __name__ == "__main__":
    env = make('Splunk-v0')
    
    # Define relevant log types
    relevant_logtypes = [
        ('wineventlog:security', '4624'),
        ('wineventlog:security', '4625'),
        ('wineventlog:system', '7040')
    ]
    
    # Wrap environment
    env = Action(env, relevant_logtypes, quota_per_step=1000)
    
    obs = env.reset()
    for _ in range(100):
        # Sample action: [quota_pct, type1_pct, type2_pct, type3_pct]
        action = env.action_space.sample()
        
        # Wrapper converts to: {logtype: count}
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get injection info
        injection_info = env.get_injection_info()
        print(f"Injected logs: {injection_info['current_logs']}")
        print(f"Quota remaining: {injection_info['remaining_quota']}")
        
        if terminated or truncated:
            obs = env.reset()