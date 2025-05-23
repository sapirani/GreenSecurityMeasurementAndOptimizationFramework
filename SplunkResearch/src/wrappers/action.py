import logging
from gymnasium.core import ActionWrapper
from gymnasium import make, spaces
import numpy as np
logger = logging.getLogger(__name__)

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
            shape=(1 + len(self.relevant_logtypes)*3,),
            dtype=np.float32
        )
        
        # Track injected logs
        self.current_logs = {}
        self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.relevant_logtypes for istrigger in [0, 1]}
        self.remaining_quota = 0
        self.inserted_logs = 0
        self.diversity_factor = 31
        self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.relevant_logtypes for istrigger in [0, 1]}
        self.info = {}
        self._disable_injection = False
        self.test_random = test_random
    # def _calculate_quota(self) -> None:
    #     """Calculate injection quotas"""
    #     self.total_additional_logs = (self.config.additional_percentage * 
    #                                 self.config.search_window * 
    #                                 self.config.logs_per_minute)
        
    #     self.step_size = int((self.total_additional_logs // self.config.search_window) * 
    #                         self.config.action_duration // 60)
    #     self.remaining_quota = self.step_size
    

    def _calculate_quota(self) -> None:
        """Calculate injection quotas"""

        self.step_size = int((self.time_manager.step_size//3600) * 2000 * self.config.additional_percentage)
        self.remaining_quota = self.step_size
    
            
    def update_fake_distribution(self): 
        """Update fake distribution with injected logs"""
        # Add injected logs to existing real distribution
        # self.fake_distribution = self.real_distribution.copy()#BUG!!

        for logtype, count in self.current_logs.items():
            formated_logtype = (*logtype.split('_')[:-1],)
            if formated_logtype in self.top_logtypes:
                self.unwrapped.fake_distribution[formated_logtype] += count
                self.unwrapped.ac_fake_distribution[formated_logtype] += count
                
            # else:
            #     self.fake_distribution['other'] += count

            # else:
            #     self.ac_fake_distribution['other'] += count
            
        # Normalize fake distribution
        self.unwrapped.fake_state = np.array([self.fake_distribution[k]/(sum(self.fake_distribution.values())+1e-8) for k in self.top_logtypes if k != 'other'])
        self.unwrapped.ac_fake_state = np.array([self.ac_fake_distribution[k]/(sum(self.ac_fake_distribution.values())+1e-8) for k in self.top_logtypes if k != 'other'])
        self.unwrapped.fake_relevant_distribution = {"_".join(logtype): self.fake_state[self.relevant_logtypes_indices[logtype]] for logtype in self.top_logtypes}
    
    def action(self, action):
        """Convert raw action to log injection dictionary"""
        # Split action into quota and distribution
        quota_pct = action[0]
        distribution = action[1:1+len(self.relevant_logtypes)]
        trigger_levels = action[1+len(self.relevant_logtypes):2*len(self.relevant_logtypes)+1]
        diversity_levels = action[2*len(self.relevant_logtypes)+1:] * self.diversity_factor
        
        # Normalize distribution
        distribution = distribution / (np.sum(distribution) + 1e-8)
        
        # Calculate number of logs to inject
        num_logs = int(quota_pct * self.remaining_quota)
        self.inserted_logs = num_logs
        # self.remaining_quota = self.quota - num_logs
        
        # Distribute logs among types
        logs_to_inject = {}
        for i, logtype in enumerate(self.relevant_logtypes):
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
        
    def inject_logs(self, logs_to_inject):
        """Inject logs into environment"""
        time_range = self.env.time_manager.action_window.to_tuple()
        logger.info(f"Action time range: {time_range}")
        if self._disable_injection:
                logger.info("Log injection disabled, not injecting logs")
                return
        for logtype, log_info in logs_to_inject.items():
            logsource, eventcode, is_trigger = logtype.split('_')
            count, diversity = log_info['count'], log_info['diversity']
            fake_logs = self.env.log_generator.generate_logs(
                logsource, eventcode, is_trigger,
                time_range, count, diversity
                )

            self.splunk_tools.write_logs_to_monitor(fake_logs, logsource)
            logger.info(
                f"inserted {len(fake_logs)} logs of type {logsource} "
                f"{eventcode} {is_trigger} with diversity {diversity}"
            )
        
    def step(self, action):
        """Inject logs and step environment"""
        if self.test_random:
            action = self.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        logger.info(f"Raw action: {action}")
        self._calculate_quota()
        logs_to_inject = self.action(action)
        logger.info(f"Action: {logs_to_inject}")
        logger.info(f"Action window: {self.time_manager.action_window.to_tuple()}")
        self.inject_logs(logs_to_inject)
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
            'episodic_inserted_logs': self.episodic_inserted_logs,
            'fake_relevant_distribution': self.unwrapped.fake_relevant_distribution,
            
            
            # 'quota_used_pct': (self.quota - self.remaining_quota) / self.quota
        }

    def reset(self, **kwargs):
        """Reset tracking on environment reset"""
        self.current_logs = {}
        self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.relevant_logtypes for istrigger in [0, 1]}
        self._calculate_quota()
        self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.relevant_logtypes for istrigger in [0, 1]}

        # self.remaining_quota = self.quota
        self.info = kwargs["options"]
        self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
        
        obs, info = self.env.reset(**kwargs)
        return obs, info


class Action2(Action):
    
    def __init__(self, env, test_random=False):
        super().__init__(env, test_random)
    
    def action(self, action):
        """Convert raw action to log injection dictionary"""
        # Split action into quota and distribution
        quota_pct = action[0]
        distribution = action[1:1+len(self.relevant_logtypes)]
        trigger_levels = action[1+len(self.relevant_logtypes):2*len(self.relevant_logtypes)+1]
        diversity_levels = action[2*len(self.relevant_logtypes)+1:] * self.diversity_factor
        
        # Normalize distribution
        distribution = distribution / (np.sum(distribution) + 1e-8)
        
        # Calculate number of logs to inject
        num_logs = int(quota_pct * self.remaining_quota)
        self.inserted_logs = num_logs
        # self.remaining_quota = self.quota - num_logs
        
        # Distribute logs among types
        logs_to_inject = {}
        for i, logtype in enumerate(self.relevant_logtypes):
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
            shape=(1 + len(self.relevant_logtypes)*2,),
            dtype=np.float32
        )
    
    def action(self, action):
        """Convert raw action to log injection dictionary"""
        # Split action into quota and distribution
        quota_pct = action[0]
        distribution = action[1:1+len(self.relevant_logtypes)]
        trigger_levels = action[1+len(self.relevant_logtypes):2*len(self.relevant_logtypes)+1]* self.diversity_factor

        # Normalize distribution
        distribution = distribution / (np.sum(distribution) + 1e-8)
        
        # Calculate number of logs to inject
        num_logs = int(quota_pct * self.remaining_quota)
        self.inserted_logs = num_logs
        # self.remaining_quota = self.quota - num_logs
        
        # Distribute logs among types
        logs_to_inject = {}
        for i, logtype in enumerate(self.relevant_logtypes):
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
                shape=(1 + len(self.relevant_logtypes),),
                dtype=np.float32
            )
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}

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

            for i, logtype in enumerate(self.env.top_logtypes):
                trigger_level_i = 0
                is_trigger = min(trigger_level_i, 1)
                if logtype in self.relevant_logtypes:
                    index = self.relevant_logtypes.index(logtype)
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
            real_distribution = self.obs[:len(self.env.top_logtypes)]
            fake_distribution = self.obs[len(self.env.top_logtypes):2*len(self.env.top_logtypes)]
            delta_distribution = real_distribution - fake_distribution
            delta_distribution = np.clip(delta_distribution, 0, 1)
            # take top 5 deltas
            delta_distribution_indices = np.argsort(delta_distribution)[-5:]
            top_5_normalized_delta_distribution = delta_distribution[delta_distribution_indices] / (np.sum(delta_distribution[delta_distribution_indices]) + 1e-8)
            allocation = []
            for i in range(len(self.env.top_logtypes)):
                if i in delta_distribution_indices:
                    allocation.append(top_5_normalized_delta_distribution[np.where(delta_distribution_indices == i)[0][0]] * quota)
                else:
                    allocation.append(0)

            return allocation

        def reset(self, **kwargs):
            """Reset tracking on environment reset"""
            self.current_logs = {}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}

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
                shape=(len(self.top_logtypes),),
                dtype=np.float32
            )
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}

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
            real_distribution = self.obs[:len(self.env.top_logtypes)]
            action = (distribution * (current_real_quantity + self.remaining_quota) - real_distribution * current_real_quantity)
            action = [max(0, a) for a in action]
            for i, logtype in enumerate(self.top_logtypes):
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
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}

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
                shape=(len(self.top_logtypes),),
                dtype=np.float32
            )
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}

        def action(self, action):
            """Convert raw action to log injection dictionary"""
            # Split action into quota and distribution
            distribution = action

            
            # Normalize distribution
            distribution = distribution / (np.sum(distribution) + 1e-8) #TODO try softmax
            current_real_quantity = self.info['total_episode_logs']
            
            # Calculate number of logs to inject
            # num_logs = self.config.additional_percentage * current_real_quantity
            num_logs = self.remaining_quota
            self.inserted_logs = num_logs
            # self.remaining_quota = self.quota - num_logs
            
            # Distribute logs among types
            logs_to_inject = {}

            for i, logtype in enumerate(self.env.top_logtypes):
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
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}

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
                shape=(len(self.top_logtypes),),
                dtype=np.float32
            )
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0]}


        def action(self, action):
            """Convert raw action to log injection dictionary"""
            # Split action into quota and distribution
            distribution = action
            
            distribution /= (np.sum(distribution) + 1e-8) #TODO try softmax

            self.inserted_logs = 0
            
            # self.remaining_quota = self.quota - num_logs
            
            # Distribute logs among types
            logs_to_inject = {}

            for i, logtype in enumerate(self.env.top_logtypes):

                log_count = int(distribution[i] * 3000)
                # log_count = int(distribution[i] * 0.10 * self.unwrapped._normalize_factor)
                
                # log_count = int(distribution[i] * 0.005 * self.unwrapped._normalize_factor)
                self.inserted_logs += log_count
                self.episodic_inserted_logs += log_count
 
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
            # self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}

            # self.remaining_quota = self.quota
            self.info = kwargs["options"]
            self.episodic_inserted_logs = 0
            
            obs, info = self.env.reset(**kwargs)
            return obs, info
        
class Action8(Action):
        """relevant logs = top logtypes"""
        
        def __init__(self, env, test_random=False):
            super().__init__(env, test_random)
            self.action_space = spaces.Box(
                low=0,
                high=1,
                shape=(len(self.top_logtypes)+ len(self.relevant_logtypes),),
                dtype=np.float32
            )
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self.diversity_factor = 31
            self.episodic_inserted_logs = 0
            self.current_real_quantity = 0
            
        def action(self, action):
            """Convert raw action to log injection dictionary"""
            # Split action into quota and distribution
            distribution = action[:len(self.top_logtypes)]
            # softmax normalization
            # distribution = np.exp(distribution) / np.sum(np.exp(distribution))
            distribution /= (np.sum(distribution) + 1e-8) 
            
            diversity_list = action[len(self.top_logtypes):]
            num_logs = self.config.additional_percentage * self.current_real_quantity
            self.inserted_logs = 0
            self.current_logs = {}
            # self.remaining_quota = self.quota - num_logs
            
            # Distribute logs among types
            logs_to_inject = {}

            for i, logtype in enumerate(self.env.top_logtypes):

                log_count = int(distribution[i] * num_logs)
                # log_count = int(distribution[i] * 0.10 * self.unwrapped._normalize_factor)
                
                # log_count = int(distribution[i] * 0.005 * self.unwrapped._normalize_factor)
                self.inserted_logs += log_count
                self.episodic_inserted_logs += log_count
                diversity = 0
                if log_count > 0:
                    if logtype in self.relevant_logtypes:
                        diversity = float(diversity_list[self.relevant_logtypes.index(logtype)])
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
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}

            # self.remaining_quota = self.quota
            self.info = kwargs["options"]
            self.episodic_inserted_logs = 0
            
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
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self.diversity_factor = 31
            self.episodic_inserted_logs = 0
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
            self.episodic_inserted_logs += log_count
            logtype = self.env.top_logtypes[int(log_index*len(self.env.top_logtypes))-1]
            if log_count > 0:
                if not logtype in self.relevant_logtypes:
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
            # self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}

            # self.remaining_quota = self.quota
            self.info = kwargs["options"]
            self.episodic_inserted_logs = 0
            
            obs, info = self.env.reset(**kwargs)
            return obs, info
        
class Action9(Action):
        """relevant logs = top logtypes"""
        
        def __init__(self, env, test_random=False):
            super().__init__(env, test_random)
            self.action_space = spaces.Box(
                low=0,
                high=1,
                shape=(len(self.top_logtypes) + len(self.relevant_logtypes),),
                dtype=np.float32
            )
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self.diversity_factor = 31
            self.episodic_inserted_logs = 0
            
        def action(self, action):
            """Convert raw action to log injection dictionary"""
            # Split action into quota and distribution


            num_logs = self.remaining_quota
            self.inserted_logs = 0
            
            # self.remaining_quota = self.quota - num_logs
            
            # Distribute logs among types
            logs_to_inject = {}

            for i, logtype in enumerate(self.env.top_logtypes):


                subtypes = 1
                diversity = 0
                if logtype in self.relevant_logtypes:
                    subtypes = 2
                    
                for j in range(subtypes):
                    is_trigger = j
                    if subtypes == 2:
                        log_count = int(action[(1-is_trigger)*i - is_trigger*self.relevant_logtypes.index(logtype)] * 300)
                    else:
                        log_count = int(action[i] * 300)
                    self.inserted_logs += log_count
                    self.episodic_inserted_logs += log_count
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
            # self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}

            # self.remaining_quota = self.quota
            self.info = kwargs["options"]
            self.episodic_inserted_logs = 0
            
            obs, info = self.env.reset(**kwargs)
            return obs, info
        
class Action10(Action8):
    """relevant logs = top logtypes"""
    
    def __init__(self, env, test_random=False):
        super().__init__(env, test_random)
        self.action_space = spaces.Box(
            low=0.0000000001,
            high=1,
            shape=(len(self.top_logtypes) + len(self.relevant_logtypes)+1,),
            dtype=np.float32
        )

        
    def action(self, action):
        """Convert raw action to log injection dictionary"""
        # Split action into quota and distribution
        distribution = action[:len(self.top_logtypes)]
        # softmax normalization
        distribution = np.exp(distribution) / np.sum(np.exp(distribution))
        # distribution /= (np.sum(distribution) + 1e-8) 
        
        diversity_list = action[len(self.top_logtypes):-1]
        quota_pct = action[-1]
        num_logs = quota_pct * self.current_real_quantity
        self.inserted_logs = 0
        
        # self.remaining_quota = self.quota - num_logs
        
        # Distribute logs among types
        logs_to_inject = {}

        for i, logtype in enumerate(self.env.top_logtypes):

            log_count = int(distribution[i] * num_logs)
            # log_count = int(distribution[i] * 0.10 * self.unwrapped._normalize_factor)
            
            # log_count = int(distribution[i] * 0.005 * self.unwrapped._normalize_factor)
            self.inserted_logs += log_count
            self.episodic_inserted_logs += log_count

            if log_count > 0:
                diversity = 0
                if logtype in self.relevant_logtypes:
                    diversity = float(diversity_list[self.relevant_logtypes.index(logtype)])
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