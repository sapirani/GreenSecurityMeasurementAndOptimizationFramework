import logging
from gymnasium.core import ActionWrapper
from gymnasium import make, spaces
import numpy as np
logger = logging.getLogger(__name__)

class Action(ActionWrapper):
    """Wrapper for managing log injection actions"""
    
    def __init__(self, env):
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
        self.diversity_factor = 10
        self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.relevant_logtypes for istrigger in [0, 1]}
        self.info = {}
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
    
    def inject_logs(self, logs_to_inject):
        """Inject logs into environment"""
        time_range = self.env.time_manager.action_window.to_tuple()
        logger.info(f"Action time range: {time_range}")
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
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        logger.info(f"Raw action: {action}")
        self._calculate_quota()
        logs_to_inject = self.action(action)
        logger.info(f"Action: {logs_to_inject}")
        logger.info(f"Action window: {self.time_manager.action_window.to_tuple()}")
        self.inject_logs(logs_to_inject)
        
        
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
        
        obs, info = self.env.reset(**kwargs)
        return obs, info

class SingleAction2(Action):
    
    def __init__(self, env):
        super().__init__(env)

        low = np.array([0, 0, 0, 0])
        high = np.array([1, len(self.relevant_logtypes)-1, 1, 10])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
    def action(self, action):
        """Convert raw action to log injection dictionary"""
        # Split action into quota and distribution
        quota_pct = action[0]
        eventcode = round(action[1])
        trigger_level = round(action[2])
        diversity_level = action[3]
        

        
        # Calculate number of logs to inject
        num_logs = int(quota_pct * self.remaining_quota)
        self.inserted_logs = num_logs
        # self.remaining_quota = self.quota - num_logs
        
        # Distribute logs among types
        logs_to_inject = {}
        logtype = self.relevant_logtypes[eventcode]
        is_trigger = trigger_level
        log_count = int(num_logs) 
        key = f"{logtype[0]}_{logtype[1]}_{int(is_trigger)}"
        if log_count > 0:
            logs_to_inject[key] = {
                'count': log_count,
                'diversity': round(diversity_level)
            }
            
            # Track logs
            self.current_logs[key] = log_count


            self.episode_logs[key] += log_count
            self.diversity_episode_logs[key] = max(round(diversity_level), self.diversity_episode_logs[key])
    
        return logs_to_inject
    
class SingleAction(Action):
    
    def __init__(self, env):
        super().__init__(env)

        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(4,),
            dtype=np.float32
        )
    
    def action(self, action):
        """Convert raw action to log injection dictionary"""
        # Split action into quota and distribution
        quota_pct = action[0]
        eventcode = round(action[1]*(len(self.relevant_logtypes) - 1))
        trigger_level = action[2]
        diversity_level = action[3] * self.diversity_factor
        

        
        # Calculate number of logs to inject
        num_logs = int(quota_pct * self.remaining_quota)
        self.inserted_logs = num_logs
        # self.remaining_quota = self.quota - num_logs
        
        # Distribute logs among types
        logs_to_inject = {}
        logtype = self.relevant_logtypes[eventcode]
        for is_trigger in [False, True]:
            log_count = int(num_logs * (is_trigger * trigger_level + (1-is_trigger) * (1 - trigger_level))) 
            key = f"{logtype[0]}_{logtype[1]}_{int(is_trigger)}"
            if log_count > 0:
                logs_to_inject[key] = {
                    'count': log_count,
                    'diversity': round(diversity_level)
                }
                
                # Track logs
                self.current_logs[key] = log_count


                self.episode_logs[key] += log_count
                self.diversity_episode_logs[key] = max(round(diversity_level), self.diversity_episode_logs[key])
        
        return logs_to_inject
    
class Action2(Action):
    
    def __init__(self, env):
        super().__init__(
            env
        )
    
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
    
    def __init__(self, env):
        super().__init__(
            env
        )
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
        
        def __init__(self, env):
            super().__init__(
                env
            )
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
        
        def __init__(self, env):
            super().__init__(
                env
            )
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
        
        def __init__(self, env):
            super().__init__(
                env
            )
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
        
        def __init__(self, env):
            super().__init__(
                env
            )
            self.action_space = spaces.Box(
                low=0,
                high=1,
                shape=(len(self.top_logtypes),),
                dtype=np.float32
            )
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0]}

        def action(self, action):
            """Convert raw action to log injection dictionary"""
            # Split action into quota and distribution
            distribution = action

            
            # Normalize distribution
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
                log_count = int(distribution[i] * 0.005 * self.unwrapped._normalize_factor)
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
            # self.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
            self._calculate_quota()
            self.diversity_episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}

            # self.remaining_quota = self.quota
            self.info = kwargs["options"]
            
            obs, info = self.env.reset(**kwargs)
            return obs, info
class RandomAction(ActionWrapper):
        
    def __init__(self, env):
        super().__init__(env)
        # Store the original step method
        self._original_step = env.step
    
    def step(self, action):
        """
        Override step to use random actions instead of provided ones.
        This works even if the wrapped environment has a custom step implementation.
        """
        # Generate random action
        random_action = self.action_space.sample()
        
        # Call the original step with our random action
        return self._original_step(random_action)

    
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