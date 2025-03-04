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
        logger.info(f"Raw action: {action}")
        self._calculate_quota()
        logs_to_inject = self.action(action)
        self.inject_logs(logs_to_inject)
        
        obs, reward, terminated, truncated, info = self.env.step(logs_to_inject)

        
        
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
        
        return super().reset(**kwargs)

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