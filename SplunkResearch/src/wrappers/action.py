import logging
from gymnasium.core import ActionWrapper
from gymnasium import make, spaces
import numpy as np
logger = logging.getLogger(__name__)

class Action(ActionWrapper):
    """Wrapper for managing log injection actions"""
    
    def __init__(self, env, relevant_logtypes, quota_per_step):
        super().__init__(env)
        self.relevant_logtypes = relevant_logtypes
        self.quota = quota_per_step
        
        # Create action space:
        # First value is quota percentage (0-1)
        # then values are distribution across log types
        # then values are triggering levels for each log type
        # then values are diversity levels for each log type
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(1 + len(relevant_logtypes)*3,),
            dtype=np.float32
        )
        
        # Track injected logs
        self.current_logs = {}
        self.episode_logs = {}
        self.remaining_quota = quota_per_step

    def action(self, action):
        """Convert raw action to log injection dictionary"""
        # Split action into quota and distribution
        quota_pct = action[0]
        distribution = action[1:1+len(self.relevant_logtypes)]
        trigger_levels = action[1+len(self.relevant_logtypes):2*len(self.relevant_logtypes)]
        diversity_levels = action[2*len(self.relevant_logtypes):]
        
        # Normalize distribution
        distribution = distribution / (np.sum(distribution) + 1e-8)
        
        # Calculate number of logs to inject
        num_logs = int(quota_pct * self.quota)
        self.remaining_quota = self.quota - num_logs
        
        # Distribute logs among types
        logs_to_inject = {}
        for i, logtype in enumerate(self.relevant_logtypes):
            for is_trigger in [False, True]:
                log_count = int(distribution[i] * num_logs * (is_trigger * trigger_levels[i] + (1-is_trigger) * (1 - trigger_levels[i]))) 
                if log_count > 0:
                    logs_to_inject[f"{logtype[0]}:{logtype[1]}:{int(is_trigger)}"] = {
                        'count': log_count,
                        'diversity': diversity_levels[i]
                    }
                    
                    # Track logs
                    self.current_logs[f"{logtype[0]}:{logtype[1]}:{int(is_trigger)}"] = log_count

                    if f"{logtype[0]}:{logtype[1]}:{int(is_trigger)}" not in self.episode_logs:
                        self.episode_logs[f"{logtype[0]}:{logtype[1]}:{int(is_trigger)}"] = 0
                    self.episode_logs[f"{logtype[0]}:{logtype[1]}:{int(is_trigger)}"] += log_count
        
        return logs_to_inject
    
    def inject_logs(self, logs_to_inject):
        """Inject logs into environment"""
        for logtype, log_info in logs_to_inject.items():
            logsource, eventcode, is_trigger = logtype.split(':')
            count, diversity = log_info['count'], log_info['diversity']
            time_range = self.env.time_manager.action_window
            
            fake_logs = self.env.log_generator.generate_logs(
                logsource, eventcode, is_trigger,
                time_range, count, diversity
                )
            self.env.splunk_tools_instance.write_logs_to_monitor(fake_logs, logsource)
            logger.info(
                f"inserted {len(fake_logs)} logs of type {logsource} "
                f"{eventcode} {is_trigger} with diversity {diversity}"
            )
        

    def get_injection_info(self):
        """Get information about current injections"""
        return {
            'current_logs': self.current_logs,
            'episode_logs': self.episode_logs,
            'remaining_quota': self.remaining_quota,
            'quota_used_pct': (self.quota - self.remaining_quota) / self.quota
        }

    def reset(self, **kwargs):
        """Reset tracking on environment reset"""
        self.current_logs = {}
        self.episode_logs = {}
        self.remaining_quota = self.quota
        return super().reset(**kwargs)

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