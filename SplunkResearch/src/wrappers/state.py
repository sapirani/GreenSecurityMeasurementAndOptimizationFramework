from gymnasium.core import ObservationWrapper
import numpy as np
from gymnasium import make, spaces
import logging

logger = logging.getLogger(__name__)

class StateWrapper(ObservationWrapper):
    """Manages log type distributions and state normalization"""
    
    def __init__(self, env, top_logtypes):
        super().__init__(env)
        top_logtypes = set(top_logtypes)|set(self.relevant_logtypes)
        
        self.top_logtypes = top_logtypes
        
        # Initialize distributions
        self.real_distribution = {logtype: 0 for logtype in top_logtypes}
        self.real_distribution['other'] = 0
        self.fake_distribution = {logtype: 0 for logtype in top_logtypes}
        self.fake_distribution['other'] = 0
        
        self.real_state = np.array([])
        self.fake_state = np.array([])
        
        # Define observation space for normalized distributions
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(self.top_logtypes)*2 + 2,),  # +2 for 'other' category
            dtype=np.float64
        )

    def get_step_info(self):
        """Get current step info"""
        return {
            'real_distribution': self.real_state,
            'fake_distribution': self.fake_state,

        }

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info.update(self.get_step_info())
        return obs, reward, done, truncated, info

    def observation(self, obs):
        """Convert current distributions to normalized state"""

        self.update_real_distribution(self.time_manager.action_window.to_tuple())
        
        self.update_fake_distribution(self.episode_logs)
        real_state = self._get_state_vector(self.real_distribution)
        fake_state = self._get_state_vector(self.fake_distribution)
        
        # Normalize states
        real_state = self._normalize(real_state)
        fake_state = self._normalize(fake_state)
        
        # Store for external use (e.g., reward calculation)
        self.real_state = real_state
        self.fake_state = fake_state
        
        # Combine states
        return np.concatenate([real_state, fake_state])

    def _get_state_vector(self, distribution):
        """Convert distribution dict to vector"""
        state = [distribution[logtype] for logtype in self.top_logtypes]
        state.append(distribution['other'])
        return np.array(state)

    def _normalize(self, state):
        """Normalize state vector"""
        total = np.sum(state)
        if total > 0:
            return state / total
        return np.ones_like(state) / len(state)

    def update_real_distribution(self, time_range):
        """Update real distribution from Splunk"""
        real_counts = self.env.splunk_tools.get_real_distribution(*time_range)

        
        for logtype, count in real_counts.items():
            if logtype in self.top_logtypes:
                self.real_distribution[logtype] += count
                
            else:
                self.real_distribution['other'] += count

    def update_fake_distribution(self, injected_logs):
        """Update fake distribution with injected logs"""
        # Add injected logs to existing real distribution
        self.fake_distribution = self.real_distribution.copy()
        for logtype, count in injected_logs.items():
            if logtype in self.top_logtypes:
                self.fake_distribution[logtype] += count
            else:
                self.fake_distribution['other'] += count

    def reset(self, *, seed=None, options=None):
        """Reset the wrapper state"""
        # Reset underlying environment first
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Reset distributions
        self.real_distribution = {logtype: 0 for logtype in self.top_logtypes}
        self.real_distribution['other'] = 0
        self.fake_distribution = {logtype: 0 for logtype in self.top_logtypes}
        self.fake_distribution['other'] = 0
        
        self.real_state = np.array([])
        self.fake_state = np.array([])
       
        new_obs = np.zeros(self.observation_space.shape)
        return new_obs, info

# Example usage:
if __name__ == "__main__":
    env = make('Splunk-v0')
    
    top_logtypes = [
        ('wineventlog:security', '4624'),
        ('wineventlog:security', '4625'),
        # ... other important log types
    ]
    
    env = StateWrapper(env, top_logtypes)
    
    obs = env.reset()
    
    # In your environment's step method:
    action = env.action_space.sample()
    injected_logs = {
        ('wineventlog:security', '4624'): 10,
        ('wineventlog:security', '4625'): 5
    }
    env.update_fake_distribution(injected_logs)  # Update fake distribution
    
    obs, reward, done, info = env.step(action)