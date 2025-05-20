import datetime
from gymnasium.core import ObservationWrapper
import numpy as np
from gymnasium import make, spaces
import logging
from gymnasium.core import ActionWrapper

logger = logging.getLogger(__name__)

class StateWrapper(ObservationWrapper):
    """Manages log type distributions and state normalization"""
    
    def __init__(self, env):
        super().__init__(env)

        self.action_wrapper = self.get_wrapper(ActionWrapper)
        
        # Initialize distributions

        

        self.total_current_logs = 0
        self.total_episode_logs = 0
        # self._normalize_factor = 500000
        # Define observation space for normalized distributions
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(self.top_logtypes)*2 + self.env.total_steps,),  # +1 for 'other' category
            dtype=np.float64
        )

      
    def get_wrapper(self, wrapper_class):
        """
        Utility function to retrieve a specific wrapper instance from the wrapper stack.
        """
        env = self.env
        while env:
            if isinstance(env, wrapper_class):
                return env
            env = getattr(env, 'env', None)
        raise ValueError(f"Wrapper {wrapper_class} not found in the wrapper stack.")

    def get_step_info(self):
        """Get current step info"""
        return {
            'real_distribution': self.real_state,
            # 'fake_distribution': self.fake_state,
            'total_current_logs': self.total_current_logs,
            'real_relevant_distribution': self.real_relevant_distribution,
            'total_episode_logs': self.total_episode_logs,
        }

    def step(self, action):
        # Execute action in underlying environment
        observation, reward, terminated, truncated, info = self.env.step(action)
        info.update(self.get_step_info())
        obs = self.observation(observation)
        info.update(self.get_step_info())            
        return obs, reward, terminated, truncated, info

    def observation(self, obs):
        """Convert current distributions to normalized state"""
        # Calculate fake distribution using the latest episode_logs
        # This happens AFTER action wrapper has updated episode_logs
        # Update real distribution AFTER action is executed
        self.update_real_distribution(self.time_manager.action_window.to_tuple())

        # Create state vectors
        real_state = self._get_state_vector(self.unwrapped.real_distribution)
        self.unwrapped.real_state = self._normalize(real_state)
        self.unwrapped.real_relevant_distribution = {"_".join(logtype): self.real_state[self.relevant_logtypes_indices[logtype]] for logtype in self.top_logtypes}

        fake_state = self._get_state_vector(self.fake_distribution)
        self.unwrapped.fake_state = self._normalize(fake_state)
        self.unwrapped.fake_relevant_distribution = {"_".join(logtype): self.fake_state[self.relevant_logtypes_indices[logtype]] for logtype in self.top_logtypes}
        # Create the final state vector
        state = np.append(self.real_state, self.fake_state)
        # state = np.append(state, min(1, self.total_episode_logs/self._normalize_factor))
        # fake_total_logs = self.total_episode_logs + sum(self.episode_logs.values())
        # state = np.append(state, min(1, fake_total_logs/self._normalize_factor))
        # append to state the step index
        # state = np.append(state, self.env.step_counter/self.env.total_steps)
        # add sparse vector for step index
        sparse_vector = np.zeros(self.env.total_steps)
        sparse_vector[self.unwrapped.step_counter] = 1
        state = np.append(state, sparse_vector)
        # add sparse vector for weekday and hour

        logger.info(f"State: {state}")
        self.unwrapped.obs = state
        return state

    # Other methods remain the same...
    def _get_state_vector(self, distribution):
        """Convert distribution dict to vector"""
        state = [distribution[logtype] for logtype in self.top_logtypes]
        # state.append(distribution['other'])
        return np.array(state)

    def _normalize(self, state):
        """Normalize state vector"""
        total = np.sum(state)
        if total > 0:
            return state / total
        return np.ones_like(state) / len(state)
    
    def update_fake_distribution_from_real(self):
        """
        Update the fake distribution based on the real distribution.
        This is a placeholder for the actual logic to update the fake distribution.
        """
        self.unwrapped.fake_distribution = {logtype: 0 for logtype in self.top_logtypes}
        self.unwrapped.fake_distribution['other'] = 0
        for logtype in self.top_logtypes:
            self.unwrapped.ac_fake_distribution[logtype] += self.unwrapped.real_distribution[logtype]
            self.unwrapped.fake_distribution[logtype] = self.unwrapped.real_distribution[logtype]
            
    def update_real_distribution(self, time_range):
        """Update real distribution from Splunk"""
        real_counts = self.env.splunk_tools.get_real_distribution(*time_range)
        self.unwrapped.real_distribution = {logtype: 0 for logtype in self.top_logtypes}
        self.unwrapped.real_distribution['other'] = 0
        self.total_current_logs = 0
        for logtype, count in real_counts.items():
            if logtype in self.top_logtypes:
                self.unwrapped.real_distribution[logtype] = count
                self.unwrapped.ac_real_distribution[logtype] += count
                self.total_current_logs += count 
                
            # else:
            #     self.unwrapped.real_distribution['other'] += count
            #     self.unwrapped.ac_real_distribution['other'] += count
                # self.total_current_logs += count
        self.total_episode_logs += self.total_current_logs
        self.action_wrapper.current_real_quantity = self.total_current_logs

    
    def reset(self, *, seed=None, options=None):
        """Reset the wrapper state"""
        logger.info("Resetting StateWrapper")
        # Reset underlying environment first
        self.total_episode_logs = 0
        self.unwrapped.done = False
        self.unwrapped.ac_real_distribution = {logtype: 0 for logtype in self.top_logtypes}
        self.unwrapped.ac_real_distribution['other'] = 0
        self.unwrapped.real_relevant_distribution = {"_".join(logtype): 0 for logtype in self.top_logtypes}
        self.unwrapped.step_counter = 0
        self.env.time_manager.advance_window(global_step=self.unwrapped.all_steps_counter, violation=False)
        self.unwrapped.fake_distribution = {logtype: 0 for logtype in self.top_logtypes}
        self.unwrapped.fake_distribution['other'] = 0
        self.fake_relevant_distribution = {"_".join(logtype): 0 for logtype in self.top_logtypes}

        self.unwrapped.ac_fake_distribution = {logtype: 0 for logtype in self.top_logtypes}
        self.unwrapped.ac_fake_distribution['other'] = 0
        # reset episode logs which are placed at lower wrapper (action)
        # self.action_wrapper.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.top_logtypes for istrigger in [0, 1]}
        
    
        new_obs = self.observation(None)
        # new_obs = np.append(self.ac_real_state, self.ac_real_state)
        options = self.get_step_info()
        obs, info = self.env.reset(seed=seed, options=options)
        info.update(self.get_step_info())
        
        return new_obs, info

class StateWrapper3(StateWrapper):
    """Manages log type distributions and state normalization"""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(len(self.top_logtypes)*2,),  # +1 for 'other' category
            # shape=(len(self.top_logtypes),),  # +1 for 'other' category
            dtype=np.float64
        )
        
        
    def observation(self, obs):
        """Convert current distributions to normalized state"""
        # Calculate fake distribution using the latest episode_logs
        # This happens AFTER action wrapper has updated episode_logs
        # Update real distribution AFTER action is executed
        if not self.unwrapped.done:
            self.update_real_distribution(self.time_manager.action_window.to_tuple())

        # Create state vectors
        real_state = self._get_state_vector(self.unwrapped.real_distribution)
        self.unwrapped.real_state = self._normalize(real_state)
        ac_real_state = self._get_state_vector(self.unwrapped.ac_real_distribution)
        self.unwrapped.ac_real_state = self._normalize(ac_real_state)
        self.unwrapped.real_relevant_distribution = {"_".join(logtype): self.unwrapped.ac_real_state[self.relevant_logtypes_indices[logtype]] for logtype in self.top_logtypes}
        if not self.unwrapped.done:
            self.update_fake_distribution_from_real()
        self.unwrapped.fake_state = self._get_state_vector(self.unwrapped.fake_distribution)
        self.unwrapped.fake_state = self._normalize(self.fake_state)
        ac_fake_state = self._get_state_vector(self.unwrapped.ac_fake_distribution)
        self.unwrapped.ac_fake_state = self._normalize(ac_fake_state)
        self.unwrapped.fake_relevant_distribution = {"_".join(logtype): self.unwrapped.ac_fake_state[self.relevant_logtypes_indices[logtype]] for logtype in self.top_logtypes}

        # Create the final state vector
        # state = self.unwrapped.real_state
        state = np.append(self.unwrapped.ac_real_state, self.unwrapped.ac_fake_state)
        # sparse_vector = np.zeros(self.env.total_steps)
        # sparse_vector[self.unwrapped.step_counter] = 1
        # state = np.append(state, sparse_vector)
        # state = np.append(state, self.current_real_quantity/100000) 
        # fake_total_logs = self.total_episode_logs + sum(self.action_wrapper.episode_logs.values())
        # state = np.append(state, fake_total_logs/500000)
        # # append to state the step index
        # # state = np.append(state, self.env.step_counter/self.env.total_steps)
        # # add sparse vector for step index

        
        # current_datetime = datetime.datetime.strptime(self.env.time_manager.action_window.end, '%m/%d/%Y:%H:%M:%S')
        # weekday_vector = np.zeros(7)
        # weekday_vector[current_datetime.weekday()] = 1
        # hour_vector = np.zeros(24)
        # hour_vector[current_datetime.hour] = 1
        # state = np.append(state, weekday_vector)
        # state = np.append(state, hour_vector)
        logger.info(f"State: {state}")
        self.unwrapped.obs = state
        return state
    
    def _normalize(self, state):
        """Normalize state vector"""
        return state / (sum(state) + 0.000000001)  # Avoid division by zero
     

     
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