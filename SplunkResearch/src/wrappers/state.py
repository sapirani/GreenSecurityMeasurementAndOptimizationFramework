import asyncio
import datetime
from importlib.metadata import distribution
from math import dist
import os
import pickle
from gymnasium.core import ObservationWrapper
import joblib
import numpy as np
from gymnasium import make, spaces
import logging
from gymnasium.core import ActionWrapper
import pandas as pd
from torch import normal

logger = logging.getLogger(__name__)
ALERT_NORMALIZE_FACTOR = 100  # based on max alerts observed in training data
# ignore warnings
logging.getLogger('sklearn').setLevel(logging.ERROR)

class StateWrapper(ObservationWrapper):
    """Manages log type distributions and state normalization"""
    
    def __init__(self, env, is_sampled):
        super().__init__(env)

        self.action_wrapper = self.get_wrapper(ActionWrapper)
        self.normal_alert_predictors = {}
        # for rule in self.unwrapped.savedsearches:
        #     self.normal_alert_predictors[rule] = joblib.load(f"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/models_{rule}_alerts.joblib")


        self.total_current_logs = 0
        self.total_episode_logs = 0
        # self._normalize_factor = 500000
        # Define observation space for normalized distributions
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(self.unwrapped.top_logtypes)*2 + self.unwrapped.total_steps,),  # +1 for 'other' category
            dtype=np.float64
        )
        self.is_sampled = is_sampled

      
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
            'real_distribution': self.unwrapped.real_state,
            # 'fake_distribution': self.fake_state,
            'total_current_logs': self.total_current_logs,
            'real_relevant_distribution': self.unwrapped.real_relevant_distribution,
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
        self.update_real_distribution(self.unwrapped.time_manager.action_window.to_tuple())

        # Create state vectors
        real_state = self._get_state_vector(self.unwrapped.real_distribution)
        self.unwrapped.real_state = self._normalize(real_state)
        self.unwrapped.real_relevant_distribution = {"_".join(logtype): self.real_state[self.unwrapped.relevant_logtypes_indices[logtype]] for logtype in self.unwrapped.top_logtypes}

        fake_state = self._get_state_vector(self.fake_distribution)
        self.unwrapped.fake_state = self._normalize(fake_state)
        self.unwrapped.fake_relevant_distribution = {"_".join(logtype): self.fake_state[self.unwrapped.relevant_logtypes_indices[logtype]] for logtype in self.unwrapped.top_logtypes}
        # Create the final state vector
        state = np.append(self.real_state, self.fake_state)
        # state = np.append(state, min(1, self.total_episode_logs/self._normalize_factor))
        # fake_total_logs = self.total_episode_logs + sum(self.episode_logs.values())
        # state = np.append(state, min(1, fake_total_logs/self._normalize_factor))
        # append to state the step index
        # state = np.append(state, self.env.step_counter/self.unwrapped.total_steps)
        # add sparse vector for step index
        sparse_vector = np.zeros(self.unwrapped.total_steps)
        sparse_vector[self.unwrapped.step_counter] = 1
        state = np.append(state, sparse_vector)
        # add sparse vector for weekday and hour

        logger.info(f"State: {state}")
        self.unwrapped.obs = state
        return state

    # Other methods remain the same...
    def _get_state_vector(self, distribution):
        """Convert distribution dict to vector"""
        state = [distribution[logtype] for logtype in self.unwrapped.top_logtypes]
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
        self.unwrapped.fake_distribution = {logtype: 0 for logtype in self.unwrapped.top_logtypes}
        self.unwrapped.fake_distribution['other'] = 0
        for logtype in self.unwrapped.top_logtypes:
            self.unwrapped.ac_fake_distribution[logtype] += self.unwrapped.real_distribution[logtype]
            self.unwrapped.fake_distribution[logtype] = self.unwrapped.real_distribution[logtype]
            
    def update_real_distribution(self, time_range):
        """Update real distribution from Splunk"""
        real_counts = self.unwrapped.splunk_tools.get_real_distribution(*time_range, is_sampled=self.is_sampled)
        self.unwrapped.real_distribution = {logtype: 0 for logtype in self.unwrapped.top_logtypes}
        self.unwrapped.real_distribution['other'] = 0
        self.total_current_logs = 0
        for logtype, count in real_counts.items():
            if logtype in self.unwrapped.top_logtypes:
                self.unwrapped.real_distribution[logtype] = count
                self.unwrapped.ac_real_distribution[logtype] += count
                self.total_current_logs += count 
                
            # else:
            #     self.unwrapped.real_distribution['other'] += count
            #     self.unwrapped.ac_real_distribution['other'] += count
                # self.total_current_logs += count
        if self.is_sampled:
            self.total_current_logs *= 10
        self.total_episode_logs += self.total_current_logs
        self.action_wrapper.current_real_quantity = self.total_current_logs

    
    def reset(self, *, seed=None, options=None):
        """Reset the wrapper state"""
        logger.info("Resetting StateWrapper")
        # Reset underlying environment first
        self.total_episode_logs = 0
        self.unwrapped.done = False
        self.unwrapped.ac_real_distribution = {logtype: 0 for logtype in self.unwrapped.top_logtypes}
        self.unwrapped.ac_real_distribution['other'] = 0
        self.unwrapped.real_relevant_distribution = {"_".join(logtype): 0 for logtype in self.unwrapped.top_logtypes}
        self.unwrapped.step_counter = 0
        self.unwrapped.time_manager.advance_window(global_step=self.unwrapped.all_steps_counter, violation=False, should_delete=self.unwrapped.should_delete, logs_qnt=self.unwrapped.episodic_fake_logs_qnt)
        if self.unwrapped.time_manager.is_delete:
            self.unwrapped.should_delete = False
        self.unwrapped.fake_distribution = {logtype: 0 for logtype in self.unwrapped.top_logtypes}
        self.unwrapped.fake_distribution['other'] = 0
        self.fake_relevant_distribution = {"_".join(logtype): 0 for logtype in self.unwrapped.top_logtypes}

        self.unwrapped.ac_fake_distribution = {logtype: 0 for logtype in self.unwrapped.top_logtypes}
        self.unwrapped.ac_fake_distribution['other'] = 0
        self.unwrapped.rules_rel_diff_alerts = {rule : 0 for rule in self.unwrapped.relevant_logtypes}
        self.unwrapped.episodic_fake_logs_qnt = 0
        self.unwrapped.episodic_inserted_logs = 0
        
        # reset episode logs which are placed at lower wrapper (action)
        # self.action_wrapper.episode_logs = {f"{key[0]}_{key[1]}_{istrigger}":0 for key in self.unwrapped.top_logtypes for istrigger in [0, 1]}
        

            
        new_obs = self.observation(None)
        # new_obs = np.append(self.ac_real_state, self.ac_real_state)
        options = self.get_step_info()
        obs, info = self.env.reset(seed=seed, options=options)
        info.update(self.get_step_info())
        
        return new_obs, info

class StateWrapper3(StateWrapper):
    """Manages log type distributions and state normalization"""

    def __init__(self, env, is_sampled=False):
        super().__init__(env, is_sampled)
        self.observation_space = spaces.Box(
            low=0,
            high=3,
            shape=(len(self.unwrapped.top_logtypes)*2+3,),  # +1 for 'other' category
            # shape=(len(self.unwrapped.top_logtypes),),  # +1 for 'other' category
            dtype=np.float64
        )
        
        
    def observation(self, obs):
        """Convert current distributions to normalized state"""
        # Calculate fake distribution using the latest episode_logs   
        # This happens AFTER action wrapper has updated episode_logs
        # Update real distribution AFTER action is executed
        if not self.unwrapped.done:
            self.update_real_distribution(self.unwrapped.time_manager.action_window.to_tuple())

        # Create state vectors
        real_state = self._get_state_vector(self.unwrapped.real_distribution)
        self.unwrapped.real_state = self._normalize(real_state)
        ac_real_state = self._get_state_vector(self.unwrapped.ac_real_distribution)
        self.unwrapped.ac_real_state = self._normalize(ac_real_state)
        self.unwrapped.real_relevant_distribution = {"_".join(logtype): self.unwrapped.ac_real_state[self.unwrapped.relevant_logtypes_indices[logtype]] for logtype in self.unwrapped.top_logtypes}
        if not self.unwrapped.done:
            self.update_fake_distribution_from_real()
        self.unwrapped.fake_state = self._get_state_vector(self.unwrapped.fake_distribution)
        self.unwrapped.fake_state = self._normalize(self.unwrapped.fake_state)
        ac_fake_state = self._get_state_vector(self.unwrapped.ac_fake_distribution)
        self.unwrapped.ac_fake_state = self._normalize(ac_fake_state)
        self.unwrapped.fake_relevant_distribution = {"_".join(logtype): self.unwrapped.ac_fake_state[self.unwrapped.relevant_logtypes_indices[logtype]] for logtype in self.unwrapped.top_logtypes}

        # Create the final state vector
        # state = self.unwrapped.real_state
        # state = np.append(self.unwrapped.real_state, self.unwrapped.fake_state)
        # compute kl divergence between real and fake distributions (normalize before)
        p = self.unwrapped.ac_real_state/ np.sum(self.unwrapped.ac_real_state)
        q = self.unwrapped.ac_fake_state / np.sum(self.unwrapped.ac_fake_state)
        distribution_distance = np.sum(p * np.log(p / q))
        state = np.append(self.unwrapped.ac_real_state, self.unwrapped.ac_fake_state)
        state = np.append(state, distribution_distance)
        # append step index component
        state = np.append(state, self.unwrapped.step_counter/self.unwrapped.total_steps)
        
        # rules_rel_diff_alerts = [value for key, value in self.unwrapped.rules_rel_diff_alerts.items()]
        # state = np.append(state, rules_rel_diff_alerts)
        # sparse_vector = np.zeros(self.unwrapped.total_steps)
        # sparse_vector[self.unwrapped.step_counter-1] = 1
        # state = np.append(state, sparse_vector)
        state = np.append(state, self.action_wrapper.current_real_quantity/100000)  # Normalize by total logs
        # real_total_logs = self.total_current_logs
        # real_total_logs = self.total_episode_logs
        # state = np.append(state, real_total_logs/500000)
        # state = np.append(state, (self.unwrapped.episodic_inserted_logs+real_total_logs)/500000)
        # # append to state the step index
        # # state = np.append(state, self.env.step_counter/self.unwrapped.total_steps)
        # # add sparse vector for step index

        # current_datetime = datetime.datetime.strptime(self.env.time_manager.action_window.end, '%m/%d/%Y:%H:%M:%S')
        # weekday_vector = np.zeros(7)
        # weekday_vector[current_datetime.weekday()] = 1
        # hour_vector = np.zeros(24)
        # hour_vector[current_datetime.hour] = 1
        # state = np.append(state, weekday_vector)
        # state = np.append(state, hour_vector)
        logger.debug(f"State: {state}")
        self.unwrapped.obs = state
        return state
    
    def _normalize(self, state):
        """Normalize state vector"""
        return (state+ 0.0000000001) / (100000)  # Avoid division by zero
        # return tate / (sum(state) + 0.0000000001)
     

     
class StateWrapper4(StateWrapper):
    """Manages log type distributions and state normalization"""

    def __init__(self, env, is_sampled=False):
        super().__init__(env, is_sampled)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(self.unwrapped.top_logtypes)*2+len(self.unwrapped.relevant_logtypes)*2 + 2,),  # +1 for 'other' category
            # shape=(len(self.unwrapped.top_logtypes),),  # +1 for 'other' category
            dtype=np.float64
        )
        self.baseline_alerts = {}
        
    def observation(self, obs):
        """Convert current distributions to normalized state"""
        # Calculate fake distribution using the latest episode_logs   
        # This happens AFTER action wrapper has updated episode_logs
        # Update real distribution AFTER action is executed
        if not self.unwrapped.done:
            self.update_real_distribution(self.unwrapped.time_manager.action_window.to_tuple())

        # Create state vectors
        real_state = self._get_state_vector(self.unwrapped.real_distribution)
        self.unwrapped.real_state = self._normalize(real_state)
        ac_real_state = self._get_state_vector(self.unwrapped.ac_real_distribution)
        self.unwrapped.ac_real_state = self._normalize(ac_real_state)
        real_sum = sum(ac_real_state)
        
        self.unwrapped.real_relevant_distribution = {"_".join(logtype): self.unwrapped.ac_real_state[self.unwrapped.relevant_logtypes_indices[logtype]] for logtype in self.unwrapped.top_logtypes}
        if not self.unwrapped.done:
            self.update_fake_distribution_from_real()
        fake_state = self._get_state_vector(self.unwrapped.fake_distribution)
        self.unwrapped.fake_state = self._normalize(fake_state)
        ac_fake_state = self._get_state_vector(self.unwrapped.ac_fake_distribution)
        self.unwrapped.ac_fake_state = self._normalize(ac_fake_state)
        fake_sum = sum(ac_fake_state)
        self.unwrapped.fake_relevant_distribution = {"_".join(logtype): self.unwrapped.ac_fake_state[self.unwrapped.relevant_logtypes_indices[logtype]] for logtype in self.unwrapped.top_logtypes}

        state = np.append(real_sum/100000, self.unwrapped.ac_real_state)
        state = np.append(state, fake_sum/100000)
        state = np.append(state, self.unwrapped.ac_fake_state)
        
        # normalized_distribution = np.array(list(self.unwrapped.ac_real_distribution.values())[:-1]) / 237158
        normalized_distribution = pd.DataFrame([self.unwrapped.ac_real_distribution]).drop("other", axis=1)/237158
        if self.unwrapped.step_counter == self.unwrapped.total_steps:
            diversities = {f"{'_'.join(key)}_1": 0 for key in self.unwrapped.top_logtypes}
        else:
            diversities = self.env.env.env.env.env.diversity_episode_logs
        
        expected_normal_alert_rates = []
        expected_fake_alert_rates = []
        if (self.unwrapped.time_manager.current_window.start, self.unwrapped.time_manager.action_window.end) not in self.baseline_alerts:
            self.baseline_alerts[(self.unwrapped.time_manager.current_window.start, self.unwrapped.time_manager.action_window.end)] = {}
        for rule, logtypes in self.unwrapped.section_logtypes.items():
            key = "_".join(logtypes[0])
            key = f"{key}_1"
            if rule in self.unwrapped.savedsearches:
                if rule not in self.baseline_alerts[(self.unwrapped.time_manager.current_window.start, self.unwrapped.time_manager.action_window.end)]:
                    # self.baseline_alerts[self.unwrapped.time_manager.action_window.end][rule] = 0
                    self.baseline_alerts[(self.unwrapped.time_manager.current_window.start, self.unwrapped.time_manager.action_window.end)][rule] = self.normal_alert_predictors[rule].predict(normalized_distribution)[0]
                    # normal_alert_rate = self.normal_alert_predictors[rule].predict(normalized_distribution.reshape(1, -1))[0]
                normal_alert_rate = self.baseline_alerts[(self.unwrapped.time_manager.current_window.start, self.unwrapped.time_manager.action_window.end)][rule]
                if (self.unwrapped.time_manager.current_window.start, self.unwrapped.time_manager.action_window.start) in self.baseline_alerts and rule in self.baseline_alerts[(self.unwrapped.time_manager.current_window.start, self.unwrapped.time_manager.action_window.start)]:
                    normal_alert_rate = max(normal_alert_rate, self.baseline_alerts[(self.unwrapped.time_manager.current_window.start, self.unwrapped.time_manager.action_window.start)][rule])
                expected_normal_alert_rates.append(normal_alert_rate/100)
                if rule in ['ESCU Windows Rapid Authentication On Multiple Hosts Rule']:
                    expected_fake_alert_rates.append((normal_alert_rate)/100)  
                else:
                    expected_fake_alert_rates.append((normal_alert_rate + diversities[key])/100)  
            
        state = np.append(state, expected_normal_alert_rates)
        state = np.append(state, expected_fake_alert_rates)

        logger.info(f"Expected normal alerts: {expected_normal_alert_rates}")
        logger.info(f"Expected fake alerts: {expected_fake_alert_rates}")
        logger.debug(f"State: {state}")
        self.unwrapped.obs = state
        return state
    
    def _normalize(self, state):
        """Normalize state vector"""
        # return (state+ 0.0000000001) / (100000)  # Avoid division by zero
        return state / (sum(state) + 0.0000000001)
     

class StateWrapper5(StateWrapper):
    """Manages log type distributions and state normalization"""

    def __init__(self, env, is_sampled=False):
        super().__init__(env, is_sampled)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(self.unwrapped.top_logtypes)*2+len(self.unwrapped.relevant_logtypes)*2 + 2,),  # +1 for 'other' category
            # shape=(len(self.unwrapped.top_logtypes),),  # +1 for 'other' category
            dtype=np.float64
        )
        self.baseline_alerts = self._load_pickle("/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/baseline/baseline_alerts.pkl", default={})
        self.ac_baseline_alerts = self._load_pickle("/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/baseline/ac_baseline_alerts.pkl", default={})
    
    def _load_pickle(self, filename, default):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)
        return default

    def _dump_pickle(self, obj, filename):
        with open(filename, "wb") as f:
            pickle.dump(obj, f)
   
    def observation(self, obs):
        """Convert current distributions to normalized state"""
        # Calculate fake distribution using the latest episode_logs   
        # This happens AFTER action wrapper has updated episode_logs
        # Update real distribution AFTER action is executed
        if not self.unwrapped.done:
            self.update_real_distribution(self.unwrapped.time_manager.action_window.to_tuple())

        # Create state vectors
        real_state = self._get_state_vector(self.unwrapped.real_distribution)
        self.unwrapped.real_state = self._normalize(real_state)
        ac_real_state = self._get_state_vector(self.unwrapped.ac_real_distribution)
        self.unwrapped.ac_real_state = self._normalize(ac_real_state)
        real_sum = sum(ac_real_state)
        
        self.unwrapped.real_relevant_distribution = {"_".join(logtype): self.unwrapped.ac_real_state[self.unwrapped.relevant_logtypes_indices[logtype]] for logtype in self.unwrapped.top_logtypes}
        if not self.unwrapped.done:
            self.update_fake_distribution_from_real()
        fake_state = self._get_state_vector(self.unwrapped.fake_distribution)
        self.unwrapped.fake_state = self._normalize(fake_state)
        ac_fake_state = self._get_state_vector(self.unwrapped.ac_fake_distribution)
        self.unwrapped.ac_fake_state = self._normalize(ac_fake_state)
        fake_sum = sum(ac_fake_state)
        self.unwrapped.fake_relevant_distribution = {"_".join(logtype): self.unwrapped.ac_fake_state[self.unwrapped.relevant_logtypes_indices[logtype]] for logtype in self.unwrapped.top_logtypes}

        state = np.append(real_sum/1000000, self.unwrapped.ac_real_state)
        state = np.append(state, fake_sum/1000000)
        state = np.append(state, self.unwrapped.ac_fake_state)
        
        # normalized_distribution = np.array(list(self.unwrapped.ac_real_distribution.values())[:-1]) / 237158
        # normalized_distribution = pd.DataFrame([self.unwrapped.ac_real_distribution]).drop("other", axis=1)/237158
        if self.unwrapped.step_counter == self.unwrapped.total_steps:
            diversities = {f"{'_'.join(key)}_1": 0 for key in self.unwrapped.top_logtypes}
        else:
            action_env = self.get_wrapper(ActionWrapper)
            diversities = action_env.diversity_episode_logs
        
        # baseline_df_row = self.unwrapped.baseline_df[(self.unwrapped.baseline_df['start_time'] == self.unwrapped.time_manager.current_window.start) &
        #                                               (self.unwrapped.baseline_df['end_time'] == self.unwrapped.time_manager.action_window.end)]
        # alert_rate = sum(baseline_df_row.groupby('search_name').first()['alert'])
        # if alert_rate != 0:
        #     results = asyncio.run(self.unwrapped.splunk_tools.run_saved_searches_no_measurement(self.unwrapped.time_manager.action_window.to_tuple()))
        # else:
        #     results = {rule: [] for rule in self.unwrapped.savedsearches}
        step_time = self.unwrapped.time_manager.action_window.to_tuple()
        ac_episode_time = (self.unwrapped.time_manager.current_window.start, self.unwrapped.time_manager.action_window.end)
        prev_ac_episode_time = (self.unwrapped.time_manager.current_window.start, self.unwrapped.time_manager.action_window.start)

        if step_time not in self.baseline_alerts:
            self.baseline_alerts[step_time] = {}
        if ac_episode_time not in self.ac_baseline_alerts:
            self.ac_baseline_alerts[ac_episode_time] = {}
        if step_time not in self.ac_baseline_alerts:
            self.ac_baseline_alerts[step_time] = {}
        created_new_baseline = False
        # --- Check if results already exist ---
        results_available = all(
            rule in self.baseline_alerts[step_time]
            for rule in self.unwrapped.savedsearches
        )

        if results_available:
            # ✅ Load from cache
            results = {
                rule: [None] * self.baseline_alerts[step_time][rule]
                for rule in self.unwrapped.savedsearches
            }
        else:
            # ❌ Run Splunk only if missing
            baseline_df_row = self.unwrapped.baseline_df[
                (self.unwrapped.baseline_df["start_time"] == self.unwrapped.time_manager.current_window.start)
                & (self.unwrapped.baseline_df["end_time"] == self.unwrapped.time_manager.action_window.end)
            ]
            alert_rate = sum(baseline_df_row.groupby("search_name").first()["alert"])

            if alert_rate != 0:
                results = asyncio.run(
                    self.unwrapped.splunk_tools.run_saved_searches_no_measurement(
                        step_time
                    )
                )
                created_new_baseline = True
            else:
                results = {rule: [] for rule in self.unwrapped.savedsearches}
        expected_normal_alert_rates = []
        expected_fake_alert_rates = []



        for rule, logtypes in self.unwrapped.section_logtypes.items():
            key = "_".join(logtypes[0])
            key = f"{key}_1"
            if rule in self.unwrapped.savedsearches:
                if rule not in self.baseline_alerts[step_time]:
                    self.baseline_alerts[step_time][rule] = len(results[rule])
                if rule not in self.ac_baseline_alerts[step_time]:
                    self.ac_baseline_alerts[ac_episode_time][rule] = len(results[rule])
                if prev_ac_episode_time in self.ac_baseline_alerts and rule in self.ac_baseline_alerts[prev_ac_episode_time]:
                    self.ac_baseline_alerts[ac_episode_time][rule] = self.ac_baseline_alerts[prev_ac_episode_time][rule] + len(results[rule])
                
                normal_alert_rate = self.ac_baseline_alerts[ac_episode_time][rule]
                expected_normal_alert_rates.append(normal_alert_rate/100)
                if rule in ['ESCU Windows Rapid Authentication On Multiple Hosts Rule']:
                    expected_fake_alert_rates.append((normal_alert_rate)/100)  
                else:
                    expected_fake_alert_rates.append((normal_alert_rate + diversities[key])/100)  

                # --- Only dump if we created new entries ---
                if created_new_baseline:
                    self._dump_pickle(self.baseline_alerts, "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/baseline/baseline_alerts.pkl")
                if created_new_baseline:
                    self._dump_pickle(self.ac_baseline_alerts, "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/baseline/ac_baseline_alerts.pkl")
        state = np.append(state, expected_normal_alert_rates)
        state = np.append(state, expected_fake_alert_rates)

        # logger.info(f"Expected normal alerts: {expected_normal_alert_rates}")
        # logger.info(f"Expected fake alerts: {expected_fake_alert_rates}")
        state = np.round(state, 2)
        # logger.info(f"State: {state}")
        self.unwrapped.obs = state
        return state
    
    def _normalize(self, state):
        """Normalize state vector"""
        # return (state+ 0.0000000001) / (100000)  # Avoid division by zero
        return state / (sum(state) + 0.0000000001)
     

     
class StateWrapper6(StateWrapper):
    """Manages log type distributions and state normalization"""

    def __init__(self, env, is_sampled=False):
        super().__init__(env, is_sampled)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            # high=3,
            shape=(len(self.unwrapped.top_logtypes)*2+2,),  # +1 for 'other' category
            # shape=(len(self.top_logtypes),),  # +1 for 'other' category
            dtype=np.float64
        )
        
        
    def observation(self, obs):
        """Convert current distributions to normalized state"""
        # Calculate fake distribution using the latest episode_logs
        # This happens AFTER action wrapper has updated episode_logs
        # Update real distribution AFTER action is executed
        if not self.unwrapped.done:
            self.update_real_distribution(self.unwrapped.time_manager.action_window.to_tuple())

        # Create state vectors
        real_state = self._get_state_vector(self.unwrapped.real_distribution)
        self.unwrapped.real_state = self._normalize(real_state)
        ac_real_state = self._get_state_vector(self.unwrapped.ac_real_distribution)
        self.unwrapped.ac_real_state = self._normalize(ac_real_state)
        real_sum = sum(ac_real_state)
        
        self.unwrapped.real_relevant_distribution = {"_".join(logtype): self.unwrapped.ac_real_state[self.unwrapped.relevant_logtypes_indices[logtype]] for logtype in self.unwrapped.top_logtypes}
        if not self.unwrapped.done:
            self.update_fake_distribution_from_real()
        fake_state = self._get_state_vector(self.unwrapped.fake_distribution)
        self.unwrapped.fake_state = self._normalize(fake_state)
        ac_fake_state = self._get_state_vector(self.unwrapped.ac_fake_distribution)
        self.unwrapped.ac_fake_state = self._normalize(ac_fake_state)
        fake_sum = sum(ac_fake_state)
        self.unwrapped.fake_relevant_distribution = {"_".join(logtype): self.unwrapped.ac_fake_state[self.unwrapped.relevant_logtypes_indices[logtype]] for logtype in self.unwrapped.top_logtypes}

        # Create the final state vector
        # state = self.unwrapped.real_state
        # state = np.append(self.unwrapped.real_state, self.unwrapped.fake_state)
        state = np.append(self.unwrapped.ac_real_state, self.unwrapped.ac_fake_state)
        
        # rules_rel_diff_alerts = [value for key, value in self.unwrapped.rules_rel_diff_alerts.items()]
        # state = np.append(state, rules_rel_diff_alerts)
        # sparse_vector = np.zeros(self.env.total_steps)
        # sparse_vector[self.unwrapped.step_counter-1] = 1
        # state = np.append(state, sparse_vector)
        # state = np.append(state, self.current_real_quantity/100000) 
        # real_total_logs = self.total_current_logs
        real_total_logs = self.total_episode_logs
        state = np.append(state, real_total_logs/500000)
        state = np.append(state, (self.unwrapped.episodic_inserted_logs+real_total_logs)/500000)
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
        # return state / (500000)  # Avoid division by zero
        return state / (sum(state) + 0.0000000001)
     
class StateWrapper7(StateWrapper):
    """Manages log type distributions and state normalization"""

    def __init__(self, env, is_sampled=False):
        super().__init__(env, is_sampled)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            # high=3,
            shape=(len(self.unwrapped.top_logtypes)*2+3,),  # +1 for 'other' category
            # shape=(len(self.top_logtypes),),  # +1 for 'other' category
            dtype=np.float64
        )
        
        
    def observation(self, obs):
        """Convert current distributions to normalized state"""
        # Calculate fake distribution using the latest episode_logs
        # This happens AFTER action wrapper has updated episode_logs
        # Update real distribution AFTER action is executed
        if not self.unwrapped.done:
            self.update_real_distribution(self.unwrapped.time_manager.action_window.to_tuple())

        # Create state vectors
        real_state = self._get_state_vector(self.unwrapped.real_distribution)
        self.unwrapped.real_state = self._normalize(real_state)
        ac_real_state = self._get_state_vector(self.unwrapped.ac_real_distribution)
        self.unwrapped.ac_real_state = self._normalize(ac_real_state)
        real_sum = sum(ac_real_state)
        
        self.unwrapped.real_relevant_distribution = {"_".join(logtype): self.unwrapped.ac_real_state[self.unwrapped.relevant_logtypes_indices[logtype]] for logtype in self.unwrapped.top_logtypes}
        if not self.unwrapped.done:
            self.update_fake_distribution_from_real()
        fake_state = self._get_state_vector(self.unwrapped.fake_distribution)
        self.unwrapped.fake_state = self._normalize(fake_state)
        ac_fake_state = self._get_state_vector(self.unwrapped.ac_fake_distribution)
        self.unwrapped.ac_fake_state = self._normalize(ac_fake_state)
        fake_sum = sum(ac_fake_state)
        self.unwrapped.fake_relevant_distribution = {"_".join(logtype): self.unwrapped.ac_fake_state[self.unwrapped.relevant_logtypes_indices[logtype]] for logtype in self.unwrapped.top_logtypes}

        # Create the final state vector
        # state = self.unwrapped.real_state
        # state = np.append(self.unwrapped.real_state, self.unwrapped.fake_state)
        state = np.append(self.unwrapped.ac_real_state, self.unwrapped.ac_fake_state)
        
        # rules_rel_diff_alerts = [value for key, value in self.unwrapped.rules_rel_diff_alerts.items()]
        # state = np.append(state, rules_rel_diff_alerts)
        # sparse_vector = np.zeros(self.env.total_steps)
        # sparse_vector[self.unwrapped.step_counter-1] = 1
        # state = np.append(state, sparse_vector)
        # state = np.append(state, self.current_real_quantity/100000) 
        # real_total_logs = self.total_current_logs
        real_total_logs = self.total_episode_logs
        state = np.append(state, real_total_logs/500000)
        state = np.append(state, (self.unwrapped.episodic_inserted_logs+real_total_logs)/500000)
        # # append to state the step index
        # # state = np.append(state, self.env.step_counter/self.env.total_steps)
        # # add sparse vector for step index
        # append step index component
        state = np.append(state, self.unwrapped.step_counter/self.unwrapped.total_steps)
        
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
        # return state / (500000)  # Avoid division by zero
        return state / (sum(state) + 0.0000000001)
     