import datetime
import random
import time
import numpy as np
import pandas as pd
import sys
import urllib3
import logging

sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src')
from env_utils import *
from time_manager import TimeManager

from datetime_manager import MockedDatetimeManager

import tensorflow as tf
from strategies.action_strategy import ActionStrategy14, ActionStrategy7, ActionStrategy8

from strategies.state_strategy import StateStrategy12, StateStrategy11, StateStrategy6, StateStrategy7, StateStrategy8
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment')
import os
from dotenv import load_dotenv
load_dotenv('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/.env')
urllib3.disable_warnings()
from splunk_tools import SplunkTools
from log_generator import LogGenerator
from resources.section_logtypes import section_logtypes

import logging
logger = logging.getLogger(__name__)
import concurrent.futures
from wrappers.reward import *


from dataclasses import dataclass
import gymnasium as gym
from gymnasium import register, spaces
import numpy as np
import logging
import datetime
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)

@dataclass
class SplunkConfig:
    """Configuration parameters for Splunk environment"""
    # Time parameters #optional
    rule_frequency: float
    search_window: int
    fake_start_datetime: str = None
    
    action_duration: int = 1
    
    # Load parameters
    logs_per_minute: int = 300
    additional_percentage: float = 0.1
    
    # Rules configuration
    savedsearches: List[str] = None
    
    # Monitoring
    num_of_measurements: int = 1
    num_of_episodes: int = 1000
    
    state_strategy: Any = "StateStrategy12"
    action_strategy: Any = "ActionStrategy14"
    env_id: str = "splunk_train-v32"

class SplunkEnv(gym.Env):
    """Splunk environment for resource consumption experiments"""
    
    def __init__(self,
                 savedsearches: List[str],
                 fake_start_datetime: str,
                 config: SplunkConfig,
                 state_strategy: Any,
                 action_strategy: Any):
        """Initialize environment."""
        super().__init__()
                # Initialize time manager
        self.time_manager = TimeManager(
            start_datetime=fake_start_datetime if fake_start_datetime else config.fake_start_datetime,
            window_size=config.search_window,
            step_size=config.action_duration,
            rule_frequency=config.rule_frequency
        )

        # Store configuration
        self.config = config
        self.relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]}))
        # Initialize tools and strategies
        self.splunk_tools  = SplunkTools(savedsearches, config.num_of_measurements, config.rule_frequency)
        self.log_generator = LogGenerator(self.relevant_logtypes, self.splunk_tools)


        self.top_logtypes = pd.read_csv("/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/resources/top_logtypes.csv")
        # include only system and security logs
        self.top_logtypes = self.top_logtypes[self.top_logtypes['source'].str.lower().isin(['wineventlog:security', 'wineventlog:system'])]
        self.top_logtypes = self.top_logtypes.sort_values(by='count', ascending=False)[['source', "EventCode"]].values.tolist()[:50]
        self.top_logtypes = [(x[0].lower(), str(x[1])) for x in self.top_logtypes]
        self.top_logtypes = set(self.top_logtypes)|set(self.relevant_logtypes)

        self.state_strategy = state_strategy(self.top_logtypes)
        self.action_strategy = action_strategy(self.relevant_logtypes, 1, 0, self.config.action_duration, self.splunk_tools, self.log_generator, 0)
        
        # Set up action and observation spaces
        self.action_space = self.action_strategy.create_action_space()
        self.observation_space = self.state_strategy.create_state()
        
        # Calculate episode parameters
        self.total_steps = self.config.search_window * 60 // self.config.action_duration
        
        # Initialize episode tracking
        self.step_counter = 0
        self.all_steps_counter = 0
        self.action_auditor = []
        self.step_violation = False
        self.done = False
        # Initialize time management
        # self._setup_time_range(fake_start_datetime if fake_start_datetime else config.fake_start_datetime)
        
        # Warm up environment
        self._warmup()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute environment step."""
        self.step_counter += 1
        self.all_steps_counter += 1
        
        # Get time window for current step
        action_window = self.time_manager.step()
        
        # Process action
        self.action_auditor.append((action_window.to_tuple(), action))
        self.action_strategy.record_action(action)
        self.remaining_quota = self.action_strategy.remaining_quota
        
        # Update state
        # self._update_state()
        obs = None
        # Check termination
        # self.step_violation = self._check_step_violation()
        self.done = self.step_violation or self._check_termination()
        truncated = False
        
        # Calculate base reward (wrappers will modify this)
        reward = 0
        
        # Get info including time information
        info = self.get_step_info()

        
        # Clean up if done
        if self.done:
            self._execute_pending_actions()
        
        return obs, reward, self.done, truncated, info
    

    
    def _execute_pending_actions(self) -> None:
        """Execute all pending actions in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    self.action_strategy.perform_action, 
                    action, 
                    time_range
                ): (time_range, action) 
                for time_range, action in self.action_auditor
            }
            
            for future in concurrent.futures.as_completed(futures):
                time_range, action = futures[future]
                try:
                    future.result()
                    logger.info(f"Action {action} completed successfully")
                except Exception as exc:
                    logger.error(f"Action {action} failed: {exc}")
    
  
           
    # def _update_state(self) -> None:
    #     """Update environment state"""
    #     #TODO: Implement state update
    #     # Get time ranges   
    #     current_window = self.time_manager.current_window.to_tuple()
        
    #     # Get distributions
    #     real_dist = self.splunk_tools.get_real_distribution(*current_window)
    #     fake_dist = self.action_strategy.get_current_distribution()
        
    #     # Update state strategy
    #     self.state_strategy.update_distributions(real_dist, fake_dist)
    #     self.state_strategy.update_quota(self.remaining_quota / self.total_additional_logs)
        
    #     # Update state
    #     self.state = self.state_strategy.update_state()

    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        if self.step_counter >= self.total_steps:
            return True
            
        if self.remaining_quota >= 1.5:
            return True
            
        return False


    def reset(self, *, seed=None, options=None):
        """Reset environment state.
        
        Args:
            seed: Optional random seed
            options: Optional configuration dictionary
            
        Returns:
            observation: Initial environment observation
            info: Information dictionary
        """
        # Optional: set random seed
        if seed is not None:
            super().reset(seed=seed)
        
        # Reset counters and tracking
        self.step_counter = 0
        self.action_auditor = []
        # Advance time window based on previous episode
        self.time_manager.advance_window(violation=self.step_violation)
        
        # Reset strategies
        self.action_strategy.reset()
        self.state = self.state_strategy.reset()
        
        # Update time range
        # self._update_time_range()
        
        # Calculate quotas
        self._calculate_quota()
        
        # Get initial info
        info = self.get_step_info()
        
        return self.state, info

    def _calculate_quota(self) -> None:
        """Calculate injection quotas"""
        self.total_additional_logs = (self.config.additional_percentage * 
                                    self.config.search_window * 
                                    self.config.logs_per_minute)
        
        self.step_size = int((self.total_additional_logs // self.config.search_window) * 
                            self.config.action_duration // 60)
        self.remaining_quota = self.step_size
        self.action_strategy.quota = self.remaining_quota

    def get_step_info(self) -> Dict[str, Any]:
        """Get information about current step"""
        return {
            'step': self.step_counter,
            'all_steps_counter': self.all_steps_counter,
            'remaining_quota': self.remaining_quota,
            'total_additional_logs': self.total_additional_logs,
            'real_distribution': self.state_strategy.real_state,
            'fake_distribution': self.state_strategy.fake_state,
            'total_steps': self.total_steps,
            'done': self.done, 
            **self.time_manager.get_time_info()
            # 'action_metrics': self.action_strategy.get_metrics(),
            # 'state_metrics': self.state_strategy.get_metrics(),
            # 'current_cpu': self.splunk_tools.get_current_cpu(),
            # 'baseline_cpu': self.splunk_tools.get_baseline_cpu(),
            # 'alert_counts': self.splunk_tools.get_alert_counts(),
            # 'expected_alerts': self.splunk_tools.get_expected_alerts()
        }



    def _warmup(self) -> None:
        """Warm up the environment"""
        for _ in range(1):
            logger.info("Running saved searches for warmup")
            self.splunk_tools.run_saved_searches_parallel(self.time_manager.current_window.to_tuple())

# Example usage:
if __name__ == "__main__":
    # Create configuration
    config = SplunkConfig(
        # fake_start_datetime="02/12/2025:00:00:00",
        rule_frequency=60,
        search_window=120,
        # savedsearches=["rule1", "rule2"],
        logs_per_minute=300,
        additional_percentage=0.1,
        action_duration=60,
        num_of_measurements=1,
        num_of_episodes=1000
    )
    savedsearches = ["Windows Event For Service Disabled",
                 "Detect New Local Admin account",
                 "ESCU Network Share Discovery Via Dir Command Rule",
                 "Known Services Killed by Ransomware",
                 "Non Chrome Process Accessing Chrome Default Dir",
                 "Kerberoasting spn request with RC4 encryption",
                 "Clop Ransomware Known Service Name"]
    fake_start_datetime = "02/01/2024:00:00:00"
    env_id = "splunk_train-v32"
    register(id=env_id,
            entry_point='custom_splunk.envs:SplunkEnv', 
            kwargs={
                    'savedsearches':savedsearches,
                    'fake_start_datetime':fake_start_datetime,

            })
    # Create base environment
    env = gym.make(id="splunk_train-v32", config=config,
                    state_strategy=StateStrategy12,
                    action_strategy=ActionStrategy14)
    
    # Add reward wrappers
    env = DistributionRewardWrapper(env, gamma=0.2)
    env = BaseRuleExecutionWrapper(env)
    env = EnergyRewardWrapper(env, alpha=0.5)
    env = AlertRewardWrapper(env, beta=0.3)
    # env = QuotaViolationWrapper(env)
    
    # Run environment
    obs, info = env.reset()
    for _ in range(config.num_of_episodes):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {info['step']}: Reward {reward}")
        if terminated or truncated:
            obs, info = env.reset()