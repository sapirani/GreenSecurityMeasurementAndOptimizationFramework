import asyncio
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
from wrappers.state import *
from wrappers.action import *


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
    baseline_num_of_measurements: int = 1
    num_of_episodes: int = 1000

    env_id: str = "splunk_train-v32"
    end_time: str = None
    is_test: bool = False

class SplunkEnv(gym.Env):
    """Splunk environment for resource consumption experiments"""
    
    def __init__(self,
                 savedsearches: List[str],
                 fake_start_datetime: str,
                 config: SplunkConfig,
                 top_logtypes: List[Tuple[str, str]]):
        """Initialize environment."""
        super().__init__()
        self.splunk_tools  = SplunkTools(savedsearches, config.rule_frequency)
        self.episodic_inserted_logs = 0

        # Initialize time manager
        self.time_manager = TimeManager(
            start_datetime=config.fake_start_datetime if config.fake_start_datetime else fake_start_datetime,
            window_size=config.search_window,
            step_size=config.action_duration,
            rule_frequency=config.rule_frequency,
            end_time=config.end_time,
            is_test=config.is_test
        )
        # Define basic action space that will be overridden by wrapper
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(1,),  # Just a placeholder
            dtype=np.float32
        )
        
        # Define basic observation space that will be overridden by wrapper
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(1,),  # Just a placeholder
            dtype=np.float32
        )

        # Store configuration
        self.config = config
        self.section_logtypes = section_logtypes
        self.relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]}))
        # concat top_logtypes and relevant_logtypes, while removing duplicates and keeping order
        top_logtypes = sorted(list(dict.fromkeys(self.relevant_logtypes + top_logtypes)))       
        self.top_logtypes = top_logtypes
        
        self.savedsearches = savedsearches
        # Initialize tools and strategies
        self.log_generator = LogGenerator(self.top_logtypes)
        # self.log_generator = LogGenerator(self.relevant_logtypes, self.splunk_tools)
        self._normalize_factor = 300000
        
        # Calculate episode parameters
        self.total_steps = self.config.search_window * 60 // self.config.action_duration
        
        # Initialize episode tracking
        self.step_counter = 0
        self.all_steps_counter = 0
        self.action_auditor = []
        self.step_violation = False
        self.done = False
        self.obs = None
        self.real_state = np.array([])
        self.fake_state = np.array([])
        self.ac_real_state = np.array([])
        self.ac_fake_state = np.array([])
        self.fake_distribution = {logtype: 0 for logtype in self.top_logtypes}
        self.fake_distribution['other'] = 0
        self.fake_relevant_distribution = {"_".join(logtype): 0 for logtype in self.top_logtypes}
        self.real_distribution = {logtype: 0 for logtype in self.top_logtypes}
        self.real_distribution['other'] = 0
        self.ac_real_distribution = {logtype: 0 for logtype in self.top_logtypes}
        self.ac_real_distribution['other'] = 0
        self.ac_fake_distribution = {logtype: 0 for logtype in self.top_logtypes}
        self.ac_fake_distribution['other'] = 0
        self.real_relevant_distribution = {"_".join(logtype): 0 for logtype in self.top_logtypes}
        self.relevant_logtypes_indices = {logtype: i for i, logtype in enumerate(self.top_logtypes) if logtype in self.top_logtypes}
        self.normal_distribution = {
                            "wineventlog:security_4624": 0.0749,
                            "wineventlog:security_4625": 0.017,
                            "wineventlog:security_4634": 0.0262,
                            "wineventlog:security_4648": 0.0336,
                            "wineventlog:security_4662": 0.0913,
                            "wineventlog:security_4663": 0,
                            "wineventlog:security_4672": 0.0719,
                            "wineventlog:security_4702": 0.1829,
                            "wineventlog:security_4732": 0,
                            "wineventlog:security_4735": 0.0175,
                            "wineventlog:security_4769": 0,
                            "wineventlog:security_4799": 0.0443,
                            "wineventlog:security_4907": 0.1764,
                            "wineventlog:security_5140": 0,
                            "wineventlog:security_5379": 0.1142,
                            "wineventlog:system_101": 0.0079,
                            "wineventlog:system_108": 0.008,
                            "wineventlog:system_1112": 0.0079,
                            "wineventlog:system_12": 0.0071,
                            "wineventlog:system_1500": 0.0079,
                            "wineventlog:system_16": 0.006,
                            "wineventlog:system_44": 0.0827,
                            "wineventlog:system_7": 0.0022,
                            "wineventlog:system_7036": 0.015,
                            "wineventlog:system_7040": 0.0046,
                            "wineventlog:system_7045": 0.0003 
                        }
        self.normal_distribution = np.array([self.normal_distribution[ "_".join(logtype)] for logtype in self.top_logtypes if  "_".join(logtype) in self.normal_distribution])
        
        self.rules_rel_diff_alerts = {rule : 0 for rule in self.relevant_logtypes}
        self.is_mock = False
        self.should_delete = False
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute environment step."""
        self.step_counter += 1
        self.all_steps_counter += 1
        logger.info(f"Total steps: {self.all_steps_counter}")
        logger.info(f"Step {self.step_counter}")

        self.done = self.step_violation or self._check_termination()
        truncated = False
        reward = 0
        info = self.get_step_info()
        obs = self.obs
        return self.obs, reward, self.done, truncated, info



 
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        if self.step_counter == self.total_steps:
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


        info = self.get_step_info()

        return np.zeros(self.observation_space.shape), info



    def get_step_info(self) -> Dict[str, Any]:
        """Get information about current step"""
        return {
            'step': self.step_counter,
            'all_steps_counter': self.all_steps_counter,
            # 'remaining_quota': self.remaining_quota,
            # 'total_additional_logs': self.total_additional_logs,
            # 'real_distribution': self.real_state,
            # 'fake_distribution': self.fake_state,
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



    def warmup(self) -> None:
        """Warm up the environment"""
        for _ in range(1):
            logger.info("Running saved searches for warmup")
            time_range = self.time_manager.current_window.to_tuple()
            asyncio.run(self.splunk_tools.run_saved_searches(time_range, None, self.config.num_of_measurements))

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
)    
    top_logtypes = pd.read_csv("/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/resources/top_logtypes.csv")
    # include only system and security logs
    top_logtypes = top_logtypes[top_logtypes['source'].str.lower().isin(['wineventlog:security', 'wineventlog:system'])]
    top_logtypes = top_logtypes.sort_values(by='count', ascending=False)[['source', "EventCode"]].values.tolist()[:50]
    top_logtypes = [(x[0].lower(), str(x[1])) for x in top_logtypes]
    env = StateWrapper(env, top_logtypes)
    env = Action(env)
    # Add reward wrappers
    env = DistributionRewardWrapper(env, gamma=0.2)
    # env = BaseRuleExecutionWrapper(env)
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