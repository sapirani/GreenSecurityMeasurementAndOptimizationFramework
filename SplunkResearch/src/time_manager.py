from dataclasses import dataclass
from typing import Tuple, Optional
import datetime
import logging

from env_utils import clean_env
from splunk_tools import SplunkTools

logger = logging.getLogger(__name__)

@dataclass
class TimeWindow:
    """Represents a time window with start and end times"""
    start: str
    end: str
    
    @property
    def duration_minutes(self) -> int:
        """Get duration in minutes"""
        start_dt = datetime.datetime.strptime(self.start, '%m/%d/%Y:%H:%M:%S')
        end_dt = datetime.datetime.strptime(self.end, '%m/%d/%Y:%H:%M:%S')
        return int((end_dt - start_dt).total_seconds() / 60)
    
    def to_tuple(self) -> Tuple[str, str]:
        """Convert to tuple format"""
        return (self.start, self.end)

class TimeManager:
    """Manages all time-related aspects of the environment"""
    
    def __init__(self, 
                 start_datetime: str,
                 window_size: int,
                 step_size: int,
                 rule_frequency: int,
                 end_time: Optional[str] = "06/31/2024:23:59:59",
                 is_test: bool = False,):
        """
        Args:
            start_datetime: Initial datetime in format '%m/%d/%Y:%H:%M:%S'
            window_size: Size of search window in seconds
            step_size: Size of each step in minutes
            rule_frequency: Frequency of rule evaluation in minutes
        """
        self.window_size = window_size
        self.step_size = step_size
        self.rule_frequency = rule_frequency
        self.first_start_datetime = start_datetime
        # Initialize time windows
        self.current_window = self._create_episode_window(start_datetime)
        self.action_window = self._create_action_window(start_datetime)
        self.end_time = end_time
        self._validate_configuration()
        self.splunk_tools = SplunkTools()
        self.is_test = is_test
        self.is_delete = False
        
    def _validate_configuration(self):
        """Validate time configuration"""
        if self.window_size*60 % self.step_size != 0:
            raise ValueError("Window size must be divisible by step size")
        if self.window_size*60 % self.rule_frequency != 0:
            raise ValueError("Window size must be divisible by rule frequency")
            
    def _create_episode_window(self, start_datetime: str) -> TimeWindow:
        """Create initial time window"""
        start_dt = datetime.datetime.strptime(start_datetime, '%m/%d/%Y:%H:%M:%S')
        end_dt = start_dt + datetime.timedelta(minutes=self.window_size)
        return TimeWindow(
            start=start_dt.strftime('%m/%d/%Y:%H:%M:%S'),
            end=end_dt.strftime('%m/%d/%Y:%H:%M:%S')
        )
        
    def _create_action_window(self, start_datetime: str) -> TimeWindow:
        """Create initial action window"""
        start_dt = datetime.datetime.strptime(start_datetime, '%m/%d/%Y:%H:%M:%S')
        end_dt = start_dt + datetime.timedelta(seconds=self.step_size)
        return TimeWindow(
            start=start_dt.strftime('%m/%d/%Y:%H:%M:%S'),
            end=end_dt.strftime('%m/%d/%Y:%H:%M:%S')
        )
    
    def step(self) -> TimeWindow:
        """Move forward one step"""
        # Create action window for current step

        self.action_window = self._create_action_window(self.get_current_time())        
        return self.action_window
        
    def advance_window(self, global_step, violation: bool = False, should_delete: bool = False, logs_qnt = None) -> TimeWindow:
        """Advance the main time window"""
        self.is_delete = False
        if violation:
            # On violation, stay at current window
            return self.current_window
        # clean env in action window
        if not self.is_test and should_delete: #and self.rule_frequency < self.window_size 
            clean_env(self.splunk_tools, (self.current_window.start, self.current_window.end), logs_qnt=logs_qnt)
        new_start_dt = datetime.datetime.strptime(self.current_window.start, '%m/%d/%Y:%H:%M:%S')
        new_start_dt += datetime.timedelta(minutes=self.rule_frequency)
        if self.end_time:
            end_datetime = datetime.datetime.strptime(self.end_time, '%m/%d/%Y:%H:%M:%S')
            if new_start_dt >= end_datetime:
                logger.info(f"End time {self.end_time} , current time {new_start_dt.strftime('%m/%d/%Y:%H:%M:%S')}")
                logger.info("End of times arived, resetting to start time")# + one hour")
                if not self.is_test and should_delete:
                    clean_env(self.splunk_tools, (self.first_start_datetime, self.end_time))
                    if should_delete:
                        self.is_delete = True
                        
                
                # Reset to start time + one hour
                start_dt = datetime.datetime.strptime(self.first_start_datetime, '%m/%d/%Y:%H:%M:%S')
                # start_dt += datetime.timedelta(hours=1)
                self.first_start_datetime = start_dt.strftime('%m/%d/%Y:%H:%M:%S')

                # Reset to start time
                self.current_window = self._create_episode_window(self.first_start_datetime)
                self.action_window = self._create_action_window(self.first_start_datetime)
                return self.current_window
        
        # Move window forward by rule frequency except in the first episode
        if global_step == 0:
            logger.info("First episode, not advancing window")
            self.current_window = self._create_episode_window(self.first_start_datetime)
            self.action_window = self._create_action_window(self.first_start_datetime)
            return self.current_window

        self.current_window = self._create_episode_window(new_start_dt.strftime('%m/%d/%Y:%H:%M:%S'))
        logger.info(f"Advanced window to {self.current_window.start} - {self.current_window.end}")
        self.action_window = self._create_action_window(new_start_dt.strftime('%m/%d/%Y:%H:%M:%S'))
        return self.current_window
        
    def get_current_time(self) -> str:
        """Get current time (end of action window or current window)"""
        if self.action_window:
            return self.action_window.end
        return self.current_window.start
        
    def get_previous_time(self, seconds: int) -> str:
        """Get time n seconds before current time"""
        current = datetime.datetime.strptime(self.get_current_time(), '%m/%d/%Y:%H:%M:%S')
        previous = current - datetime.timedelta(seconds=seconds)
        return previous.strftime('%m/%d/%Y:%H:%M:%S')
        
    def get_time_info(self) -> dict:
        """Get time information for current state"""
        current_dt = datetime.datetime.strptime(self.get_current_time(), '%m/%d/%Y:%H:%M:%S')
        return {
            'current_time': self.get_current_time(),
            'current_window': self.current_window.to_tuple(),
            'action_window': self.action_window.to_tuple() if self.action_window else None,
            'week_day': current_dt.weekday(),
            'hour': current_dt.hour
        }

# wrapper for time menaging
from gymnasium.core import Wrapper
class TimeWrapper(Wrapper):
    # def __init__(self, env, start_datetime, window_size, step_size, rule_frequency):
    #     super().__init__(env)
    #     self.time_manager = TimeManager(start_datetime, window_size, step_size, rule_frequency)
        
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if not info['done']:
            action_window = self.unwrapped.time_manager.step()
            info['action_window'] = action_window
        return obs, reward, done, truncated, info
        # return self.env.step(action)
    

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Advance time window based on previous episode
        # self.time_manager.advance_window(violation=self.step_violation)
        # action_window = self.time_manager.step()
        # info['action_window'] = action_window
        return obs, info
        