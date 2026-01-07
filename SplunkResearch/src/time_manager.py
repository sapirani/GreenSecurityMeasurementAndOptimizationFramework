from calendar import month
from dataclasses import dataclass
import random
from typing import Tuple, Optional, List
import datetime
import logging

# Assuming these exist in your project structure
from env_utils import *
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
                 is_test: bool = False,
                 is_eval: bool = False):
        """
        Args:
            start_datetime: Initial datetime in format '%m/%d/%Y:%H:%M:%S'
            window_size: Size of search window in seconds (Episode duration)
            step_size: Size of each step in minutes (Action duration)
            rule_frequency: Frequency of rule evaluation in minutes (Stride between episode starts)
        """
        self.fmt = '%m/%d/%Y:%H:%M:%S'
        self.window_size = window_size
        self.step_size = step_size
        self.rule_frequency = rule_frequency
        self.first_start_datetime = start_datetime
        self.end_time = end_time
        
        self.is_test = is_test
        self.is_delete = False
        
        self.splunk_tools = SplunkTools()
        
        self._validate_configuration()
        
        # --- NEW: Initialize Queue Logic ---
        # We generate all valid start times once and shuffle them.
        self.unvisited_starts = self._generate_episode_starts()
        if is_eval:
            # For evaluation, we want a fixed order for reproducibility
            logger.info("Evaluation mode: using sorted episode start times for reproducibility.")
            self.unvisited_starts.sort()
        else:
            random.shuffle(self.unvisited_starts)
        logger.info(f"Initialized TimeManager with {len(self.unvisited_starts)} unique episode start times.")

        # Initialize current windows (Set to the very first time initially)
        self.current_window = self._create_episode_window(start_datetime)
        self.action_window = self._create_action_window(start_datetime)

    def _validate_configuration(self):
        """Validate time configuration"""
        if self.window_size*60 % self.step_size != 0:
            raise ValueError("Window size must be divisible by step size")
        # Note: If rule_frequency is just the stride between episodes, strictly implies it doesn't *have* to divide window_size, 
        # but maintaining the check if your logic requires it.
        if self.window_size*60 % self.rule_frequency != 0:
             # Make this a warning if you want to allow overlapping slides that don't match perfectly
             pass 

    def _generate_episode_starts(self) -> List[datetime.datetime]:
        """
        Generates a list of all possible start datetimes between first_start and end_time.
        The stride between starts is determined by self.rule_frequency (in minutes).
        """
        starts = []
        current = datetime.datetime.strptime(self.first_start_datetime, self.fmt)
        end_limit = datetime.datetime.strptime(self.end_time, self.fmt)
        
        # We stop when current + window_size > end_limit so the window doesn't go out of bounds
        window_duration = datetime.timedelta(minutes=self.window_size)
        step_delta = datetime.timedelta(seconds=self.step_size)

        while current + window_duration <= end_limit:
            starts.append(current)
            current += step_delta
            
        return starts

    def _create_episode_window(self, start_datetime: str) -> TimeWindow:
        """Create episode time window (The 'State' window)"""
        start_dt = datetime.datetime.strptime(start_datetime, self.fmt)
        end_dt = start_dt + datetime.timedelta(minutes=self.window_size)
        return TimeWindow(
            start=start_dt.strftime(self.fmt),
            end=end_dt.strftime(self.fmt)
        )
        
    def _create_action_window(self, start_datetime: str) -> TimeWindow:
        """Create action window (The 'Step' window)"""
        start_dt = datetime.datetime.strptime(start_datetime, self.fmt)
        end_dt = start_dt + datetime.timedelta(seconds=self.step_size)
        return TimeWindow(
            start=start_dt.strftime(self.fmt),
            end=end_dt.strftime(self.fmt)
        )
    
    def step(self) -> TimeWindow:
        """Move forward one step WITHIN the current episode"""
        self.action_window = self._create_action_window(self.get_current_time())        
        return self.action_window
        
    def advance_window(self, global_step, violation: bool = False, should_delete: bool = False, logs_qnt = None) -> TimeWindow:
        """
        Advance to the NEXT EPISODE (start time).
        Prioritizes unvisited time windows by popping from the shuffled queue.
        """
        empty_monitored_files(SYSTEM_MONITOR_FILE_PATH)
        empty_monitored_files(SECURITY_MONITOR_FILE_PATH)
        self.is_delete = False
        # Optional: Handle explicit deletion requests if needed
        if not self.is_test and should_delete:
           clean_env(self.splunk_tools, time_range=self.current_window.to_tuple(), logs_qnt=logs_qnt)
           self.is_delete = True
           
        if violation:
            # On violation, stay at current window (retry logic)
            return self.current_window

        # --- Handle First Step Special Case ---
        if global_step == 0:
            logger.info("First episode, using initial start time")
            # Ensure we are set to the first start time defined in init
            self.current_window = self._create_episode_window(self.first_start_datetime)
            self.action_window = self._create_action_window(self.first_start_datetime)
            return self.current_window

        # --- NEW LOGIC: Pop from Queue ---
        if not self.unvisited_starts:
            logger.info("All time windows visited! Resetting and reshuffling queue.")
            self.unvisited_starts = self._generate_episode_starts()
            random.shuffle(self.unvisited_starts)

        # Get next random (but unique) start time
        next_start_dt = self.unvisited_starts.pop()
        next_start_str = next_start_dt.strftime(self.fmt)

        # Update State
        self.current_window = self._create_episode_window(next_start_str)
        self.action_window = self._create_action_window(next_start_str)
        
        logger.info(f"Advanced window to {self.current_window.start} - {self.current_window.end} (Remaining in queue: {len(self.unvisited_starts)})")
        

            
        return self.current_window
        
    def get_current_time(self) -> str:
        """Get current time (end of action window or current window start)"""
        if self.action_window:
            return self.action_window.end
        return self.current_window.start
        
    def get_previous_time(self, seconds: int) -> str:
        """Get time n seconds before current time"""
        current = datetime.datetime.strptime(self.get_current_time(), self.fmt)
        previous = current - datetime.timedelta(seconds=seconds)
        return previous.strftime(self.fmt)
        
    def get_time_info(self) -> dict:
        """Get time information for current state"""
        current_dt = datetime.datetime.strptime(self.get_current_time(), self.fmt)
        return {
            'current_time': self.get_current_time(),
            'current_window': self.current_window.to_tuple(),
            'action_window': self.action_window.to_tuple() if self.action_window else None,
            'week_day': current_dt.weekday(),
            'hour': current_dt.hour
        }

# wrapper for time managing
from gymnasium.core import Wrapper
class TimeWrapper(Wrapper):
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if not info.get('done', False):
            # Only advance the intra-episode step if the episode isn't done
            action_window = self.unwrapped.time_manager.step()
            info['action_window'] = action_window
        return obs, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        # Note: The actual advance_window call usually happens inside the Env's reset 
        # or immediately before it in the training loop, depending on your architecture.
        # If your Env calls manager.advance_window() internally, this is fine.
        return self.env.reset(seed=seed, options=options)