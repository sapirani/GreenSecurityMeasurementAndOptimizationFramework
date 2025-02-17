from dataclasses import dataclass
from typing import Tuple, Optional
import datetime
import logging

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
                 rule_frequency: int):
        """
        Args:
            start_datetime: Initial datetime in format '%m/%d/%Y:%H:%M:%S'
            window_size: Size of search window in minutes
            step_size: Size of each step in minutes
            rule_frequency: Frequency of rule evaluation in minutes
        """
        self.window_size = window_size
        self.step_size = step_size
        self.rule_frequency = rule_frequency
        
        # Initialize time windows
        self.current_window = self._create_initial_window(start_datetime)
        self.action_window = None
        
        self._validate_configuration()
        
    def _validate_configuration(self):
        """Validate time configuration"""
        if self.window_size*60 % self.step_size != 0:
            raise ValueError("Window size must be divisible by step size")
        if self.window_size*60 % self.rule_frequency != 0:
            raise ValueError("Window size must be divisible by rule frequency")
            
    def _create_initial_window(self, start_datetime: str) -> TimeWindow:
        """Create initial time window"""
        start_dt = datetime.datetime.strptime(start_datetime, '%m/%d/%Y:%H:%M:%S')
        end_dt = start_dt + datetime.timedelta(minutes=self.window_size)
        return TimeWindow(
            start=start_dt.strftime('%m/%d/%Y:%H:%M:%S'),
            end=end_dt.strftime('%m/%d/%Y:%H:%M:%S')
        )
        
    def step(self) -> TimeWindow:
        """Move forward one step"""
        # Create action window for current step

        start_dt = datetime.datetime.strptime(self.get_current_time(), '%m/%d/%Y:%H:%M:%S')
        step_end = start_dt + datetime.timedelta(seconds=self.step_size)
        
        self.action_window = TimeWindow(
            start=start_dt.strftime('%m/%d/%Y:%H:%M:%S'),
            end=step_end.strftime('%m/%d/%Y:%H:%M:%S')
        )
        
        return self.action_window
        
    def advance_window(self, violation: bool = False) -> TimeWindow:
        """Advance the main time window"""
        if violation:
            # On violation, stay at current window
            return self.current_window
            
        # Move window forward by rule frequency
        start_dt = datetime.datetime.strptime(self.current_window.start, '%m/%d/%Y:%H:%M:%S')
        start_dt += datetime.timedelta(minutes=self.rule_frequency)
        end_dt = start_dt + datetime.timedelta(minutes=self.window_size)
        
        self.current_window = TimeWindow(
            start=start_dt.strftime('%m/%d/%Y:%H:%M:%S'),
            end=end_dt.strftime('%m/%d/%Y:%H:%M:%S')
        )
        
        logger.info(f"Advanced window to {self.current_window.start} - {self.current_window.end}")
        self.action_window = None
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