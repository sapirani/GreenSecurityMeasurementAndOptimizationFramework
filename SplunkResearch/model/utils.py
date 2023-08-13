import datetime
import logging
from unittest.mock import patch
import time

class MockedDatetime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        # Calculate the time elapsed since the program started
        delta = MockedDatetimeManager._initial_real_datetime.utcnow() - MockedDatetimeManager._initial_real_datetime
        
        # Return the fake datetime adjusted by the elapsed time
        return MockedDatetimeManager._fake_start_datetime + delta

class MockedDatetimeManager:
    
    _initial_real_datetime = datetime.datetime.utcnow()
    _fake_start_datetime = datetime.datetime(2023, 1, 1, 12, 0, 0)
    
    def __init__(self, fake_start_datetime=None):
        if fake_start_datetime:
            MockedDatetimeManager._fake_start_datetime = fake_start_datetime
        
        # Setting up logging
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Ensure that logging uses the real datetime functions
        for handler in self.logger.handlers:
            handler.formatter.converter = self._real_now
    
    def _real_now(self, *args, **kwargs):
        return self._initial_real_datetime.timetuple()
    
    def get_current_datetime(self):
        with patch('datetime.datetime', MockedDatetime):
            current_dt = datetime.datetime.now()
            return current_dt.strftime('%m/%d/%Y:%H:%M:%S')
    
    def log(self, message):
        self.logger.info(message)
        
    def add_time(self, initial_datetime_str, hours=0, minutes=0, seconds=0):
        """
        Add the given number of hours, minutes, and seconds to the fake datetime.
        
        :param hours: Hours to add to the fake datetime.
        :param minutes: Minutes to add to the fake datetime.
        :param seconds: Seconds to add to the fake datetime.
        """
        delta = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return self.get_new_datetime(initial_datetime_str,delta)
    
    def subtract_time(self, initial_datetime_str, hours=0, minutes=0, seconds=0):
        """
        Subtract the given number of hours, minutes, and seconds from the fake datetime.
        
        :param hours: Hours to subtract from the fake datetime.
        :param minutes: Minutes to subtract from the fake datetime.
        :param seconds: Seconds to subtract from the fake datetime.
        """
        delta = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return self.get_new_datetime(initial_datetime_str, -delta)
        
    def get_new_datetime(self,initial_datetime_str,delta):
        """
        Adjust the provided datetime string by the given delta (hours, minutes, seconds) 
        and return the new datetime as a string.
        
        :param initial_datetime_str: Initial datetime in the format "MM/DD/YYYY:HH:MM:SS".
        :param hours: Hours to adjust the datetime by.
        :param minutes: Minutes to adjust the datetime by.
        :param seconds: Seconds to adjust the datetime by.
        :return: New datetime in the format "MM/DD/YYYY:HH:MM:SS".
        """
        # Convert the initial datetime string to a datetime object
        initial_datetime = datetime.datetime.strptime(initial_datetime_str, '%m/%d/%Y:%H:%M:%S')
        
        # Adjust the datetime by the given delta
        new_datetime = initial_datetime + delta
        
        # Convert the new datetime object back to the desired string format
        return new_datetime.strftime('%m/%d/%Y:%H:%M:%S')
    
    def wait_til_next_rule_frequency(self, rule_frequency):
        """
        Wait until the current datetime is rounded to the next rule frequency.
        
        :param rule_frequency: Rule frequency in minutes.
        """
        fake_now = self.get_current_datetime()
        split_fake_now = fake_now.split(':')
        while ((int(split_fake_now[2])+1) % int(rule_frequency) != 0) or (int(split_fake_now[3]) < 40):
            fake_now = self.get_current_datetime()
            split_fake_now = fake_now.split(':')            
            time.sleep(1)