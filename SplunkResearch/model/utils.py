import datetime
import logging
from unittest.mock import patch
import time

class MockedDatetime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        # Calculate the time elapsed since the program started
        delta = MockedDatetimeManager._initial_real_datetime.now() - MockedDatetimeManager._initial_real_datetime
        
        # Return the fake datetime adjusted by the elapsed time
        return MockedDatetimeManager._fake_start_datetime + delta

class MockedDatetimeManager:
    
    _initial_real_datetime = datetime.datetime.now()
    _fake_start_datetime = datetime.datetime(2023, 1, 1, 12, 0, 0)
    
    def __init__(self, fake_start_datetime=None, log_file_path=None):
        if fake_start_datetime:
            MockedDatetimeManager._fake_start_datetime = fake_start_datetime
        
       
        self.current_fake_time = fake_start_datetime
        self.logger = logging.getLogger(__name__)
         # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.logger.setLevel(logging.INFO)

        if log_file_path:
            file_handler = logging.FileHandler(log_file_path)
            # Set the formatter for the file handler
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Ensure that logging uses the real datetime functions
        for handler in self.logger.handlers:
            handler.formatter.converter = self._real_now
    
    def _real_now(self, *args, **kwargs):
        real_dt = self.get_real_current_datetime()
        return datetime.datetime.strptime(real_dt, '%m/%d/%Y:%H:%M:%S').timetuple()

    
    def get_real_current_datetime(self):
        current_dt = datetime.datetime.now()
        return current_dt.strftime('%m/%d/%Y:%H:%M:%S')
    
    def set_fake_current_datetime(self, fake_current_datetime):
        self.current_fake_time = datetime.datetime.strptime(fake_current_datetime, '%m/%d/%Y:%H:%M:%S') 
    
    def get_fake_current_datetime(self):
        return self.current_fake_time.strftime('%m/%d/%Y:%H:%M:%S')
        # with patch('datetime.datetime', MockedDatetime):
        #     current_dt = datetime.datetime.now()
        #     return current_dt.strftime('%m/%d/%Y:%H:%M:%S')
        
    def log(self, message):
        self.logger.info(message)
        
    def add_time(self, initial_datetime_str, hours=0, minutes=0, seconds=0):
        delta = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return self.get_new_datetime(initial_datetime_str,delta)
    
    def subtract_time(self, initial_datetime_str, hours=0, minutes=0, seconds=0):
        delta = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return self.get_new_datetime(initial_datetime_str, -delta)
        
    def get_new_datetime(self,initial_datetime_str,delta):
        # Convert the initial datetime string to a datetime object
        initial_datetime = datetime.datetime.strptime(initial_datetime_str, '%m/%d/%Y:%H:%M:%S')
        
        # Adjust the datetime by the given delta
        new_datetime = initial_datetime + delta
        
        # Convert the new datetime object back to the desired string format
        return new_datetime.strftime('%m/%d/%Y:%H:%M:%S')
    
    def round_to_next_rule_frequency(self, rule_frequency):
        """
        Round the current fake datetime to the next rule frequency.
        :param rule_frequency: Rule frequency in minutes.
        """
        fake_now = self.get_fake_current_datetime()
        split_fake_now = fake_now.split(':')
        fake_now = self.add_time(fake_now, minutes=(int(rule_frequency) - int(split_fake_now[2])) % int(rule_frequency))
        fake_now = self.subtract_time(fake_now, seconds=int(split_fake_now[3]))
        self.set_fake_current_datetime(fake_now)
        
    
    def wait_til_next_rule_frequency(self, rule_frequency):
        """
        Wait until the current datetime is rounded to the next rule frequency.
        :param rule_frequency: Rule frequency in minutes.
        """
        self.log(f"waiting for next measurement")
        fake_now = self.get_real_current_datetime()
        split_fake_now = fake_now.split(':')
        while ((int(split_fake_now[2])+1) % int(rule_frequency) != 0) or (int(split_fake_now[3]) < (60 - (rule_frequency * 60) //15)):
            fake_now = self.get_real_current_datetime()
            split_fake_now = fake_now.split(':')            
            time.sleep(1)
            
if __name__ == "__main__":
    manager = MockedDatetimeManager(fake_start_datetime=datetime.datetime(2023, 1, 1, 12, 0, 0), log_file_path="test.log")
    for i in range(10):
        manager.log(manager.get_fake_current_datetime())
        manager.log("This is a test message.")
        new_time = manager.add_time(manager.get_fake_current_datetime(), seconds=21.4)
        manager.set_fake_current_datetime(new_time)
        manager.log(manager.get_fake_current_datetime())
    manager.round_to_next_rule_frequency(5)
    manager.log(manager.get_fake_current_datetime())