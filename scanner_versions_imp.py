import time
import psutil


class FullScanner:
    def __init__(self, running_os):
        self.running_os = running_os

    def check_if_battery_plugged(self):
        battery = psutil.sensors_battery()
        if battery is not None and battery.power_plugged:  # ensure that charging cable is unplugged in laptop
            raise Exception("Unplug charging cable during measurements!")

    def save_battery_stat(self, battery_df, time_interval):
        """_summary_: take battery information and append it to a dataframe

        Raises:
            Exception: if the computer is charging or using desktop computer cant get battery information
        """
        # Fetch the battery information
        battery = psutil.sensors_battery()
        if battery is None:  # if desktop computer (has no battery)
            return

        if battery.power_plugged:
            raise Exception("Unplug charging cable during measurements!")

        self.running_os.insert_battery_state_to_df(battery_df, time_interval, battery.percent)

    def save_general_battery(self, f):
        """
        This function writes battery info to a file.
        On laptop devices, charger must be unplugged!
        :param f: text file to write the battery info
        """
        battery = psutil.sensors_battery()
        if battery is None:  # if desktop computer (has no battery)
            return

        if battery.power_plugged:
            raise Exception("Unplug charging cable during measurements!")

        f.write("----Battery----\n")

        self.running_os.save_battery_capacity(f)
        
    def calc_time_interval(self, starting_time):
        """
        :return: the time passed since starting the program
        """
        return time.time() - starting_time

    def scan_sleep(self,sec):
        time.sleep(sec)


class LiteScanner(FullScanner):
    def check_if_battery_plugged(self):
        pass

    def save_battery_stat(self, battery_df, time_interval):
        pass

    def save_general_battery(self, f):
        pass

    def calc_time_interval(self, starting_time):
        """
        :return: the time passed since starting the program
        """
        return time.time()

    def scan_sleep(self, sec):
        time.sleep(0)


class WithoutBatteryScanner(FullScanner):
    def check_if_battery_plugged(self):
        pass

    def save_battery_stat(self, battery_df, time_interval):
        pass

    def save_general_battery(self, f):
        pass

