from typing import TextIO

import pandas as pd
import psutil

from utils.general_consts import BatteryColumns
from resource_monitors.system_monitor.battery.battery_monitor_interface import AbstractBatteryMonitor


class BatteryMonitor(AbstractBatteryMonitor):
    def __init__(self, running_os):
        self.running_os = running_os

    def check_if_battery_plugged(self):
        battery = psutil.sensors_battery()
        if battery is not None and battery.power_plugged:  # ensure that charging cable is unplugged in laptop
            raise Exception("Unplug charging cable during measurements!")

    def save_battery_stat(self, battery_df: pd.DataFrame, time_interval: float):
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

    def save_general_battery(self, f: TextIO):
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

    def is_battery_too_low(self, battery_df: pd.DataFrame) -> bool:
        if len(battery_df) == 0:
            return False

        current_mwh = battery_df.iloc[len(battery_df) - 1].at[BatteryColumns.CAPACITY]
        return current_mwh <= 9000
