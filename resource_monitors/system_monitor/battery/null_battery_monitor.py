from typing import TextIO

import pandas as pd

from resource_monitors.system_monitor.battery.battery_monitor_interface import AbstractBatteryMonitor


class NullBatteryMonitor(AbstractBatteryMonitor):
    def check_if_battery_plugged(self):
        pass

    def save_battery_stat(self, battery_df: pd.DataFrame, time_interval: float):
        pass

    def save_general_battery(self, f: TextIO):
        pass

    def is_battery_too_low(self, battery_df: pd.DataFrame) -> bool:
        return False
