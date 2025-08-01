from typing import TextIO, Optional

import pandas as pd

from resource_monitors.system_monitor.battery.battery_monitor_interface import AbstractBatteryMonitor


class NullBatteryMonitor(AbstractBatteryMonitor):
    def check_if_battery_plugged(self):
        pass

    def get_current_metrics(self, battery_df: pd.DataFrame, time_interval: float):
        pass

    def save_general_battery(self, f: TextIO):
        pass

    def is_battery_too_low(self, battery_capacity: Optional[float]) -> bool:
        return False
