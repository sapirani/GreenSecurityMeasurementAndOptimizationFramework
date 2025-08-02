from typing import TextIO, Optional

import pandas as pd

from resource_usage_recorder.system_recorder.battery.battery_usage_recorder import SystemBatteryResults
from resource_usage_recorder.system_recorder.battery.abstract_battery_usage_recorder import AbstractBatteryUsageRecorder


class NullBatteryUsageRecorder(AbstractBatteryUsageRecorder):
    def check_if_battery_plugged(self):
        pass

    def get_current_metrics(self) -> SystemBatteryResults:
        pass

    def save_general_battery(self, f: TextIO):
        pass

    def is_battery_too_low(self, battery_capacity: Optional[float]) -> bool:
        return False
