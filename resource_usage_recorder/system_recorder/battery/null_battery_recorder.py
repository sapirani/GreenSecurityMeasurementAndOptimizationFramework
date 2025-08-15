from dataclasses import dataclass
from typing import TextIO, Optional

from resource_usage_recorder import MetricResult
from resource_usage_recorder.system_recorder.battery.abstract_battery_usage_recorder import AbstractBatteryUsageRecorder


@dataclass
class NullBatteryResults(MetricResult):
    battery_remaining_capacity_mWh = None


class NullBatteryUsageRecorder(AbstractBatteryUsageRecorder):
    def check_if_battery_plugged(self):
        pass

    def get_current_metrics(self) -> NullBatteryResults:
        return NullBatteryResults()

    def save_general_battery(self, f: TextIO):
        pass

    def is_battery_too_low(self, battery_capacity: Optional[float]) -> bool:
        return False
