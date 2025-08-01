import dataclasses
from typing import TextIO, Optional
import psutil

from operating_systems.abstract_operating_system import AbstractOSFuncs
from resource_monitors import MetricResult, MetricRecorder
from resource_monitors.system_monitor.battery.battery_monitor_interface import AbstractBatteryMonitor


@dataclasses.dataclass
class SystemBatteryResults(MetricResult):
    battery_percent: float
    battery_remaining_capacity_mWh: float
    battery_voltage_mV: float


class SystemBatteryUsageRecorder(MetricRecorder, AbstractBatteryMonitor):
    def __init__(self, running_os: AbstractOSFuncs):
        self.running_os = running_os

    def check_if_battery_plugged(self):
        battery = psutil.sensors_battery()
        if battery is not None and battery.power_plugged:  # ensure that charging cable is unplugged in laptop
            raise Exception("Unplug charging cable during measurements!")

    def get_current_metrics(self):
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

        mwh, voltage = self.running_os.get_battery_capacity_and_voltage()
        return SystemBatteryResults(
            battery_percent=battery.percent,
            battery_remaining_capacity_mWh=mwh,
            battery_voltage_mV=voltage
        )

    # TODO: REMOVE THIS FUNCTIONALITY INTO A DEDICATED CLASS?
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

    def is_battery_too_low(self, battery_capacity: Optional[float]) -> bool:
        if battery_capacity is None:
            return False

        return battery_capacity <= 2500
