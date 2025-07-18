from utils.general_consts import BatteryMonitorType
from operating_systems.abstract_operating_system import AbstractOSFuncs
from resource_monitors.system_monitor.battery.battery_monitor import BatteryMonitor
from resource_monitors.system_monitor.battery.null_battery_monitor import NullBatteryMonitor


def battery_monitor_factory(battery_monitor_type: BatteryMonitorType, running_os: AbstractOSFuncs):
    if battery_monitor_type == BatteryMonitorType.FULL:
        return BatteryMonitor(running_os)
    elif battery_monitor_type == BatteryMonitorType.WITHOUT_BATTERY:
        return NullBatteryMonitor()

    raise Exception("Selected process monitor type is not supported")
