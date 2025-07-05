from resource_monitors.system_monitor.battery.battery_monitor_interface import BatteryMonitorInterface


class NullBatteryMonitor(BatteryMonitorInterface):
    def check_if_battery_plugged(self):
        pass

    def save_battery_stat(self, battery_df, time_interval):
        pass

    def save_general_battery(self, f):
        pass

    def is_battery_too_low(self, mwh):
        return False
