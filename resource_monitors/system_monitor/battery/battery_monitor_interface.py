import abc


class BatteryMonitorInterface(abc.ABC):
    @abc.abstractmethod
    def check_if_battery_plugged(self):
        pass

    @abc.abstractmethod
    def save_battery_stat(self, battery_df, time_interval):
        """_summary_: take battery information and append it to a dataframe

        Raises:
            Exception: if the computer is charging or using desktop computer cant get battery information
        """
        pass

    @abc.abstractmethod
    def save_general_battery(self, f):
        """
        This function writes battery info to a file.
        On laptop devices, charger must be unplugged!
        :param f: text file to write the battery info
        """
        pass

    @abc.abstractmethod
    def is_battery_too_low(self, battery_df):
        pass
