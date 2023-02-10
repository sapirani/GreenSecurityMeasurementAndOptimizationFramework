import platform
import subprocess
from abc import ABC, abstractmethod

from general_consts import pc_types, GB, physical_memory_types, disk_types
from powershell_helper import get_powershell_result_list_format
from program_parameters import DEFAULT_SCREEN_TURNS_OFF_TIME, DEFAULT_TIME_BEFORE_SLEEP_MODE


class OSFuncsInterface:
    def is_tamper_protection_enabled(self):
        return True

    def change_real_time_protection(self, should_disable=True):
        pass

    def init_thread(self):
        pass

    def save_battery_capacity(self, f):
        pass

    def save_system_information(self, f):
        pass

    def save_physical_memory(self, f):
        pass

    def save_disk_information(self, f):
        pass

    @abstractmethod
    def insert_battery_state_to_df(self, battery_df, time_interval, battery_percent):
        pass

    @abstractmethod
    def get_computer_info(self):
        pass

    @abstractmethod
    def popen(self, command):
        pass

    @abstractmethod
    # make balance the default
    def change_power_plan(self, name, guid):
        pass

    @abstractmethod
    def change_sleep_and_turning_screen_off_settings(self, screen_time=DEFAULT_SCREEN_TURNS_OFF_TIME,
                                                     sleep_time=DEFAULT_TIME_BEFORE_SLEEP_MODE):
        pass


class WindowsOS(OSFuncsInterface):
    import wmi
    c = wmi.WMI()
    t = wmi.WMI(moniker="//./root/wmi")

    def init_thread(self):
        import pythoncom
        pythoncom.CoInitialize()

    def popen(self, command):
        return subprocess.Popen(["powershell", "-Command", command],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def get_computer_info(self):
        wmi_system = WindowsOS.c.Win32_ComputerSystem()[0]

        return f"{wmi_system.Manufacturer} {wmi_system.SystemFamily} {wmi_system.Model} " \
                        f"{platform.system()} {platform.release()}"

    def is_tamper_protection_enabled(self):
        """_summary_: tamper protection should be disabled for the program to work properly

            Raises:
                Exception: if could not check if tamper protection enabled

            Returns:
                _type_ : bool -- True if tamper protection is enabled, False otherwise
            """
        result = subprocess.run(
            ["powershell", "-Command", "Get-MpComputerStatus | Select IsTamperProtected | Format-List"],
            capture_output=True)
        if result.returncode != 0:
            raise Exception("Could not check if tamper protection enabled", result.stderr)

        # return bool(re.search("IsTamperProtected\s*:\sTrue", str(result.stdout)))
        return get_powershell_result_list_format(result.stdout)[0]["IsTamperProtected"] == "True"

    def change_power_plan(self, name, guid):
        result = subprocess.run(["powershell", "-Command", "powercfg /s " + guid], capture_output=True)
        if result.returncode != 0:
            raise Exception(f'An error occurred while switching to the power plan: {name}', result.stderr)

    def change_real_time_protection(self, should_disable=True):

        protection_mode = "1" if should_disable else "0"
        result = subprocess.run(["powershell", "-Command",
                                 f'Start-Process powershell -ArgumentList("Set-MpPreference -DisableRealTimeMonitoring {protection_mode}") -Verb runAs -WindowStyle hidden'],
                                capture_output=True)
        if result.returncode != 0:
            raise Exception("Could not change real time protection", result.stderr)

    def change_sleep_and_turning_screen_off_settings(self, screen_time=DEFAULT_SCREEN_TURNS_OFF_TIME,
                                                     sleep_time=DEFAULT_TIME_BEFORE_SLEEP_MODE):
        """_summary_ : change the sleep and turning screen off settings

        Args:
            screen_time :Defaults to DEFAULT_SCREEN_TURNS_OFF_TIME.
            sleep_time : Defaults to DEFAULT_TIME_BEFORE_SLEEP_MODE.
        """
        result_screen = subprocess.run(["powershell", "-Command", f"powercfg /Change monitor-timeout-dc {screen_time}"],
                                       capture_output=True)
        if result_screen.returncode != 0:
            raise Exception(f'An error occurred while changing turning off the screen to never', result_screen.stderr)

        result_sleep_mode = subprocess.run(
            ["powershell", "-Command", f"powercfg /Change standby-timeout-dc {sleep_time}"],
            capture_output=True)
        if result_sleep_mode.returncode != 0:
            raise Exception(f'An error occurred while disabling sleep mode', result_sleep_mode.stderr)

    def insert_battery_state_to_df(self, battery_df, time_interval, battery_percent):
        import wmi
        t = wmi.WMI(moniker="//./root/wmi")

        new_row_index = len(battery_df.index)

        for i, b in enumerate(t.ExecQuery('Select * from BatteryStatus where Voltage > 0')):
            battery_df.loc[new_row_index + i] = [
                time_interval,
                battery_percent,
                b.RemainingCapacity,
                b.Voltage
            ]

    def save_battery_capacity(self, f):
        batts1 = WindowsOS.c.CIM_Battery(Caption='Portable Battery')
        for i, b in enumerate(batts1):
            f.write('Battery %d Design Capacity: %d mWh\n' % (i, b.DesignCapacity or 0))

        batts = WindowsOS.t.ExecQuery('Select * from BatteryFullChargedCapacity')
        for i, b in enumerate(batts):
            f.write('Battery %d Fully Charged Capacity: %d mWh\n' % (i, b.FullChargedCapacity))

    def save_system_information(self, f):
        wmi_system = WindowsOS.c.Win32_ComputerSystem()[0]

        f.write(f"PC Type: {pc_types[wmi_system.PCSystemType]}\n")
        f.write(f"Manufacturer: {wmi_system.Manufacturer}\n")
        f.write(f"System Family: {wmi_system.SystemFamily}\n")
        f.write(f"Model: {wmi_system.Model}\n")

    def save_physical_memory(self, f):
        c = WindowsOS.wmi.WMI()
        wmi_physical_memory = c.Win32_PhysicalMemory()

        for physical_memory in wmi_physical_memory:
            f.write(f"\nName: {physical_memory.Tag}\n")
            f.write(f"Manufacturer: {physical_memory.Manufacturer}\n")
            f.write(f"Capacity: {int(physical_memory.Capacity) / GB}\n")
            f.write(f"Memory Type: {physical_memory_types[physical_memory.SMBIOSMemoryType]}\n")
            f.write(f"Speed: {physical_memory.Speed} MHz\n")

    def save_disk_information(self, f):
        wmi_logical_disks = WindowsOS.c.Win32_LogicalDisk()
        result = subprocess.run(["powershell", "-Command",
                                 "Get-Disk | Select FriendlyName, Manufacturer, Model,  PartitionStyle, NumberOfPartitions,"
                                 " PhysicalSectorSize, LogicalSectorSize, BusType | Format-List"], capture_output=True)
        if result.returncode != 0:
            raise Exception(f'An error occurred while getting disk information', result.stderr)

        logical_disks_info = get_powershell_result_list_format(result.stdout)

        result = subprocess.run(["powershell", "-Command",
                                 "Get-PhysicalDisk | Select FriendlyName, Manufacturer, Model, FirmwareVersion, MediaType"
                                 " | Format-List"], capture_output=True)
        if result.returncode != 0:
            raise Exception(f'An error occurred while getting disk information', result.stderr)

        physical_disks_info = get_powershell_result_list_format(result.stdout)

        f.write("\n----Physical Disk Information----\n")
        for physical_disk_info in physical_disks_info:
            f.write(f"\nName: {physical_disk_info['FriendlyName']}\n")
            f.write(f"Manufacturer: {physical_disk_info['Manufacturer']}\n")
            f.write(f"Model: {physical_disk_info['Model']}\n")
            f.write(f"Media Type: {physical_disk_info['MediaType']}\n")
            f.write(f"Disk Firmware Version: {physical_disk_info['FirmwareVersion']}\n")

        f.write("\n----Logical Disk Information----\n")
        try:
            for index, logical_disk_info in enumerate(logical_disks_info):
                f.write(f"\nName: {logical_disk_info['FriendlyName']}\n")
                f.write(f"Manufacturer: {logical_disk_info['Manufacturer']}\n")
                f.write(f"Model: {logical_disk_info['Model']}\n")
                f.write(f"Disk Type: {disk_types[wmi_logical_disks[index].DriveType]}\n")
                f.write(f"Partition Style: {logical_disk_info['PartitionStyle']}\n")
                f.write(f"Number Of Partitions: {logical_disk_info['NumberOfPartitions']}\n")
                f.write(f"Physical Sector Size: {logical_disk_info['PhysicalSectorSize']} bytes\n")
                f.write(f"Logical Sector Size: {logical_disk_info['LogicalSectorSize']}  bytes\n")
                f.write(f"Bus Type: {logical_disk_info['BusType']}\n")
                f.write(f"FileSystem: {wmi_logical_disks[index].FileSystem}\n")
        except Exception:
            pass


class LinuxOS(OSFuncsInterface):
    def popen(self, command):
        return subprocess.Popen(["/usr/bin/gnome-terminal", command],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
