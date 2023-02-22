import ctypes
import platform
import subprocess
from abc import ABC, abstractmethod

from general_consts import pc_types, GB, physical_memory_types, disk_types, NEVER_TURN_SCREEN_OFF, \
    NEVER_GO_TO_SLEEP_MODE, MINUTE, YES_BUTTON, NO_BUTTON, PowerPlan
from powershell_helper import get_powershell_result_list_format
from program_parameters import DEFAULT_SCREEN_TURNS_OFF_TIME, DEFAULT_TIME_BEFORE_SLEEP_MODE, power_plan


class OSFuncsInterface:
    def is_tamper_protection_enabled(self):
        """
        Needed only in Windows
        :return: True if tamper protection is enabled, False otherwise
        """
        return True

    def change_real_time_protection(self, should_disable=True):
        """
        Needed only in Windows
        :param should_disable: determine if real time protection will be turned on or off
        :return: None
        """
        pass

    def init_thread(self):
        """
        probably only needed in Windows
        :return: None
        """
        pass

    def save_battery_capacity(self, f):
        pass

    def save_system_information(self, f):   # TODO
        pass

    def save_physical_memory(self, f):      # TODO
        pass

    def save_disk_information(self, f):     # TODO
        pass

    def message_box(self, title, text, style):
        pass

    @abstractmethod
    def insert_battery_state_to_df(self, battery_df, time_interval, battery_percent): # did not work in ubuntu
        pass

    @abstractmethod
    def get_computer_info(self):
        pass

    @staticmethod
    def popen(command, find_child_id_func, should_use_powershell):
        def process_obj_and_pid(command_lst):
            p = subprocess.Popen(command_lst, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if should_use_powershell:
                return p, find_child_id_func(p.pid)
            return p, p.pid

        if should_use_powershell:
            command = ["powershell", "-Command", command]
        else:
            command = list(map(lambda s: s.strip('"'), command.split()))

        try:
            return process_obj_and_pid(command)
        except FileNotFoundError as e:
            if command[0] == "python":
                command[0] = "python3"
                return process_obj_and_pid(command)
            else:
                raise e

    @abstractmethod
    # TODO: make balance the default
    def change_power_plan(self, name, identifier):
        pass

    @abstractmethod
    def get_chosen_power_plan_identifier(self):
        pass

    def get_default_power_plan_name(self):
        pass

    def get_default_power_plan_identifier(self):
        pass

    @abstractmethod
    def change_sleep_and_turning_screen_off_settings(self, screen_time=DEFAULT_SCREEN_TURNS_OFF_TIME,
                                                     sleep_time=DEFAULT_TIME_BEFORE_SLEEP_MODE):
        pass


class WindowsOS(OSFuncsInterface):
    def __init__(self):
        import wmi
        self.c = wmi.WMI()
        self.t = wmi.WMI(moniker="//./root/wmi")

    @classmethod
    def save_antivirus_version(cls, f, program_name):
        result = subprocess.run(["powershell", "-Command", "Get-MpComputerStatus | Select AMEngineVersion,"
                                                           " AMProductVersion, AMServiceVersion | Format-List"],
                                capture_output=True)
        if result.returncode != 0:
            raise Exception(f'Could not get {program_name} version', result.stderr)

        import powershell_helper
        version_dict = powershell_helper.get_powershell_result_list_format(result.stdout)[0]
        f.write(f"Anti Malware Engine Version: {version_dict['AMEngineVersion']}\n")
        f.write(f"Anti Malware Client Version: {version_dict['AMProductVersion']}\n")
        f.write(f"Anti Malware Service Version: {version_dict['AMServiceVersion']}\n\n")

    def init_thread(self):
        import pythoncom
        pythoncom.CoInitialize()

    """def popen(self, command):
        command_list = list(map(lambda s: s.strip('"'), command.split()))
        return subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #return subprocess.Popen(["powershell", "-Command", command],
        #                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
"""
    def get_computer_info(self):
        wmi_system = self.c.Win32_ComputerSystem()[0]

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

    def change_power_plan(self, name, identifier):
        result = subprocess.run(["powershell", "-Command", "powercfg /s " + identifier], capture_output=True)
        if result.returncode != 0:
            raise Exception(f'An error occurred while switching to the power plan: {name}', result.stderr)

    def get_chosen_power_plan_identifier(self):
        return power_plan[1]

    def get_default_power_plan_name(self):
        return PowerPlan.BALANCED[0]

    def get_default_power_plan_identifier(self):
        return PowerPlan.BALANCED[1]

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
        batts1 = self.c.CIM_Battery(Caption='Portable Battery')
        for i, b in enumerate(batts1):
            f.write('Battery %d Design Capacity: %d mWh\n' % (i, b.DesignCapacity or 0))

        batts = self.t.ExecQuery('Select * from BatteryFullChargedCapacity')
        for i, b in enumerate(batts):
            f.write('Battery %d Fully Charged Capacity: %d mWh\n' % (i, b.FullChargedCapacity))

    def save_system_information(self, f):
        wmi_system = self.c.Win32_ComputerSystem()[0]

        f.write(f"PC Type: {pc_types[wmi_system.PCSystemType]}\n")
        f.write(f"Manufacturer: {wmi_system.Manufacturer}\n")
        f.write(f"System Family: {wmi_system.SystemFamily}\n")
        f.write(f"Model: {wmi_system.Model}\n")

    def save_physical_memory(self, f):
        wmi_physical_memory = self.c.Win32_PhysicalMemory()

        for physical_memory in wmi_physical_memory:
            f.write(f"\nName: {physical_memory.Tag}\n")
            f.write(f"Manufacturer: {physical_memory.Manufacturer}\n")
            f.write(f"Capacity: {int(physical_memory.Capacity) / GB}\n")
            f.write(f"Memory Type: {physical_memory_types[physical_memory.SMBIOSMemoryType]}\n")
            f.write(f"Speed: {physical_memory.Speed} MHz\n")

    def save_disk_information(self, f):
        wmi_logical_disks = self.c.Win32_LogicalDisk()
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

    def message_box(self, title, text, style):
        return ctypes.windll.user32.MessageBoxW(0, text, title, style)


class LinuxOS(OSFuncsInterface):
    @staticmethod
    def get_value_of_terminal_res(res):
        res_lst = res.stdout.decode("utf-8").strip().split("\n")
        return list(map(lambda res_line: res_line[res_line.rfind(":") + 2:].strip(), res_lst))

    def get_computer_info(self):
        res = subprocess.run("dmidecode | grep -A3 '^System Information' | grep Manufacturer",
                             capture_output=True, shell=True)

        if res.returncode != 0:
            raise Exception(f'An error occurred while changing screen settings', res.stderr)

        manufacturer, = LinuxOS.get_value_of_terminal_res(res)

        return f"{manufacturer} {platform.system()} {platform.release()}"

    def change_sleep_and_turning_screen_off_settings(self, screen_time=DEFAULT_SCREEN_TURNS_OFF_TIME,
                                                     sleep_time=DEFAULT_TIME_BEFORE_SLEEP_MODE):

        result_screen = subprocess.run(f'sudo -H -u $SUDO_USER DISPLAY:=0 DBUS_SESSION_BUS_ADDRESS='
                                       f'unix:path=/run/user/$SUDO_UID/bus gsettings set org.gnome.desktop.session '
                                       f'idle-delay {screen_time * MINUTE}',
                                       capture_output=True, shell=True)

        if result_screen.returncode != 0:
            raise Exception(f'An error occurred while changing screen settings', result_screen.stderr)

        result_sleep = subprocess.run(f'sudo systemctl {"mask" if sleep_time == NEVER_GO_TO_SLEEP_MODE else "unmask"} '
                                      f'sleep.target suspend.target hibernate.target hybrid-sleep.target',
                                      capture_output=True, shell=True)

        if result_sleep.returncode != 0:
            raise Exception(f'An error occurred while changing sleep mode', result_sleep.stderr)

        # is the following command necessary??? suppose to control the idle time before going to sleep
        # gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout 400

    def message_box(self, title, text, style):
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()

        canvas1 = tk.Canvas(root, width=100, height=50)
        canvas1.pack()

        msg_box = tk.messagebox.askquestion(title, text, icon='warning')
        root.destroy()

        if msg_box == 'yes':
            return YES_BUTTON
        return NO_BUTTON

    def insert_battery_state_to_df(self, battery_df, time_interval, battery_percent):
        res = subprocess.run("upower -i /org/freedesktop/UPower/devices/battery_BAT0 | grep -E 'energy:|voltage'",
                             capture_output=True, shell=True)

        if res.returncode != 0:
            raise Exception(f'An error occurred while reading battery capacity and voltage', res.stderr)

        battery_capacity, voltage = LinuxOS.get_value_of_terminal_res(res)

        battery_df.loc[len(battery_df.index)] = [
                time_interval,
                battery_percent,
                float(battery_capacity.split()[0]) * 1000,
                float(voltage.split()[0]) * 1000
            ]

    def change_power_plan(self, name, identifier):
        # this is the command to switch to performance plan
        if identifier is None:
            raise Exception(f'The power plan "{name}" is not supported in Linux')

        res = subprocess.run(f"echo {identifier} | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
                             capture_output=True, shell=True)

        if res.returncode != 0:
            raise Exception(f'An error occurred while changing power plan', res.stderr)

        # "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"

    def get_chosen_power_plan_identifier(self):
        return power_plan[2]

    def get_default_power_plan_name(self):
        return PowerPlan.POWER_SAVER[0]

    def get_default_power_plan_identifier(self):
        return PowerPlan.POWER_SAVER[2]

    def save_battery_capacity(self, f):
        res = subprocess.run("upower -i /org/freedesktop/UPower/devices/battery_BAT0 |"
                             " grep -E 'energy-full-design|energy-empty'",
                             capture_output=True, shell=True)

        if res.returncode != 0:
            raise Exception(f'An error occurred while saving general battery capacity', res.stderr)

        empty_capacity, full_capacity = LinuxOS.get_value_of_terminal_res(res)

        f.write(f'Battery Design Capacity: {float(empty_capacity.split()[0]) * 1000} mWh\n')
        f.write(f'Battery Fully Charged Capacity: {float(full_capacity.split()[0]) * 1000} mWh\n')
