import ctypes
import logging
import platform
import subprocess
import threading
from threading import Thread
import os
import signal
from typing_extensions import override
from utils import general_functions
from utils.general_consts import PowerPlan, pc_types, GB, physical_memory_types, disk_types
from utils.general_functions import get_powershell_result_list_format
from operating_systems.abstract_operating_system import AbstractOSFuncs
from program_parameters import power_plan, DEFAULT_SCREEN_TURNS_OFF_TIME, DEFAULT_TIME_BEFORE_SLEEP_MODE

logger = logging.getLogger("measurements_logger")


class WindowsOS(AbstractOSFuncs):
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

        version_dict = general_functions.get_powershell_result_list_format(result.stdout)[0]
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

    def get_computer_info(self, is_inside_container: bool):
        wmi_system = self.c.Win32_ComputerSystem()[0]

        if is_inside_container:
            return f"results_{self.get_hostname()}"

        hardware_info = f"{wmi_system.Manufacturer} {wmi_system.SystemFamily} {wmi_system.Model}"

        return f"results_{hardware_info}_{platform.system()}_{platform.release()}"

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

    def get_page_faults(self, psutil_process):
        return psutil_process.memory_info().num_page_faults

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

            logger.info(
                "Battery measurements",
                extra={
                    "battery_percent": battery_percent,
                    "battery_remaining_capacity_mWh": b.RemainingCapacity,
                    "battery_voltage_mV": b.Voltage
                }
            )

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

    def is_posix(self):
        return False

    def get_container_total_cpu_usage(self) -> float:
        raise NotImplementedError("Not implemented total cpu for windows container")

    def get_container_number_of_cores(self) -> float:
        raise NotImplementedError("Not implemented number of cores for windows container")

    def get_container_total_memory_usage(self) -> tuple[float, float]:
        raise NotImplementedError("Not implemented total memory for windows container")

    @override
    def wait_for_thread_termination(self, thread: Thread, done_scanning_event: threading.Event) -> None:
        """
        Since signal handling in windows cannot be interrupted, while waiting to the measurement thread we cannot
        actively stop the program (for example, when using CTRL+C).
        Hence, we have no choice but avoid blocking, and give a chance for interruption by the signal
        ***Important***: you must ensure that done_scanning_event is set when measurement thread terminates
        (in any possible case, such as reaching predefined timeout, receiving CTRL+C,
        depleting predefined threshold of battery and so on...)
        """
        while not done_scanning_event.wait(timeout=2):
            if not thread.is_alive():
                break

        thread.join()

    @override
    def wait_for_process_termination(self, process: subprocess.Popen, done_scanning_event: threading.Event) -> int:
        """
        Since signal handling in windows cannot be interrupted, while waiting to the process we cannot
        actively stop the program (for example, when using CTRL+C).
        Hence, we have no choice but avoid blocking, and give a chance for interruption by the signal
        ***Important***: you must ensure that done_scanning_event is set when measurement thread terminates
        (in any possible case, such as reaching predefined timeout, receiving CTRL+C,
        depleting predefined threshold of battery and so on...)
        """
        # TODO: rename done_scanning_event after unifying with should_scan
        while not done_scanning_event.wait(timeout=2):
            if process.poll() is not None:
                break

        return process.wait()

    @override
    def kill_process_gracefully(self, process_pid: int):
        """
        The process might provide a custom handler function for signal.SIGBREAK
        """
        os.kill(process_pid, signal.CTRL_BREAK_EVENT)
