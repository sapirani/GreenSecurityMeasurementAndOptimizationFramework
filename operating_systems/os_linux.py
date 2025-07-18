import logging
import platform
import subprocess
import os
import signal
import psutil

from utils.general_consts import MINUTE, NEVER_GO_TO_SLEEP_MODE, YES_BUTTON, NO_BUTTON, PowerPlan
from operating_systems.abstract_operating_system import AbstractOSFuncs
from program_parameters import DEFAULT_SCREEN_TURNS_OFF_TIME, DEFAULT_TIME_BEFORE_SLEEP_MODE, power_plan
from resource_monitors.container_monitor.linux_resources.total_cpu_usage import LinuxContainerCPUReader
from resource_monitors.container_monitor.linux_resources.total_memory_usage import LinuxContainerMemoryReader

logger = logging.getLogger("measurements_logger")


class LinuxOS(AbstractOSFuncs):
    def __init__(self, is_inside_container: bool):
        self.__container_cpu_usage_reader = LinuxContainerCPUReader() if is_inside_container else None
        self.__container_memory_usage_reader = LinuxContainerMemoryReader() if is_inside_container else None

    @staticmethod
    def get_value_of_terminal_res(res):
        res_lst = res.stdout.decode("utf-8").strip().split("\n")
        return list(map(lambda res_line: res_line[res_line.rfind(":") + 2:].strip(), res_lst))

    def get_computer_info(self, is_inside_container: bool):
        if is_inside_container:
            return f"results_{self.get_hostname()}"

        hardware_info_res = subprocess.run("dmidecode | grep -A3 '^System Information' | grep Manufacturer",
                                           capture_output=True, shell=True)

        if hardware_info_res.returncode != 0:
            print(f"Warning! An error occurred while getting computer manufacturer: {hardware_info_res.stderr}")
            print("Warning! Ensure that the parameter is_inside_container is set to True if you run inside container")

        manufacturer, = LinuxOS.get_value_of_terminal_res(hardware_info_res)

        if manufacturer:
            return f"results_{manufacturer}_{platform.system()}_{platform.release()}"
        return f"results_{platform.system()}_{platform.release()}"

    def change_sleep_and_turning_screen_off_settings(self, screen_time=DEFAULT_SCREEN_TURNS_OFF_TIME,
                                                     sleep_time=DEFAULT_TIME_BEFORE_SLEEP_MODE):
        # avoid turning the screen off (avoid suspend)
        result_screen = subprocess.run(f'sudo -H -u $SUDO_USER DISPLAY:=0 DBUS_SESSION_BUS_ADDRESS='
                                       f'unix:path=/run/user/$SUDO_UID/bus gsettings set org.gnome.desktop.session '
                                       f'idle-delay {screen_time * MINUTE}',
                                       capture_output=True, shell=True)

        if result_screen.returncode != 0:
            raise Exception(f'An error occurred while changing screen settings', result_screen.stderr)

        # avoid dimming when inactive
        res_dimming = subprocess.run(f'sudo -H -u $SUDO_USER DISPLAY:=0 DBUS_SESSION_BUS_ADDRESS='
                                     f'unix:path=/run/user/$SUDO_UID/bus '
                                     f'gsettings set org.gnome.settings-daemon.plugins.power '
                                     f'idle-dim {"false" if sleep_time == NEVER_GO_TO_SLEEP_MODE else "true"}',
                                     capture_output=True, shell=True)

        if res_dimming.returncode != 0:
            raise Exception(f'An error occurred while changing dimming settings', res_dimming.stderr)

        # avoid from going to sleep
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

        battery_capacity_string, voltage_string = LinuxOS.get_value_of_terminal_res(res)

        battery_capacity = float(battery_capacity_string.split()[0]) * 1000
        voltage = float(voltage_string.split()[0]) * 1000

        battery_df.loc[len(battery_df.index)] = [
                time_interval,
                battery_percent,
                battery_capacity,
                voltage
            ]

        logger.info(
            "Battery measurements",
            extra={
                "battery_percent": battery_percent,
                "battery_remaining_capacity_mWh": battery_capacity,
                "battery_voltage_mV": voltage
            }
        )

    def change_power_plan(self, name, identifier):
        # this is the command to switch to performance plan
        if identifier is None:
            raise Exception(f'The power plan "{name}" is not supported in Linux')

        res = subprocess.run(f"echo {identifier} | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
                             capture_output=True, shell=True)

        if res.returncode != 0:
            raise Exception(f'An error occurred while changing power plan', res.stderr)

        # "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"

    def get_page_faults(self, psutil_process):
        # this is the command to switch to performance plan
        res = subprocess.run(f"ps -o min_flt,maj_flt {psutil_process.pid}",
                             capture_output=True, shell=True)

        if res.returncode != 0 and psutil.pid_exists(psutil_process.pid):
            raise ChildProcessError(f'An error occurred while getting process {psutil_process.pid} page faults', res.stderr)

        output_lines = res.stdout.decode("utf-8").strip().split("\n")

        if len(output_lines) < 2:
            return 0

        faults_res = output_lines[1].split()
        minor_faults, major_faults = int(faults_res[0].strip()), int(faults_res[1].strip())
        return minor_faults + major_faults

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

    def is_posix(self):
        return True

    def get_container_total_cpu_usage(self) -> float:
        if self.__container_cpu_usage_reader is None:
            raise ValueError("Can't call this method when not inside container")
        return self.__container_cpu_usage_reader.get_cpu_percent()

    def get_container_number_of_cores(self) -> float:
        if self.__container_cpu_usage_reader is None:
            raise ValueError("Can't call this method when not inside container")
        return self.__container_cpu_usage_reader.get_number_of_cpu_cores()

    def get_container_total_memory_usage(self) -> tuple[float, float]:
        if self.__container_memory_usage_reader is None:
            raise ValueError("Can't call this method when not inside container")
        usage_in_bytes = self.__container_memory_usage_reader.get_memory_usage_bytes()
        usage_percent = self.__container_memory_usage_reader.get_memory_usage_percent()
        return usage_in_bytes, usage_percent

    def kill_process_gracefully(self, process_pid: int):
        """
        The process might provide a custom handler function for signal.SIGINT
        """
        os.kill(process_pid, signal.SIGINT)
