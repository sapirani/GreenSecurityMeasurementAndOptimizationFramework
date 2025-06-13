import shlex
import subprocess
import threading
from abc import abstractmethod
from threading import Thread

import psutil

from program_parameters import DEFAULT_SCREEN_TURNS_OFF_TIME, DEFAULT_TIME_BEFORE_SLEEP_MODE


class AbstractOSFuncs:
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
    def get_computer_info(self, is_inside_container: bool):
        pass

    @abstractmethod
    def get_page_faults(self, psutil_process):
        pass

    @staticmethod
    def get_hostname():
        import socket
        return socket.gethostname()

    @staticmethod
    def popen(command, find_child_id_func, should_use_powershell, is_posix, should_find_child_id=False, f_stdout=subprocess.PIPE, f_stderr=subprocess.PIPE):
        print(command)
        def process_obj_and_pid(command_lst):
            p = subprocess.Popen(command_lst, stdout=f_stdout, stderr=f_stderr)
            pid = p.pid

            if should_use_powershell or should_find_child_id:
                pid = find_child_id_func(p, is_posix)
                print(pid)
                if pid is not None and not should_use_powershell:
                    p = psutil.Process(pid)
            return p, pid

        if should_use_powershell:
            command = ["powershell", "-Command", command]
        else:
            # command = shlex.split(commnd, posix=is_posix)
            command = list(map(lambda s: s.strip('"'), shlex.split(command, posix=is_posix)))

        try:
            return process_obj_and_pid(command)
        except FileNotFoundError as e:
            if command[0] == "python":
                command[0] = "python3"
                return process_obj_and_pid(command)
            else:
                raise e

    @staticmethod
    def run(command, should_use_powershell, is_posix, f=subprocess.PIPE):
        if should_use_powershell:
            command = ["powershell", "-Command", command]
        else:
            # command = shlex.split(command, posix=is_posix)
            command = list(map(lambda s: s.strip('"'), shlex.split(command, posix=is_posix)))

        try:
            return subprocess.run(command,stdout=f)
        except FileNotFoundError as e:
            if command[0] == "python":
                command[0] = "python3"
                return subprocess.run(command, stdout=f)
            else:
                raise e

    def wait_for_measurement_termination(self, measurement_thread: Thread, done_scanning_event: threading.Event):
        measurement_thread.join()

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

    def is_posix(self):
        pass

    @abstractmethod
    def get_container_total_cpu_usage(self) -> float:
        pass

    @abstractmethod
    def get_container_total_memory_usage(self) -> tuple[float, float]:
        pass
