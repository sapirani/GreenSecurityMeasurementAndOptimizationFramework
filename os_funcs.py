import subprocess
from abc import ABC, abstractmethod

from powershell_helper import get_powershell_result_list_format
from program_parameters import DEFAULT_SCREEN_TURNS_OFF_TIME, DEFAULT_TIME_BEFORE_SLEEP_MODE


class OSFuncsInterface:
    def is_tamper_protection_enabled(self):
        return True

    def change_real_time_protection(self, should_disable=True):
        pass

    def init_thread(self):
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
    def init_thread(self):
        import pythoncom
        pythoncom.CoInitialize()

    def popen(self, command):
        return subprocess.Popen(["powershell", "-Command", command],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

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


class LinuxOS(OSFuncsInterface):
    def popen(self, command):
        return subprocess.Popen(["/usr/bin/gnome-terminal", command],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
