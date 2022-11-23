import subprocess
import time

import powershell_helper
from general_consts import *


class ProgramInterface:
    def get_program_name(self) -> str:
        pass

    def get_process_name(self) -> str:
        pass

    def get_command(self) -> str:
        pass

    def path_adjustments(self) -> str:
        return ""

    def general_information_before_measurement(self, f):
        pass

    def find_child_id(self, process_pid) -> int | None:
        # result_screen = subprocess.run(["powershell", "-Command", f'Get-WmiObject Win32_Process -Filter "ParentProcessID={process_pid}" | Select ProcessID'],
        #                               capture_output=True)
        # if result_screen.returncode != 0:
        #    raise Exception(result_screen.stderr)

        # scanning_process_id = int(str(result_screen.stdout).split("\\r\\n")[3: -3][0].strip())
        # print(scanning_process_id)

        try:
            children = []
            for i in range(300):
                children = psutil.Process(process_pid).children()
                if len(children) != 0:
                    break
                time.sleep(0.1)

        except psutil.NoSuchProcess:
            return None

        if len(children) != 1:
            return None

        return children[0].pid

    # TODO: add no scan
    def calc_measurement_time(self):
        pass


class AntivirusProgram(ProgramInterface):
    def __init__(self, scan_type, custom_scan_path):
        if custom_scan_path == "" or custom_scan_path == '""':
            self.custom_scan_path = None
        else:
            self.custom_scan_path = custom_scan_path
        self.scan_type = scan_type

    def get_program_name(self):
        return "Windows Defender"

    def get_process_name(self) -> str:
        return "MsMpEng.exe"

    def get_command(self) -> str:
        custom_scan_query = "" if self.custom_scan_path is None else f" -ScanPath {self.custom_scan_path}"
        return f"Start-MpScan -ScanType {self.scan_type}" + custom_scan_query
        #return '"C:\\ProgramData\\Microsoft\\Windows Defender\\Platform\\4.18.2210.6-0\\MpCmdRun.exe" -Scan -ScanType 1'

    def path_adjustments(self):
        return self.scan_type

    def general_information_before_measurement(self, f):
        if self.scan_type == ScanType.CUSTOM_SCAN:
            f.write(f'Scan Path: {self.custom_scan_path}\n\n')

        result = subprocess.run(["powershell", "-Command", "Get-MpComputerStatus | Select AMEngineVersion,"
                                                           " AMProductVersion, AMServiceVersion | Format-List"],
                                capture_output=True)
        if result.returncode != 0:
            raise Exception(f'Could not get {self.get_program_name()} version', result.stderr)

        version_dict = powershell_helper.get_powershell_result_list_format(result.stdout)[0]
        f.write(f"Anti Malware Engine Version: {version_dict['AMEngineVersion']}\n")
        f.write(f"Anti Malware Client Version: {version_dict['AMProductVersion']}\n")
        f.write(f"Anti Malware Service Version: {version_dict['AMServiceVersion']}\n\n")

    def find_child_id(self, process_pid):
        for i in range(3):  # try again and again
            for proc in psutil.process_iter():
                if proc.name() == self.get_process_name():
                    return proc.pid

        return None


class DummyAntivirusProgram(ProgramInterface):
    def __init__(self, scan_path):
        self.scan_path = scan_path

    def get_program_name(self):
        return "Dummy Antivirus"

    def get_command(self) -> str:
        return f"python FilesReader.py {self.scan_path}"

    def general_information_before_measurement(self, f):
        f.write(f'Scan Path: {self.scan_path}\n\n')


class IDSProgram(ProgramInterface):
    def __init__(self, ids_type, interface_name, log_dir, installation_dir="C:\Program Files"):
        self.ids_type = ids_type
        self.interface_name = interface_name
        self.log_dir = log_dir
        self.installation_dir = installation_dir

    def get_program_name(self):
        return "IDS"

    def get_command(self):
        return rf"& '{self.installation_dir}\{self.ids_type}\{self.ids_type.lower()}.exe' -i {self.interface_name} -l '{self.installation_dir}\{self.ids_type}\{self.log_dir}'"


class NoScanProgram(ProgramInterface):
    def get_program_name(self) -> str:
        return "No Scan"