import os
import subprocess
import time

import powershell_helper
from general_consts import *

from typing import Union


class ProgramInterface:
    def __init__(self):
        self.results_path = None
        self.processes_ids = None

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

    def should_use_powershell(self) -> bool:
        return False

    def set_results_dir(self, results_path):
        self.results_path = results_path

    def set_processes_ids(self, processes_ids):
        self.processes_ids = processes_ids

    ######## probably not needed anymore because we don't start powershell process if not needed in popen
    ######## so the id of the process we are interested in is the pid returned from popen process
    def find_child_id(self, process_pid) -> Union[int, None]:  #from python 3.10 - int | None:
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


class AntivirusProgram(ProgramInterface):
    def __init__(self, scan_type, custom_scan_path):
        super().__init__()
        if custom_scan_path == "" or custom_scan_path == '""':
            self.custom_scan_path = None
        else:
            self.custom_scan_path = custom_scan_path
        self.scan_type = scan_type

    def get_program_name(self):
        return "Windows Defender"

    def get_process_name(self) -> str:
        return "MsMpEng.exe"

    def should_use_powershell(self) -> bool:
        return True

    def get_command(self) -> Union[str, list]:
        custom_scan_query = "" if self.custom_scan_path is None else f" -ScanPath {self.custom_scan_path}"
        return f"Start-MpScan -ScanType {self.scan_type}" + custom_scan_query
        #return ["powershell", "-Command", f"Start-MpScan -ScanType {self.scan_type}" + custom_scan_query]
        #["powershell", "-Command", command]
        #return '"C:\\ProgramData\\Microsoft\\Windows Defender\\Platform\\4.18.2210.6-0\\MpCmdRun.exe" -Scan -ScanType 1'

    def path_adjustments(self):
        return self.scan_type

    def general_information_before_measurement(self, f):
        if self.scan_type == ScanType.CUSTOM_SCAN:
            f.write(f'Scan Path: {self.custom_scan_path}\n\n')

        from os_funcs import WindowsOS
        WindowsOS.save_antivirus_version(f, self.get_program_name())

    def find_child_id(self, process_pid):
        for i in range(3):  # try again and again
            for proc in psutil.process_iter():
                if proc.name() == self.get_process_name():
                    return proc.pid

        return None


class LogAnomalyDetection(ProgramInterface):
    def __init__(self, model_name, action, script_relative_path, installation_dir):
        super().__init__()
        self.installation_dir = installation_dir
        self.model_name = model_name
        self.script_relative_path = script_relative_path
        self.action = action

    def get_program_name(self):
        return "LogAnomalyDetection"

    def get_command(self):
        return rf"& python -m '{self.script_relative_path}' {self.action}"


class DummyAntivirusProgram(ProgramInterface):
    def __init__(self, scan_path):
        super().__init__()
        self.scan_path = scan_path

    def get_program_name(self):
        return "Dummy Antivirus"

    def get_command(self) -> str:
        return f"python {os.path.join('DummyPrograms', 'FilesReader.py')} {self.scan_path}"

    def general_information_before_measurement(self, f):
        f.write(f'Scan Path: {self.scan_path}\n\n')


class UserActivityProgram(ProgramInterface):
    def get_program_name(self):
        return "User Activity"

    def get_command(self) -> str:
        return f'python {os.path.join("UserActivity", "user_activity.py")} ' \
               f'"{os.path.join(self.results_path, "tasks_times.csv")}"'


class IDSProgram(ProgramInterface):
    def __init__(self, ids_type, interface_name, log_dir, installation_dir="C:\Program Files"):
        super().__init__()
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


class PerfmonProgram(ProgramInterface):
    def __init__(self, program_name):
        super().__init__()
        self.results_path = None
        self.program_name = program_name

    def get_program_name(self) -> str:
        return "Performance Monitor"

    def get_process_name(self) -> str:
        pass

    def should_use_powershell(self) -> bool:
        return True

    def get_command(self) -> str:
        def process_counters_variables(process_id):
            #name = process_name.replace(".exe", "")

            return f"""$proc_id{process_id}={process_id}
            $proc_path{process_id}=((Get-Counter "\\Process(*)\\ID Process").CounterSamples | ? {{$_.RawValue -eq $proc_id{process_id}}}).Path
            $proc_base_path{process_id} = ($proc_path{process_id} -replace "\\\\id process$","")
            
            $io_read_bytes{process_id} = $proc_base_path{process_id} + "\\IO Read Bytes/sec"
            $io_write_bytes{process_id} = $proc_base_path{process_id} + "\\IO Write Bytes/sec"
            $io_read_operations{process_id} = $proc_base_path{process_id} + "\\IO Read Operations/sec"
            $io_write_operations{process_id} = $proc_base_path{process_id} + "\\IO Write Operations/sec"
            """

        def process_counters():
            counters = ""
            for process_id in self.processes_ids:
                counters += f'''
                $io_read_bytes{process_id},
                $io_write_bytes{process_id},
                $io_read_operations{process_id},
                $io_write_operations{process_id},
                '''
            return counters

        processes_vars = ""

        for process_id in self.processes_ids:
            processes_vars += process_counters_variables(process_id) + "\n"

        return f'''Get-Counter 
        {processes_vars}
        $gc = {process_counters()} "\\PhysicalDisk(_Total)\\Disk Reads/sec",
        "\\PhysicalDisk(_Total)\\Disk Writes/sec",
        "\\PhysicalDisk(_Total)\\Disk Read Bytes/sec",
        "\\PhysicalDisk(_Total)\\Disk Write Bytes/sec",
        "\\Processor(_Total)\\% Processor Time", 
        "\\Power Meter(_Total)\\Power"
        Get-Counter -counter $gc -Continuous | Export-Counter -FileFormat "CSV" -Path "C:{self.results_path}\\perfmon.csv"'''

        #return f'Get-Counter gc = "\\PhysicalDisk(_Total)\\Disk Reads/sec", "\\PhysicalDisk(_Total)\\Disk Writes/sec", "\\PhysicalDisk(_Total)\\Disk Read Bytes/sec", "\\PhysicalDisk(_Total)\\Disk Write Bytes/sec", "\\Processor(_Total)\\% Processor Time" Get-Counter -counter $gc -Continuous | Export-Counter -FileFormat "CSV" -Path "{self.results_path}\\perfmon.csv"'

    def find_child_id(self, process_pid) -> Union[int, None]:  #from python 3.10 - int | None:
        return None
