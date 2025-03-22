import os
import re
import subprocess
import sys
import time
from typing import Union

from general_consts import *
from os_funcs import OSFuncsInterface


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

    def on_exit(self):
        pass

    def path_adjustments(self) -> str:
        return ""

    def general_information_before_measurement(self, f):
        pass

    def should_find_child_id(self):
        return False

    def should_use_powershell(self) -> bool:
        return False

    def set_results_dir(self, results_path):
        self.results_path = results_path

    def set_processes_ids(self, processes_ids):
        self.processes_ids = processes_ids

    def kill_process(self, p, is_posix):
        # global max_timeout_reached
        p.terminate()
        # max_timeout_reached = True

    def process_ignore_cond(self, p):
        return (p.pid == SYSTEM_IDLE_PID)

    ######## probably not needed anymore because we don't start powershell process if not needed in popen
    ######## so the id of the process we are interested in is the pid returned from popen process
    def find_child_id(self, p, is_posix) -> Union[int, None]:  # from python 3.10 - int | None:
        # result_screen = subprocess.run(["powershell", "-Command", f'Get-WmiObject Win32_Process -Filter "ParentProcessID={process_pid}" | Select ProcessID'],
        #                               capture_output=True)
        # if result_screen.returncode != 0:
        #    raise Exception(result_screen.stderr)

        # scanning_process_id = int(str(result_screen.stdout).split("\\r\\n")[3: -3][0].strip())
        # print(scanning_process_id)

        try:
            children = []
            process_pid = p.pid
            for i in range(600):
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

    def path_adjustments(self):
        return self.scan_type

    def general_information_before_measurement(self, f):
        if self.scan_type == ScanType.CUSTOM_SCAN:
            f.write(f'Scan Path: {self.custom_scan_path}\n\n')

        self.save_av_version(f)
        self.save_scan_configuration(f)

    def save_av_version(self, f):
        pass

    def save_scan_configuration(self, f):
        pass


class DefenderProgram(AntivirusProgram):
    def get_program_name(self):
        return "Windows Defender"

    def get_process_name(self) -> str:
        return "MsMpEng.exe"

    def should_use_powershell(self) -> bool:
        return True

    def get_command(self) -> Union[str, list]:
        custom_scan_query = "" if self.custom_scan_path is None else f" -ScanPath {self.custom_scan_path}"
        return f"Start-MpScan -ScanType {self.scan_type}" + custom_scan_query
        # return ["powershell", "-Command", f"Start-MpScan -ScanType {self.scan_type}" + custom_scan_query]
        # ["powershell", "-Command", command]
        # return '"C:\\ProgramData\\Microsoft\\Windows Defender\\Platform\\4.18.2210.6-0\\MpCmdRun.exe" -Scan -ScanType 1'

    def save_av_version(self, f):
        from os_funcs import WindowsOS
        WindowsOS.save_antivirus_version(f, self.get_program_name())

    def find_child_id(self, p, is_posix):
        for i in range(3):  # try again and again
            for proc in psutil.process_iter():
                if proc.name() == self.get_process_name():
                    return proc.pid

        return None


class ClamAVProgram(AntivirusProgram):
    def __init__(self, scan_type, custom_scan_path, recursive, should_optimize, should_mitigate_timestomping):
        super().__init__(scan_type, custom_scan_path)
        self.recursive = recursive
        self.should_optimize = should_optimize
        self.should_mitigate_timestomping = should_mitigate_timestomping

    def get_program_name(self):
        return "ClamAV"

    def get_process_name(self) -> str:
        return "clamscan.exe"

    def save_scan_configuration(self, f):
        f.write(f"Recursive: {self.recursive}, optimize: {self.should_optimize}, mitigate_timestomping: {self.should_mitigate_timestomping}\n\n")

    def get_command(self) -> Union[str, list]:
        if self.scan_type == ScanType.FULL_SCAN:
            path_to_scan = "C:\\"
        elif self.scan_type == ScanType.CUSTOM_SCAN:
            path_to_scan = self.custom_scan_path
        else:
            raise Exception(f"{self.scan_type} is not supported in {self.get_program_name()}")

        recursive_str = "--recursive" if self.recursive else ""
        optimize_str = "--optimize" if self.should_optimize else ""
        timestomping_str = "--mitigate-timestomping" if self.should_mitigate_timestomping else ""

        #return fr'"C:\Program Files\ClamAV\clamscan.exe" --recursive {path_to_scan}'
        #return fr'"C:\dev\clamav\build\install\clamscan.exe" {recursive_str} {optimize_str} {timestomping_str} {path_to_scan} --exclude-dir "green security"'
        return fr'"C:\dev\clamav\build\install\clamscan.exe" {recursive_str} {optimize_str} {timestomping_str} {path_to_scan}'


class SophosAVProgram(AntivirusProgram):
    def get_program_name(self):
        return "Sophos"

    def get_process_name(self) -> str:
        return "sophosinterceptxcli.exe"

    def get_command(self) -> Union[str, list]:
        if self.scan_type == ScanType.FULL_SCAN:
            path_to_scan = "--system"
        elif self.scan_type == ScanType.CUSTOM_SCAN:
            path_to_scan = self.custom_scan_path
        else:
            raise Exception(f"{self.scan_type} is not supported in {self.get_program_name()}")
        return fr"C:\Program Files\Sophos\Endpoint Defense\sophosinterceptxcli.exe scan --noui {path_to_scan}"


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


class CPUConsumer(ProgramInterface):
    def __init__(self, cpu_percent_to_consume, running_time):
        super().__init__()
        self.cpu_percent_to_consume = cpu_percent_to_consume
        if running_time is None:
            self.running_time = 10 * MINUTE
        else:
            self.running_time = running_time

    def get_program_name(self):
        return "CPU Consumer"

    def get_command(self) -> str:
        return rf"python {os.path.join('DummyPrograms', 'CPUConsumer.py')} {self.cpu_percent_to_consume} {self.running_time}"


class MemoryConsumer(ProgramInterface):
    def __init__(self, memory_chunk_size, consumption_speed, running_time):
        super().__init__()
        self.memory_chunk_size = memory_chunk_size
        self.consumption_speed = consumption_speed
        if running_time is None:
            self.running_time = 10 * MINUTE
        else:
            self.running_time = running_time

    def general_information_before_measurement(self, f):
        f.write(f"Memory Consumer - chunk size: {self.memory_chunk_size} bytes,"
                f" speed: {self.consumption_speed} bytes per second\n\n")

    def get_program_name(self):
        return "Memory Consumer"

    def get_command(self) -> str:
        return fr"python {os.path.join('DummyPrograms', 'DummyMemoryConsumer.py')} {self.memory_chunk_size} {self.consumption_speed} {self.running_time}"


class IOWriteConsumer(ProgramInterface):
    def __init__(self, directory_path):
        super().__init__()
        self.directory_path = directory_path

    def get_program_name(self):
        return "IO Write Dummy"

    def get_command(self) -> str:
        return f"python {os.path.join('FilesCreators', 'file_generator.py')} {self.directory_path}"


class PythonServer(ProgramInterface):
    def get_program_name(self):
        return "Python Server"

    def get_command(self) -> str:
        return f"python -m http.server"


class IDSProgram(ProgramInterface):
    def __init__(self, interface_name, pcap_list_dirs, log_dir, configuration_file_path=None,
                 installation_dir=r"C:\Program Files"):
        super().__init__()
        self.interface_name = interface_name
        self.pcap_list_dirs = pcap_list_dirs
        self.log_dir = log_dir
        self.installation_dir = installation_dir
        self.configuration_file_path = configuration_file_path


class SuricataProgram(IDSProgram):
    def get_program_name(self):
        return "Suricata IDS"

    def get_command(self):
        return rf"& '{self.installation_dir}\{IDSType.SURICATA}\{IDSType.SURICATA.lower()}.exe' -i {self.interface_name} -l '{self.installation_dir}\{IDSType.SURICATA}\{self.log_dir}'"


class SnortProgram(IDSProgram):
    def get_program_name(self):
        return "Snort IDS"

    def get_command(self) -> str:
        base_command = f"snort -q -l {self.log_dir} -A fast -c {self.configuration_file_path} "
        if self.interface_name is not None:
            return base_command + f"-i {self.interface_name}"

        elif self.pcap_list_dirs:
            return base_command + f'--pcap-list="{" ".join(self.pcap_list_dirs)}"'

        # return f"snort -q -l {self.log_dir} -i {self.interface_name} -A fast -c {self.configuration_file_path}"


class SplunkProgram(ProgramInterface):
    def get_program_name(self):
        return "Splunk Enterprise SIEM"

    def get_command(self) -> str:
        return "splunk start"

    def kill_process(self, p, is_posix):
        print("extracting")
        # TODO Extraction doesnt working!
        extract_command = f'splunk search "index=eventgen" -output csv -maxout 20000000 -auth shoueii:sH231294'
        print(extract_command)
        with open(os.path.join(self.results_path,"processes.txt"), 'w') as f:
            OSFuncsInterface.run( "pgrep -a splunk", self.should_use_powershell(), is_posix=is_posix, f=f)
        with open(os.path.join(self.results_path,"logs_output.csv"), 'w') as f:
            OSFuncsInterface.run(extract_command, self.should_use_powershell(), is_posix=is_posix, f=f)
            f.flush()
        extract_command = f'splunk search "index=main" -output csv -maxout 20000000 -auth shoueii:sH231294'
        with open(os.path.join(self.results_path,"alerts_output.csv"), 'w') as f:
            OSFuncsInterface.run(extract_command, self.should_use_powershell(), is_posix=is_posix, f=f)
            f.flush()
        # print(extract_process.stderr.read().decode('utf-8'))
        # time.sleep(80)
        print("stopping")
        OSFuncsInterface.run("splunk stop", self.should_use_powershell(), is_posix=is_posix)
        time.sleep(30)
        print("cleaning")
        OSFuncsInterface.run("splunk clean eventdata -index eventgen -f", self.should_use_powershell(), is_posix=is_posix)
        OSFuncsInterface.run("splunk clean eventdata -index main -f", self.should_use_powershell(), is_posix=is_posix)
    
    def process_ignore_cond(self, p):
        return super(SplunkProgram, self).process_ignore_cond(p) or (not p.name().__contains__('splunk'))

    # def should_use_powershell(self) -> bool:
    #     return True
    def should_find_child_id(self) -> bool:
        return True

    def find_child_id(self, p, is_posix) -> Union[int, None]:  # from python 3.10 - int | None:
        try:
            children = None
            # time.sleep(40)
            p.wait()
            result = OSFuncsInterface.run("splunk status", self.should_use_powershell(), is_posix=is_posix)
            print(result)

            stdout = result.stdout.decode('utf-8')
            if is_posix:
                run_match = re.search(fr'(PID:\s*(\d+))', stdout)
                print(run_match)
            else:
                run_match = re.search(fr'(pid\s*(\d+))', stdout)

            # stop_match_linux = re.search(f'splunkd is not running.', stdout)
            # stop_match_windows = re.search(f'Splunkd: Stopped', stdout)
            if run_match:
                children = int(run_match.group(1).split()[1])
                print(f"pid: {children}")
                return children
            else:
                raise Exception("Splunk didn't started correctly!")
        except psutil.NoSuchProcess:
            return None

    def should_use_powershell(self) -> bool:
        return False


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
            # name = process_name.replace(".exe", "")

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

        # return f'Get-Counter gc = "\\PhysicalDisk(_Total)\\Disk Reads/sec", "\\PhysicalDisk(_Total)\\Disk Writes/sec", "\\PhysicalDisk(_Total)\\Disk Read Bytes/sec", "\\PhysicalDisk(_Total)\\Disk Write Bytes/sec", "\\Processor(_Total)\\% Processor Time" Get-Counter -counter $gc -Continuous | Export-Counter -FileFormat "CSV" -Path "{self.results_path}\\perfmon.csv"'

    def find_child_id(self, p, is_posix) -> Union[int, None]:  # from python 3.10 - int | None:
        return None
