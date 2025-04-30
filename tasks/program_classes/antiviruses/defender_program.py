from typing import Union

import psutil

from tasks.program_classes.antiviruses.antivirus_program import AntivirusProgram


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
