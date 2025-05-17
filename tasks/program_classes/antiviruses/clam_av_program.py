from typing import Union

from general_consts import ScanType
from tasks.program_classes.antiviruses.antivirus_program import AntivirusProgram


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
