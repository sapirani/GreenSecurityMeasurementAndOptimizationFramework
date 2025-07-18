from typing import Union

from utils.general_consts import ScanType
from tasks.program_classes.antiviruses.antivirus_program import AntivirusProgram


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
