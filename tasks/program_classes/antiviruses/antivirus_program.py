from utils.general_consts import ScanType
from tasks.program_classes.abstract_program import ProgramInterface


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
