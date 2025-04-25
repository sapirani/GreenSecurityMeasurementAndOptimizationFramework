import pandas as pd

class SummaryVersionConsts:
    pass


ALL_PROCESSES_CSV = fr""
SUMMARY_CSV = fr""
HARDWARE_DETAILS_CSV = fr""


class MeasurementExtractor:
    def __init__(self, summary_version: SummaryVersionConsts, measurement_dir: str):
        self.measurement_dir = measurement_dir
        self.summary_version = summary_version

    def extract_system_summary_result(self) -> dict[str, any]:
        pass

    def extract_process_summary_result(self, no_scan_mode: bool) -> dict[str, any]:
        pass


    def extract_processes_result(self, processes: list[str]) -> dict[str, any]:
        pass

    def extract_hardware_result(self) -> dict[str, any]:
        pass