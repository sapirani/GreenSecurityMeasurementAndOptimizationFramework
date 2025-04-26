import os

import pandas as pd
from typing import Optional

from measurements_model.config import SystemColumns, ProcessColumns, AllProcessesFileFields, HardwareColumns
from measurements_model.dataset_creation.dataset_utils import return_dict_as_sample
from measurements_model.dataset_creation.summary_version_columns import SummaryVersionCols, DuduSummaryVersionCols

ALL_PROCESSES_CSV = fr"processes_data.csv"
SUMMARY_CSV = fr"summary.xlsx"
HARDWARE_DETAILS_CSV = fr""
DEFAULT_SUMMARY_VERSION = DuduSummaryVersionCols()

SYSTEM_FEATURES = [SystemColumns.DURATION_COL, SystemColumns.CPU_SYSTEM_COL, SystemColumns.MEMORY_SYSTEM_COL,
                   SystemColumns.DISK_READ_BYTES_SYSTEM_COL, SystemColumns.DISK_READ_COUNT_SYSTEM_COL,
                   SystemColumns.DISK_WRITE_BYTES_SYSTEM_COL, SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL,
                   SystemColumns.DISK_READ_TIME, SystemColumns.DISK_WRITE_TIME,
                   SystemColumns.PAGE_FAULT_SYSTEM_COL,
                   SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL]

PROCESS_SUMMARY_FEATURES = [ProcessColumns.CPU_PROCESS_COL, ProcessColumns.MEMORY_PROCESS_COL,
                            ProcessColumns.DISK_READ_BYTES_PROCESS_COL, ProcessColumns.DISK_READ_COUNT_PROCESS_COL,
                            ProcessColumns.DISK_WRITE_BYTES_PROCESS_COL, ProcessColumns.DISK_WRITE_COUNT_PROCESS_COL,
                            ProcessColumns.PAGE_FAULTS_PROCESS_COL]


class MeasurementExtractor:
    def __init__(self, summary_version: Optional[SummaryVersionCols], measurement_dir: str):
        self.measurement_dir = measurement_dir
        self.summary_version = summary_version if summary_version else DEFAULT_SUMMARY_VERSION

    def __extract_system_file(self, need_process: bool) -> dict[str, any]:
        summary_file_path = os.path.join(self.measurement_dir, SUMMARY_CSV)
        df_summary = pd.read_excel(summary_file_path)
        df_summary = df_summary.set_index("Metric")

        if need_process:
            df_columns = self.summary_version.get_process_summary_columns()
            total_df_column = self.summary_version.get_total_process_column()
            features = PROCESS_SUMMARY_FEATURES
        else:
            df_columns = self.summary_version.get_system_summary_columns()
            total_df_column = self.summary_version.get_total_system_column()
            features = SYSTEM_FEATURES

        values = [df_summary.loc[summary_col, total_df_column] for summary_col in df_columns]
        return return_dict_as_sample(features, values)

    def extract_system_summary_result(self) -> dict[str, any]:
        return self.__extract_system_file(need_process=False)

    def extract_process_summary_result(self, no_scan_mode: bool, process_name: str) -> dict[str, any]:
        if no_scan_mode:
            return self.__extract_process_result(process_name)
        return self.__extract_system_file(need_process=True)

    def __extract_process_result(self, process_name: str) -> dict[str, any]:
        df_all_processes = pd.read_csv(os.path.join(self.measurement_dir, ALL_PROCESSES_CSV))
        df_specific_process = df_all_processes[
            df_all_processes[AllProcessesFileFields.PROCESS_NAME_COL] == process_name]

        sample = {ProcessColumns.CPU_PROCESS_COL: df_specific_process[AllProcessesFileFields.CPU].mean(),
                  ProcessColumns.MEMORY_PROCESS_COL: df_specific_process[AllProcessesFileFields.MEMORY].mean(),
                  ProcessColumns.DISK_READ_BYTES_PROCESS_COL: df_specific_process[
                      AllProcessesFileFields.DISK_READ_BYTES].sum(),
                  ProcessColumns.DISK_READ_COUNT_PROCESS_COL: df_specific_process[
                      AllProcessesFileFields.DISK_READ_COUNT].sum(),
                  ProcessColumns.DISK_WRITE_BYTES_PROCESS_COL: df_specific_process[
                      AllProcessesFileFields.DISK_WRITE_BYTES].sum(),
                  ProcessColumns.DISK_WRITE_COUNT_PROCESS_COL: df_specific_process[
                      AllProcessesFileFields.DISK_WRITE_COUNT].sum(),
                  ProcessColumns.PAGE_FAULTS_PROCESS_COL: df_specific_process[
                      AllProcessesFileFields.PAGE_FAULTS].sum()}
        return sample

    def extract_hardware_result(self, use_default: bool = True) -> dict[str, any]:
        if use_default:
            return {HardwareColumns.PC_TYPE: "Mobile Device", HardwareColumns.PC_MANUFACTURER: "Dell Inc.",
                    HardwareColumns.SYSTEM_FAMILY: "Latitude", HardwareColumns.MACHINE_TYPE: "AMD64",
                    HardwareColumns.DEVICE_NAME: "MININT-NT4GD33", HardwareColumns.OPERATING_SYSTEM: "Windows",
                    HardwareColumns.OPERATING_SYSTEM_RELEASE: "10",
                    HardwareColumns.OPERATING_SYSTEM_VERSION: "10.0.19045",
                    HardwareColumns.PROCESSOR_NAME: "Intel64 Family 6 Model 140 Stepping 1, GenuineIntel",
                    HardwareColumns.PROCESSOR_PHYSICAL_CORES: "4", HardwareColumns.PROCESSOR_TOTAL_CORES: "8",
                    HardwareColumns.PROCESSOR_MAX_FREQ: "1805.00", HardwareColumns.PROCESSOR_MIN_FREQ: "0.00",
                    HardwareColumns.TOTAL_RAM: "15.732791900634766",
                    HardwareColumns.PHYSICAL_DISK_NAME: "NVMe Micron 2450 NVMe 512GB",
                    HardwareColumns.PHYSICAL_DISK_MANUFACTURER: "NVMe",
                    HardwareColumns.PHYSICAL_DISK_MODEL: "Micron 2450 NVMe 512GB",
                    HardwareColumns.PHYSICAL_DISK_MEDIA_TYPE: "SSD",
                    HardwareColumns.LOGICAL_DISK_NAME: "NVMe Micron 2450 NVMe 512GB",
                    HardwareColumns.LOGICAL_DISK_MANUFACTURER: "NVMe",
                    HardwareColumns.LOGICAL_DISK_MODEL: "Micron 2450 NVMe 512GB",
                    HardwareColumns.LOGICAL_DISK_DISK_TYPE: "Fixed",
                    HardwareColumns.LOGICAL_DISK_PARTITION_STYLE: "GPT",
                    HardwareColumns.LOGICAL_DISK_NUMBER_OF_PARTITIONS: "5", HardwareColumns.PHYSICAL_SECTOR_SIZE: "512",
                    HardwareColumns.LOGICAL_SECTOR_SIZE: "512", HardwareColumns.BUS_TYPE: "RAID",
                    HardwareColumns.FILESYSTEM: "NTFS", HardwareColumns.BATTERY_DESIGN_CAPACITY: "61970",
                    HardwareColumns.FULLY_CHARGED_BATTERY_CAPACITY: "47850"}

        else:
            df_hardware = pd.read_csv(os.path.join(self.measurement_dir, HARDWARE_DETAILS_CSV))
            df_hardware = df_hardware.iloc[:, 1:]
            return df_hardware.to_dict("records")[0]
