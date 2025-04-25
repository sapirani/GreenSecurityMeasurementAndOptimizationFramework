import os
from pathlib import Path

import pandas as pd

from measurements_model.config import IDLEColumns, SystemColumns
from measurements_model.dataset_creation.measurement_extractor import MeasurementExtractor

IS_NO_SCAN_MODE = True
SUMMARY_VERSION = None

class DatasetCreator:
    def __init__(self, idle_dir_path: str, measurements_dir_path: str):
        self.idle_dir_path = idle_dir_path
        self.measurements_dir_path = measurements_dir_path


    def __read_idle_stats(self) -> dict[str, any]:
        idle_extractor = MeasurementExtractor(SUMMARY_VERSION, self.idle_dir_path)
        idle_results = idle_extractor.extract_system_summary_result()
        return {
            IDLEColumns.DURATION_COL: idle_results[SystemColumns.DURATION_COL],
            IDLEColumns.CPU_IDLE_COL: idle_results[SystemColumns.CPU_SYSTEM_COL],
            IDLEColumns.MEMORY_IDLE_COL: idle_results[SystemColumns.MEMORY_SYSTEM_COL],
            IDLEColumns.DISK_READ_BYTES_IDLE_COL: idle_results[SystemColumns.DISK_READ_BYTES_SYSTEM_COL],
            IDLEColumns.DISK_READ_COUNT_IDLE_COL: idle_results[SystemColumns.DISK_READ_COUNT_SYSTEM_COL],
            IDLEColumns.DISK_WRITE_BYTES_IDLE_COL: idle_results[SystemColumns.DISK_WRITE_BYTES_SYSTEM_COL],
            IDLEColumns.DISK_WRITE_COUNT_IDLE_COL: idle_results[SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL],
            IDLEColumns.DISK_READ_TIME: idle_results[SystemColumns.DISK_READ_TIME],
            IDLEColumns.DISK_WRITE_TIME: idle_results[SystemColumns.DISK_WRITE_TIME],
            IDLEColumns.PAGE_FAULT_IDLE_COL: idle_results[SystemColumns.PAGE_FAULT_SYSTEM_COL],
            IDLEColumns.ENERGY_TOTAL_USAGE_IDLE_COL: idle_results[SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL],
        }


    def __extract_sample(self, measurement_extractor: MeasurementExtractor, idle_results: dict[str, any]) -> pd.Series:
        system_summary_results = measurement_extractor.extract_system_summary_result()
        process_summary_results = measurement_extractor.extract_process_summary_result(no_scan_mode=IS_NO_SCAN_MODE)
        hardware_results = measurement_extractor.extract_hardware_result()
        print(" === System Info: === ")
        print(system_summary_results)
        print(" === Process Info: ===")
        print(process_summary_results)
        print(" === Hardware Info: ===")
        print(hardware_results)

        new_sample = {**process_summary_results, **system_summary_results, **idle_results, **hardware_results}
        return pd.Series(new_sample)


    def __read_measurements(self) -> pd.DataFrame:
        idle_results = self.__read_idle_stats()
        df_rows = []
        for measurement_dir in Path(self.measurements_dir_path).iterdir():
            if measurement_dir.is_dir():
                print("Collecting info from " + measurement_dir.name)
                measurement_extractor = MeasurementExtractor(summary_version=None, measurement_dir=measurement_dir)
                new_sample = self.__extract_sample(measurement_extractor, idle_results)
                df_rows.append(new_sample)

        return pd.DataFrame(df_rows)

    def create_dataset(self) -> pd.DataFrame:
        return self.__read_measurements()





