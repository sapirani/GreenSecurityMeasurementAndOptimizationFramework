import re
from pathlib import Path

import pandas as pd

from SapirModel.MeasurementConstants import *


def read_summary_file(directory):
    df_summary = pd.read_excel(os.path.join(directory, SUMMARY_FILE_NAME))
    df_summary = df_summary.set_index("Metric")

    sample = {SystemColumns.DURATION_COL: df_summary.loc[SummaryFields.DURATION, SummaryFields.TOTAL_COLUMN],
              SystemColumns.CPU_SYSTEM_COL: df_summary.loc[SummaryFields.CPU, SummaryFields.TOTAL_COLUMN],
              SystemColumns.MEMORY_SYSTEM_COL: df_summary.loc[SummaryFields.MEMORY, SummaryFields.TOTAL_COLUMN],
              SystemColumns.DISK_READ_BYTES_SYSTEM_COL: df_summary.loc[SummaryFields.IO_READ_BYTES, SummaryFields.TOTAL_COLUMN],
              SystemColumns.DISK_READ_COUNT_SYSTEM_COL: df_summary.loc[SummaryFields.IO_READ_COUNT, SummaryFields.TOTAL_COLUMN],
              SystemColumns.DISK_WRITE_BYTES_SYSTEM_COL: df_summary.loc[SummaryFields.IO_WRITE_BYTES, SummaryFields.TOTAL_COLUMN],
              SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL: df_summary.loc[SummaryFields.IO_WRITE_COUNT, SummaryFields.TOTAL_COLUMN],
              SystemColumns.DISK_READ_TIME: df_summary.loc[SummaryFields.DISK_IO_READ_TIME, SummaryFields.TOTAL_COLUMN],
              SystemColumns.DISK_WRITE_TIME: df_summary.loc[SummaryFields.DISK_IO_WRITE_TIME, SummaryFields.TOTAL_COLUMN],
              SystemColumns.PAGE_FAULT_SYSTEM_COL: df_summary.loc[SummaryFields.PAGE_FAULTS, SummaryFields.TOTAL_COLUMN],
              SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL: df_summary.loc[SummaryFields.ENERGY_CONSUMPTION, SummaryFields.TOTAL_COLUMN]}

    return sample


def read_hardware_information_file(directory):
    df_hardware = pd.read_csv(os.path.join(directory, HARDWARE_INFORMATION_NAME))
    return df_hardware.to_dict("records")[0]


def read_all_processes_file(directory):
    df_process = pd.read_csv(os.path.join(directory, PROCESSES_FILE_NAME))
    print(df_process)
    df_process = df_process[df_process[AllProcessesFileFields.PROCESS_NAME_COL] == HEAVYLOAD_PROCESS_NAME]
    print(df_process)

    sample = {ProcessColumns.CPU_PROCESS_COL: df_process[AllProcessesFileFields.CPU].mean(),
              ProcessColumns.MEMORY_PROCESS_COL: df_process[AllProcessesFileFields.MEMORY].mean(),
              ProcessColumns.DISK_READ_BYTES_PROCESS_COL: df_process[AllProcessesFileFields.DISK_READ_BYTES].sum(),
              ProcessColumns.DISK_READ_COUNT_PROCESS_COL: df_process[AllProcessesFileFields.DISK_READ_COUNT].sum(),
              ProcessColumns.DISK_WRITE_BYTES_PROCESS_COL: df_process[AllProcessesFileFields.DISK_WRITE_BYTES].sum(),
              ProcessColumns.DISK_WRITE_COUNT_PROCESS_COL: df_process[AllProcessesFileFields.DISK_WRITE_COUNT].sum(),
              ProcessColumns.PAGE_FAULTS_PROCESS_COL: df_process[AllProcessesFileFields.PAGE_FAULTS].sum()}
    return sample


def read_idle_information():
    df_idle_summary = pd.read_excel(IDLE_DIRECTORY_PATH)
    df_idle_summary = df_idle_summary.set_index("Metric")

    sample = {IDLEColumns.DURATION_COL: df_idle_summary.loc[SummaryFields.DURATION, SummaryFields.TOTAL_COLUMN],
              IDLEColumns.CPU_IDLE_COL: df_idle_summary.loc[SummaryFields.CPU, SummaryFields.TOTAL_COLUMN],
              IDLEColumns.MEMORY_IDLE_COL: df_idle_summary.loc[SummaryFields.MEMORY, SummaryFields.TOTAL_COLUMN],
              IDLEColumns.DISK_READ_BYTES_IDLE_COL: df_idle_summary.loc[SummaryFields.IO_READ_BYTES, SummaryFields.TOTAL_COLUMN],
              IDLEColumns.DISK_READ_COUNT_IDLE_COL: df_idle_summary.loc[SummaryFields.IO_READ_COUNT, SummaryFields.TOTAL_COLUMN],
              IDLEColumns.DISK_WRITE_BYTES_IDLE_COL: df_idle_summary.loc[SummaryFields.IO_WRITE_BYTES, SummaryFields.TOTAL_COLUMN],
              IDLEColumns.DISK_WRITE_COUNT_IDLE_COL: df_idle_summary.loc[SummaryFields.IO_WRITE_COUNT, SummaryFields.TOTAL_COLUMN],
              IDLEColumns.DISK_READ_TIME: df_idle_summary.loc[SummaryFields.DISK_IO_READ_TIME, SummaryFields.TOTAL_COLUMN],
              IDLEColumns.DISK_WRITE_TIME: df_idle_summary.loc[SummaryFields.DISK_IO_WRITE_TIME, SummaryFields.TOTAL_COLUMN],
              IDLEColumns.PAGE_FAULT_IDLE_COL: df_idle_summary.loc[SummaryFields.PAGE_FAULTS, SummaryFields.TOTAL_COLUMN],
              IDLEColumns.ENERGY_TOTAL_USAGE_IDLE_COL: df_idle_summary.loc[SummaryFields.ENERGY_CONSUMPTION, SummaryFields.TOTAL_COLUMN]}

    return sample


def read_data_from_directory(directory, is_train):
    process_info = read_all_processes_file(directory)  # Read information about the process that consumes combination of resources
    system_info = read_summary_file(directory)  # Read information about the system during the measurement
    idle_info = read_idle_information()  # Read information about idle measurement of the device
    hardware_info = read_hardware_information_file(directory)  # Read information about the device hardware

    # TODO: remove process col from system col in all resources (since the measurements are in NO SCAN mode)
    # TODO: change idle directory
    print(" === System Info: === ")
    print(system_info)
    print(" === Process Info: ===")
    print(process_info)
    print(" === idle Info: ===")
    print(idle_info)
    print(" === Hardware Info: ===")
    print(hardware_info)

    new_sample = {**process_info, **system_info, **idle_info, **hardware_info}

    if is_train:
        new_sample = {**new_sample, **{ProcessColumns.ENERGY_USAGE_PROCESS_COL: (system_info[SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL] - idle_info[IDLEColumns.ENERGY_TOTAL_USAGE_IDLE_COL])}}

    print(" === Combined: === ")
    print(new_sample)
    return new_sample


def read_directories(df, main_directory, is_train):
    for dir in Path(main_directory).iterdir():
        if dir.is_dir():
            print("Collecting info from " + dir.name)
            new_sample = read_data_from_directory(dir, is_train)
            new_sample_df = pd.DataFrame.from_dict(new_sample)
            df = pd.concat([df, new_sample_df])

    csv_name = "trainset.csv" if is_train else "testset.csv"
    df.to_csv(csv_name)
    return df





