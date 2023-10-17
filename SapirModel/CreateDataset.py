import re
from operator import contains
from pathlib import Path

import pandas as pd

from SapirModel.MeasurementConstants import *

def return_dict_as_sample(features, info):
    sample = {}
    for index, feature in enumerate(features):
        sample[feature] = float(info[index])

    return sample

def read_summary_file_system(file_path, summery_version_dudu):
    df_summary = pd.read_excel(file_path)
    df_summary = df_summary.set_index("Metric")
    if summery_version_dudu:
        return [df_summary.loc[SummaryFieldsDuduVersion.DURATION, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.CPU_SYSTEM, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.MEMORY_SYSTEM, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.IO_READ_BYTES_SYSTEM, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.IO_READ_COUNT_SYSTEM, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.IO_WRITE_BYTES_SYSTEM, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.IO_WRITE_COUNT_SYSTEM, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.DISK_IO_READ_TIME, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.DISK_IO_WRITE_TIME, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.PAGE_FAULTS, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.ENERGY_CONSUMPTION, SummaryFieldsDuduVersion.TOTAL_COLUMN]]

    else:
        return [df_summary.loc[SummaryFieldsOtherVersion.DURATION, SummaryFieldsOtherVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.CPU, SummaryFieldsOtherVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.MEMORY, SummaryFieldsOtherVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.IO_READ_BYTES, SummaryFieldsOtherVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.IO_READ_COUNT, SummaryFieldsOtherVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.IO_WRITE_BYTES, SummaryFieldsOtherVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.IO_WRITE_COUNT, SummaryFieldsOtherVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.DISK_IO_READ_TIME, SummaryFieldsOtherVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.DISK_IO_WRITE_TIME, SummaryFieldsOtherVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.PAGE_FAULTS, SummaryFieldsOtherVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.ENERGY_CONSUMPTION, SummaryFieldsOtherVersion.TOTAL_COLUMN]]


def read_summary_file_process(file_path, summery_version_dudu=False):
    df_summary = pd.read_excel(file_path)
    df_summary = df_summary.set_index("Metric")
    if summery_version_dudu:
        return [df_summary.loc[SummaryFieldsDuduVersion.CPU_PROCESS, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.MEMORY_PROCESS, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.IO_READ_BYTES_PROCESS, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.IO_READ_COUNT_PROCESS, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.IO_WRITE_BYTES_PROCESS, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.IO_WRITE_COUNT_PROCESS, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.DISK_IO_READ_TIME, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.DISK_IO_WRITE_TIME, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.PAGE_FAULTS, SummaryFieldsDuduVersion.TOTAL_COLUMN],
                df_summary.loc[SummaryFieldsDuduVersion.ENERGY_CONSUMPTION, SummaryFieldsDuduVersion.TOTAL_COLUMN]]

    else:
        return [df_summary.loc[SummaryFieldsOtherVersion.CPU, SummaryFieldsOtherVersion.PROCESS_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.MEMORY, SummaryFieldsOtherVersion.PROCESS_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.IO_READ_BYTES, SummaryFieldsOtherVersion.PROCESS_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.IO_READ_COUNT, SummaryFieldsOtherVersion.PROCESS_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.IO_WRITE_BYTES, SummaryFieldsOtherVersion.PROCESS_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.IO_WRITE_COUNT, SummaryFieldsOtherVersion.PROCESS_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.DISK_IO_READ_TIME, SummaryFieldsOtherVersion.PROCESS_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.DISK_IO_WRITE_TIME, SummaryFieldsOtherVersion.PROCESS_COLUMN],
                df_summary.loc[SummaryFieldsOtherVersion.PAGE_FAULTS, SummaryFieldsOtherVersion.PROCESS_COLUMN]]


def read_summary_system(directory, summery_version_dudu):
    info_system = read_summary_file_system(os.path.join(directory, SUMMARY_FILE_NAME), summery_version_dudu)
    system_features = [SystemColumns.DURATION_COL, SystemColumns.CPU_SYSTEM_COL, SystemColumns.MEMORY_SYSTEM_COL,
                       SystemColumns.DISK_READ_BYTES_SYSTEM_COL, SystemColumns.DISK_READ_COUNT_SYSTEM_COL,
                       SystemColumns.DISK_WRITE_BYTES_SYSTEM_COL, SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL,
                       SystemColumns.DISK_READ_TIME, SystemColumns.DISK_WRITE_TIME, SystemColumns.PAGE_FAULT_SYSTEM_COL,
                       SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL]
    return return_dict_as_sample(system_features, info_system)


def read_hardware_information_file(directory):
    if os.path.isfile(os.path.join(directory, HARDWARE_INFORMATION_NAME)):
        df_hardware = pd.read_csv(os.path.join(directory, HARDWARE_INFORMATION_NAME))
        return df_hardware.to_dict("records")[0]

    else:
        return {
            HardwareColumns.PC_TYPE: "Mobile Device", HardwareColumns.PC_MANUFACTURER: "Dell Inc.",
            HardwareColumns.SYSTEM_FAMILY: "Latitude", HardwareColumns.MACHINE_TYPE: "AMD64",
            HardwareColumns.DEVICE_NAME: "MININT-NT4GD33", HardwareColumns.OPERATING_SYSTEM: "Windows",
            HardwareColumns.OPERATING_SYSTEM_RELEASE: "10", HardwareColumns.OPERATING_SYSTEM_VERSION: "10.0.19045",
            HardwareColumns.PROCESSOR_NAME: "Intel64 Family 6 Model 140 Stepping 1, GenuineIntel",
            HardwareColumns.PROCESSOR_PHYSICAL_CORES: "4", HardwareColumns.PROCESSOR_TOTAL_CORES: "8",
            HardwareColumns.PROCESSOR_MAX_FREQ: "1805.00", HardwareColumns.PROCESSOR_MIN_FREQ: "0.00",
            HardwareColumns.TOTAL_RAM: "15.732791900634766", HardwareColumns.PHYSICAL_DISK_NAME: "NVMe Micron 2450 NVMe 512GB",
            HardwareColumns.PHYSICAL_DISK_MANUFACTURER: "NVMe", HardwareColumns.PHYSICAL_DISK_MODEL: "Micron 2450 NVMe 512GB",
            HardwareColumns.PHYSICAL_DISK_MEDIA_TYPE: "SSD", HardwareColumns.LOGICAL_DISK_NAME: "NVMe Micron 2450 NVMe 512GB",
            HardwareColumns.LOGICAL_DISK_MANUFACTURER: "NVMe", HardwareColumns.LOGICAL_DISK_MODEL: "Micron 2450 NVMe 512GB",
            HardwareColumns.LOGICAL_DISK_DISK_TYPE: "Fixed", HardwareColumns.LOGICAL_DISK_PARTITION_STYLE: "GPT",
            HardwareColumns.LOGICAL_DISK_NUMBER_OF_PARTITIONS: "5", HardwareColumns.PHYSICAL_SECTOR_SIZE: "512",
            HardwareColumns.LOGICAL_SECTOR_SIZE: "512", HardwareColumns.BUS_TYPE: "RAID",
            HardwareColumns.FILESYSTEM: "NTFS", HardwareColumns.BATTERY_DESIGN_CAPACITY: "61970",
            HardwareColumns.FULLY_CHARGED_BATTERY_CAPACITY: "47850"}


def read_all_processes_file(directory):
    df_process = pd.read_csv(os.path.join(directory, PROCESSES_FILE_NAME))
    print(df_process)
    df_heavyload_process = df_process[df_process[AllProcessesFileFields.PROCESS_NAME_COL] == HEAVYLOAD_PROCESS_NAME]
    print(df_heavyload_process)

    sample = {ProcessColumns.CPU_PROCESS_COL: df_heavyload_process[AllProcessesFileFields.CPU].mean(),
              ProcessColumns.MEMORY_PROCESS_COL: df_heavyload_process[AllProcessesFileFields.MEMORY].mean(),
              ProcessColumns.DISK_READ_BYTES_PROCESS_COL: df_heavyload_process[AllProcessesFileFields.DISK_READ_BYTES].sum(),
              ProcessColumns.DISK_READ_COUNT_PROCESS_COL: df_heavyload_process[AllProcessesFileFields.DISK_READ_COUNT].sum(),
              ProcessColumns.DISK_WRITE_BYTES_PROCESS_COL: df_heavyload_process[AllProcessesFileFields.DISK_WRITE_BYTES].sum(),
              ProcessColumns.DISK_WRITE_COUNT_PROCESS_COL: df_heavyload_process[AllProcessesFileFields.DISK_WRITE_COUNT].sum(),
              ProcessColumns.PAGE_FAULTS_PROCESS_COL: df_heavyload_process[AllProcessesFileFields.PAGE_FAULTS].sum()}
    return sample

def read_idle_information(summery_version_dudu):
    info_idle = read_summary_file_system(IDLE_SUMMARY_PATH, summery_version_dudu)
    idle_features = [IDLEColumns.DURATION_COL, IDLEColumns.CPU_IDLE_COL, IDLEColumns.MEMORY_IDLE_COL,
                     IDLEColumns.DISK_READ_BYTES_IDLE_COL, IDLEColumns.DISK_READ_COUNT_IDLE_COL,
                     IDLEColumns.DISK_WRITE_BYTES_IDLE_COL, IDLEColumns.DISK_WRITE_COUNT_IDLE_COL,
                     IDLEColumns.DISK_READ_TIME, IDLEColumns.DISK_WRITE_TIME, IDLEColumns.PAGE_FAULT_IDLE_COL,
                     IDLEColumns.ENERGY_TOTAL_USAGE_IDLE_COL]

    return return_dict_as_sample(idle_features, info_idle)


def read_summary_system_and_process(directory, summery_version_dudu):
    dict_summary_system = read_summary_system(directory, summery_version_dudu)
    info_process_from_summery = read_summary_file_process(os.path.join(directory, SUMMARY_FILE_NAME), summery_version_dudu)
    process_features = [ProcessColumns.CPU_PROCESS_COL, ProcessColumns.MEMORY_PROCESS_COL,
                                   ProcessColumns.DISK_READ_BYTES_PROCESS_COL, ProcessColumns.DISK_READ_COUNT_PROCESS_COL,
                                   ProcessColumns.DISK_WRITE_BYTES_PROCESS_COL, ProcessColumns.DISK_WRITE_COUNT_PROCESS_COL,
                                   ProcessColumns.PAGE_FAULTS_PROCESS_COL]

    return return_dict_as_sample(process_features, info_process_from_summery), dict_summary_system
    pass


def read_data_from_directory(directory, is_train, idle_info, summery_version_dudu, no_scan_mode):
    if no_scan_mode:
        process_info = read_all_processes_file(directory)  # Read information about the process that consumes combination of resources
        system_info = read_summary_system(directory, summery_version_dudu)  # Read information about the system during the measurement
    else:
        process_info, system_info = read_summary_system_and_process(directory, summery_version_dudu)

    hardware_info = read_hardware_information_file(directory)  # Read information about the device hardware

    # TODO: remove process col from system col in all resources (since the measurements are in NO SCAN mode)
    # TODO: change idle directory
    print(" === System Info: === ")
    print(system_info)
    print(" === Process Info: ===")
    print(process_info)
    print(" === Hardware Info: ===")
    print(hardware_info)

    new_sample = {**process_info, **system_info, **idle_info, **hardware_info}

    if is_train:
        new_sample = {**new_sample, **{ProcessColumns.ENERGY_USAGE_PROCESS_COL: (system_info[SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL] - idle_info[IDLEColumns.ENERGY_TOTAL_USAGE_IDLE_COL])}}

    print(" === Combined: === ")
    print(new_sample)
    return new_sample


def read_directories(df, main_directory, is_train, summery_version_dudu, no_scan_mode):

    idle_info = read_idle_information(True)  # Read information about idle measurement of the device
    print(" === idle Info: ===")
    print(idle_info)
    df_rows = []
    for dir in Path(main_directory).iterdir():
        if dir.is_dir():
            print("Collecting info from " + dir.name)

            new_sample = read_data_from_directory(dir, is_train, idle_info, summery_version_dudu, no_scan_mode)
            df_rows.append(new_sample)
            #df = pd.concat([df, new_sample_df])


    return pd.DataFrame.from_dict(df_rows, orient='columns')

def pre_process_data(df):
    df = df[df[ProcessColumns.ENERGY_USAGE_PROCESS_COL] >= 0]
    return df


def initialize_dataset(is_train=True):
    cols = DATASET_COLUMNS + [ProcessColumns.ENERGY_USAGE_PROCESS_COL] if is_train else DATASET_COLUMNS
    return pd.DataFrame(columns=cols)

def create_train_set():
    print("======== Creating Train Dataset ========")
    train_df = initialize_dataset(True)
    train_df = read_directories(train_df, TRAIN_MEASUREMENTS_DIR_PATH, is_train=True, summery_version_dudu=True, no_scan_mode=True)
    print(train_df)
    train_df = pre_process_data(train_df)
    train_df.to_csv(TRAIN_SET_PATH)

def create_test_set():
    print("======== Creating Test Dataset ========")

    # Change is_train to false if there is no need in target column
    test_df = initialize_dataset(True)
    test_df = read_directories(test_df, TEST_MEASUREMENTS_DIR_PATH, is_train=True, summery_version_dudu=False, no_scan_mode=False)
    print(test_df)
    print()
    print(test_df.dtypes)
    test_df = pre_process_data(test_df)
    test_df.to_csv(TEST_SET_PATH)

def main():
    create_train_set()
    create_test_set()


if __name__ == '__main__':
    main()



