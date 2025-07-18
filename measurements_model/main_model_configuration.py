from measurements_model.config import ALL_MEASUREMENTS_NATIVE_VERSION_WITH_NETWORK_DIR_PATH, ProcessColumns
from measurements_model.dataset_creation.summary_version_columns import SystemResourcesIsolationSummaryVersionCols, OtherSummaryVersionCols

# ALL_MEASUREMENTS_DIRS_PATH = ALL_MEASUREMENTS_OTHER_VERSION_WITH_NETWORK_DIR_PATH # can be any directory that holds 'Measurement' directories
ALL_MEASUREMENTS_DIRS_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\Green Security Experiments\Experiments - HeavyLoad combinations - No scan\No scan - heavyLoad - 200525\No scan - heavyLoad - different combinations"

IS_NO_SCAN_MODE = True
IDLE_SUMMARY_VERSION = SystemResourcesIsolationSummaryVersionCols()  # stays the same, unless we change idle directory
MEASUREMENTS_SUMMARY_VERSION = SystemResourcesIsolationSummaryVersionCols()  # changes by the type of dirs inside the "ALL MEASUREMENTS DIR"

NO_SCAN_HEAVYLOAD_PROCESS = "HeavyLoad.exe"
RANDOM_PROCESS_WITH_OTHER_SUMMARY_VERSION = "java"  # todo: change to actual process when there are real results
PROCESS_NAME = NO_SCAN_HEAVYLOAD_PROCESS if IS_NO_SCAN_MODE and isinstance(MEASUREMENTS_SUMMARY_VERSION,
                                                                           SystemResourcesIsolationSummaryVersionCols) else RANDOM_PROCESS_WITH_OTHER_SUMMARY_VERSION  # change accordingly to the process that we want to monitor in the all processes file

NETWORK_SENT_BYTES_COLUMN_NAME = None  # ProcessColumns.NETWORK_BYTES_SENT_PROCESS_COL # can be ether this or None (or system same name)
NETWORK_RECEIVED_BYTES_COLUMN_NAME = None  # ProcessColumns.NETWORK_BYTES_RECEIVED_PROCESS_COL # can be ether this or None (or system same name)
