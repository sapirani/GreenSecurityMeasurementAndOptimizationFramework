from enum import Enum

from energy_model.configs.columns import SystemColumns, ProcessColumns
from utils.general_consts import MINUTE

DEFAULT_BATCH_INTERVAL_SECONDS = [5 * MINUTE]
MINIMAL_BATCH_DURATION = DEFAULT_BATCH_INTERVAL_SECONDS[0] * 0.5
TIMESTAMP_COLUMN_NAME = "timestamp"
IDLE_SESSION_ID_NAME = "idle"

COLUMNS_TO_CALCULATE_DIFF = [SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL]
COLUMNS_TO_SUM = [SystemColumns.CPU_SYSTEM_COL, ProcessColumns.CPU_PROCESS_COL,
                  SystemColumns.MEMORY_SYSTEM_COL, ProcessColumns.MEMORY_PROCESS_COL,
                  SystemColumns.DISK_READ_COUNT_SYSTEM_COL, ProcessColumns.DISK_READ_COUNT_PROCESS_COL,
                  SystemColumns.DISK_READ_BYTES_SYSTEM_COL, ProcessColumns.DISK_READ_BYTES_PROCESS_COL,
                  SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL, ProcessColumns.DISK_WRITE_COUNT_PROCESS_COL,
                  SystemColumns.DISK_WRITE_BYTES_SYSTEM_COL, ProcessColumns.DISK_WRITE_BYTES_PROCESS_COL,
                  SystemColumns.NETWORK_PACKETS_RECEIVED_SYSTEM_COL, ProcessColumns.NETWORK_PACKETS_RECEIVED_PROCESS_COL,
                  SystemColumns.NETWORK_BYTES_RECEIVED_SYSTEM_COL, ProcessColumns.NETWORK_BYTES_RECEIVED_PROCESS_COL,
                  SystemColumns.NETWORK_PACKETS_SENT_SYSTEM_COL, ProcessColumns.NETWORK_PACKETS_SENT_PROCESS_COL,
                  SystemColumns.NETWORK_BYTES_SENT_SYSTEM_COL, ProcessColumns.NETWORK_BYTES_SENT_PROCESS_COL,
                  SystemColumns.DISK_READ_TIME, SystemColumns.DURATION_COL,
                  SystemColumns.DISK_WRITE_TIME, ProcessColumns.PAGE_FAULT_PROCESS_COL]


class AggregationName:
    SUM: str = "sum"
    FIRST_SAMPLE: str = "first"
    LAST_SAMPLE: str = "last"

class DatasetReaderType(Enum):
    ProcessOfInterest = 1
    AllProcesses = 2

class DatasetCreatorType(Enum):
    Basic = 1
    WithAggregation = 2
    WithEnergyAggregation = 3
    WithProcessRatio = 4

class TargetCalculatorType(Enum):
    SystemBased = 1
    IdleBased = 2
    BatteryDrainBased = 3


DEFAULT_TARGET_CALCULATOR = TargetCalculatorType.SystemBased
DEFAULT_DATASET_READER = DatasetReaderType.ProcessOfInterest
DEFAULT_DATASET_CREATOR = DatasetCreatorType.Basic
DEFAULT_FILTERING_SINGLE_PROCESS = True
