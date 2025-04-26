import pandas as pd

from measurements_model.config import HardwareColumns
from measurements_model.dataset_processing.process_data.feature_selection.process_and_full_system_feature_selector import \
    ProcessAndTotalSystem


class ProcessAndSystemNoHardware(ProcessAndTotalSystem):
    """
    Includes all process and system features.
    Doesn't include any idle or hardware features.
    Includes the energy consumption of a process BUT not the total energy consumption of the system.
    The system features are NOT a subtraction of idle and process.
    """

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().select_features(df)
        return df.drop([HardwareColumns.PC_TYPE, HardwareColumns.PC_MANUFACTURER, HardwareColumns.SYSTEM_FAMILY,
                        HardwareColumns.MACHINE_TYPE,
                        HardwareColumns.DEVICE_NAME, HardwareColumns.OPERATING_SYSTEM,
                        HardwareColumns.OPERATING_SYSTEM_RELEASE, HardwareColumns.OPERATING_SYSTEM_VERSION,
                        HardwareColumns.PROCESSOR_NAME, HardwareColumns.PROCESSOR_PHYSICAL_CORES,
                        HardwareColumns.PROCESSOR_TOTAL_CORES, HardwareColumns.PROCESSOR_MAX_FREQ,
                        HardwareColumns.PROCESSOR_MIN_FREQ, HardwareColumns.TOTAL_RAM,
                        HardwareColumns.PHYSICAL_DISK_NAME, HardwareColumns.PHYSICAL_DISK_MANUFACTURER,
                        HardwareColumns.PHYSICAL_DISK_MODEL,
                        HardwareColumns.PHYSICAL_DISK_MEDIA_TYPE, HardwareColumns.LOGICAL_DISK_NAME,
                        HardwareColumns.LOGICAL_DISK_MANUFACTURER,
                        HardwareColumns.LOGICAL_DISK_MODEL, HardwareColumns.LOGICAL_DISK_DISK_TYPE,
                        HardwareColumns.LOGICAL_DISK_PARTITION_STYLE,
                        HardwareColumns.LOGICAL_DISK_NUMBER_OF_PARTITIONS, HardwareColumns.PHYSICAL_SECTOR_SIZE,
                        HardwareColumns.LOGICAL_SECTOR_SIZE,
                        HardwareColumns.BUS_TYPE, HardwareColumns.FILESYSTEM, HardwareColumns.BATTERY_DESIGN_CAPACITY,
                        HardwareColumns.FULLY_CHARGED_BATTERY_CAPACITY],
                       axis=1)
