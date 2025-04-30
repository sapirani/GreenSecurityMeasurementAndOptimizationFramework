from abc import ABC, abstractmethod

from measurements_model.config import SummaryFieldsDuduVersion, SummaryFieldsOtherVersion


class SummaryVersionCols(ABC):
    @abstractmethod
    def get_system_summary_columns(self) -> list[str]:
        pass

    @abstractmethod
    def get_process_summary_columns(self) -> list[str]:
        pass

    @abstractmethod
    def get_total_system_column(self) -> str:
        pass

    @abstractmethod
    def get_total_process_column(self) -> str:
        pass


class DuduSummaryVersionCols(SummaryVersionCols):
    def get_system_summary_columns(self) -> list[str]:
        return [SummaryFieldsDuduVersion.DURATION, SummaryFieldsDuduVersion.CPU_SYSTEM,
                SummaryFieldsDuduVersion.MEMORY_SYSTEM,
                SummaryFieldsDuduVersion.IO_READ_BYTES_SYSTEM, SummaryFieldsDuduVersion.IO_READ_COUNT_SYSTEM,
                SummaryFieldsDuduVersion.IO_WRITE_BYTES_SYSTEM, SummaryFieldsDuduVersion.IO_WRITE_COUNT_SYSTEM,
                SummaryFieldsDuduVersion.DISK_IO_READ_TIME, SummaryFieldsDuduVersion.DISK_IO_WRITE_TIME,
                SummaryFieldsDuduVersion.PAGE_FAULTS, SummaryFieldsDuduVersion.ENERGY_CONSUMPTION]

    def get_process_summary_columns(self) -> list[str]:
        return [SummaryFieldsDuduVersion.CPU_PROCESS, SummaryFieldsDuduVersion.MEMORY_PROCESS,
                SummaryFieldsDuduVersion.IO_READ_BYTES_PROCESS, SummaryFieldsDuduVersion.IO_READ_COUNT_PROCESS,
                SummaryFieldsDuduVersion.IO_WRITE_BYTES_PROCESS, SummaryFieldsDuduVersion.IO_WRITE_COUNT_PROCESS,
                SummaryFieldsDuduVersion.DISK_IO_READ_TIME, SummaryFieldsDuduVersion.DISK_IO_WRITE_TIME,
                SummaryFieldsDuduVersion.PAGE_FAULTS, SummaryFieldsDuduVersion.ENERGY_CONSUMPTION]

    def get_total_system_column(self) -> str:
        return SummaryFieldsDuduVersion.TOTAL_COLUMN

    def get_total_process_column(self) -> str:
        return SummaryFieldsDuduVersion.TOTAL_COLUMN


class OtherSummaryVersionCols(SummaryVersionCols):
    def get_system_summary_columns(self) -> list[str]:
        return [SummaryFieldsOtherVersion.DURATION, SummaryFieldsOtherVersion.CPU, SummaryFieldsOtherVersion.MEMORY,
                SummaryFieldsOtherVersion.IO_READ_BYTES, SummaryFieldsOtherVersion.IO_READ_COUNT,
                SummaryFieldsOtherVersion.IO_WRITE_BYTES, SummaryFieldsOtherVersion.IO_WRITE_COUNT,
                SummaryFieldsOtherVersion.DISK_IO_READ_TIME, SummaryFieldsOtherVersion.DISK_IO_WRITE_TIME,
                SummaryFieldsOtherVersion.PAGE_FAULTS, SummaryFieldsOtherVersion.ENERGY_CONSUMPTION]

    def get_process_summary_columns(self) -> list[str]:
        return [SummaryFieldsOtherVersion.CPU, SummaryFieldsOtherVersion.MEMORY,
                SummaryFieldsOtherVersion.IO_READ_BYTES, SummaryFieldsOtherVersion.IO_READ_COUNT,
                SummaryFieldsOtherVersion.IO_WRITE_BYTES, SummaryFieldsOtherVersion.IO_WRITE_COUNT,
                SummaryFieldsOtherVersion.DISK_IO_READ_TIME, SummaryFieldsOtherVersion.DISK_IO_WRITE_TIME,
                SummaryFieldsOtherVersion.PAGE_FAULTS]

    def get_total_system_column(self) -> str:
        return SummaryFieldsOtherVersion.TOTAL_COLUMN

    def get_total_process_column(self) -> str:
        return SummaryFieldsOtherVersion.PROCESS_COLUMN
