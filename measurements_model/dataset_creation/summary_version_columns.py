from abc import ABC, abstractmethod

from measurements_model.config import SummaryFieldsDuduVersion, SummaryFieldsOtherVersion, SystemColumns, ProcessColumns


class SummaryVersionCols(ABC):

    @abstractmethod
    def get_system_summary_mapping(self, include_network_col: bool = False) -> dict[str, str]:
        pass

    @abstractmethod
    def get_process_summary_mapping(self, include_network_col: bool = False) -> dict[str, str]:
        pass

    @abstractmethod
    def get_total_system_column(self) -> str:
        pass

    @abstractmethod
    def get_total_process_column(self) -> str:
        pass


class DuduSummaryVersionCols(SummaryVersionCols):
    def get_system_summary_mapping(self, include_network_col: bool = False) -> dict[str, str]:
        if not include_network_col:
            return {
                SystemColumns.DURATION_COL: SummaryFieldsDuduVersion.DURATION,
                SystemColumns.CPU_SYSTEM_COL: SummaryFieldsDuduVersion.CPU_SYSTEM,
                SystemColumns.MEMORY_SYSTEM_COL: SummaryFieldsDuduVersion.MEMORY_SYSTEM,
                SystemColumns.DISK_READ_BYTES_SYSTEM_COL: SummaryFieldsDuduVersion.IO_READ_BYTES_SYSTEM,
                SystemColumns.DISK_READ_COUNT_SYSTEM_COL: SummaryFieldsDuduVersion.IO_READ_COUNT_SYSTEM,
                SystemColumns.DISK_WRITE_BYTES_SYSTEM_COL: SummaryFieldsDuduVersion.IO_WRITE_BYTES_SYSTEM,
                SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL: SummaryFieldsDuduVersion.IO_WRITE_COUNT_SYSTEM,
                SystemColumns.DISK_READ_TIME: SummaryFieldsDuduVersion.DISK_IO_READ_TIME,
                SystemColumns.DISK_WRITE_TIME: SummaryFieldsDuduVersion.DISK_IO_WRITE_TIME,
                SystemColumns.PAGE_FAULT_SYSTEM_COL: SummaryFieldsDuduVersion.PAGE_FAULTS,
                SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL: SummaryFieldsDuduVersion.ENERGY_CONSUMPTION
            }

        raise RuntimeError("Dudu summary version does not include network stats!")

    def get_process_summary_mapping(self, include_network_col: bool = False) -> dict[str, str]:
        if not include_network_col:
            return {
                ProcessColumns.CPU_PROCESS_COL: SummaryFieldsDuduVersion.CPU_PROCESS,
                ProcessColumns.MEMORY_PROCESS_COL: SummaryFieldsDuduVersion.MEMORY_PROCESS,
                ProcessColumns.DISK_READ_BYTES_PROCESS_COL: SummaryFieldsDuduVersion.IO_READ_BYTES_PROCESS,
                ProcessColumns.DISK_READ_COUNT_PROCESS_COL: SummaryFieldsDuduVersion.IO_READ_COUNT_PROCESS,
                ProcessColumns.DISK_WRITE_BYTES_PROCESS_COL: SummaryFieldsDuduVersion.IO_WRITE_BYTES_PROCESS,
                ProcessColumns.DISK_WRITE_COUNT_PROCESS_COL: SummaryFieldsDuduVersion.IO_WRITE_COUNT_PROCESS
            }

        raise RuntimeError("Dudu summary version does not include network stats!")

    def get_total_system_column(self) -> str:
        return SummaryFieldsDuduVersion.TOTAL_COLUMN

    def get_total_process_column(self) -> str:
        return SummaryFieldsDuduVersion.TOTAL_COLUMN


class OtherSummaryVersionCols(SummaryVersionCols):
    def __get_summary_columns(self, feature_names: list[str], for_system: bool = True,
                               include_network_col: bool = False) -> dict[str, str]:
        current_columns = [SummaryFieldsOtherVersion.CPU, SummaryFieldsOtherVersion.MEMORY,
                           SummaryFieldsOtherVersion.IO_READ_BYTES, SummaryFieldsOtherVersion.IO_READ_COUNT,
                           SummaryFieldsOtherVersion.IO_WRITE_BYTES, SummaryFieldsOtherVersion.IO_WRITE_COUNT,
                           SummaryFieldsOtherVersion.DISK_IO_READ_TIME, SummaryFieldsOtherVersion.DISK_IO_WRITE_TIME]

        if include_network_col:
            current_columns = current_columns + [SummaryFieldsOtherVersion.NETWORK_SENT_TOTAL,
                                                 SummaryFieldsOtherVersion.NETWORK_SENT_PACKET_COUNT,
                                                 SummaryFieldsOtherVersion.NETWORK_RECEIVED_TOTAL,
                                                 SummaryFieldsOtherVersion.NETWORK_RECEIVED_PACKET_COUNT]

        if for_system:
            current_columns = [SummaryFieldsOtherVersion.DURATION,
                               SummaryFieldsOtherVersion.PAGE_FAULTS,
                               SummaryFieldsOtherVersion.ENERGY_CONSUMPTION] + current_columns

        return {feature_name: current_col for feature_name, current_col in zip(feature_names, current_columns)}


    def get_system_summary_mapping(self, include_network_col: bool = False) -> dict[str, str]:
        feature_cols = [SystemColumns.DURATION_COL, SystemColumns.PAGE_FAULT_SYSTEM_COL,
                        SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL,
                        SystemColumns.CPU_SYSTEM_COL, SystemColumns.MEMORY_SYSTEM_COL,
                        SystemColumns.DISK_READ_BYTES_SYSTEM_COL, SystemColumns.DISK_READ_COUNT_SYSTEM_COL,
                        SystemColumns.DISK_WRITE_BYTES_SYSTEM_COL, SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL,
                        SystemColumns.DISK_READ_TIME, SystemColumns.DISK_WRITE_TIME]

        if include_network_col:
            feature_cols += [SystemColumns.NETWORK_BYTES_SENT_SYSTEM_COL, SystemColumns.NETWORK_PACKETS_SENT_SYSTEM_COL,
                             SystemColumns.NETWORK_BYTES_RECEIVED_SYSTEM_COL, SystemColumns.NETWORK_PACKETS_RECEIVED_SYSTEM_COL]

        return self.__get_summary_columns(feature_names=feature_cols, for_system=True,
                                          include_network_col=include_network_col)


    def get_process_summary_mapping(self, include_network_col: bool = False) -> dict[str, str]:
        feature_cols = [ProcessColumns.CPU_PROCESS_COL, ProcessColumns.MEMORY_PROCESS_COL,
                        ProcessColumns.DISK_READ_BYTES_PROCESS_COL, ProcessColumns.DISK_READ_COUNT_PROCESS_COL,
                        ProcessColumns.DISK_WRITE_BYTES_PROCESS_COL, ProcessColumns.DISK_WRITE_COUNT_PROCESS_COL]

        if include_network_col:
            feature_cols += [ProcessColumns.NETWORK_BYTES_SENT_PROCESS_COL, ProcessColumns.NETWORK_PACKETS_SENT_PROCESS_COL,
                             ProcessColumns.NETWORK_BYTES_RECEIVED_PROCESS_COL, ProcessColumns.NETWORK_PACKETS_RECEIVED_PROCESS_COL]


        return self.__get_summary_columns(feature_names=feature_cols, for_system=False, include_network_col=include_network_col)


    def get_total_system_column(self) -> str:
        return SummaryFieldsOtherVersion.TOTAL_COLUMN

    def get_total_process_column(self) -> str:
        return SummaryFieldsOtherVersion.PROCESS_COLUMN
