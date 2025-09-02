import pandas as pd

from aggregative_results.DTOs.aggregators_features.energy_model_features.system_energy_model_features import \
    SystemEnergyModelFeatures
from measurements_model.config import SummaryFieldsNativeVersion
from measurements_model.dataset_creation.data_extractors.summary_extractors.abstract_summary_extractor import \
    AbstractSummaryExtractor
from measurements_model.dataset_creation.data_extractors.summary_extractors.summary_columns import SummaryColumns

INDEX_COLUMN = "Metric"

# todo: remove when moving permanently to the new approach.
class NativeSummaryExtractor(AbstractSummaryExtractor):
    def extract_system_data(self, summary_file_path: str) -> SystemEnergyModelFeatures:
        df = pd.read_excel(summary_file_path)
        df = df.set_index(INDEX_COLUMN)

        summary_columns = SummaryColumns(
            total_column=SummaryFieldsNativeVersion.TOTAL_COLUMN,
            cpu_column=SummaryFieldsNativeVersion.CPU,
            memory_column=SummaryFieldsNativeVersion.MEMORY,
            disk_read_bytes_column=SummaryFieldsNativeVersion.DISK_IO_READ_BYTES,
            disk_read_count_column=SummaryFieldsNativeVersion.DISK_IO_READ_COUNT,
            disk_write_bytes_column=SummaryFieldsNativeVersion.DISK_IO_WRITE_BYTES,
            disk_write_count_column=SummaryFieldsNativeVersion.DISK_IO_WRITE_COUNT,
            disk_read_time_column=SummaryFieldsNativeVersion.DISK_IO_READ_TIME,
            disk_write_time_column=SummaryFieldsNativeVersion.DISK_IO_WRITE_TIME,
            number_of_page_faults_column=SummaryFieldsNativeVersion.PAGE_FAULTS,
            duration_column=SummaryFieldsNativeVersion.DURATION,
            total_energy_consumption_column=SummaryFieldsNativeVersion.ENERGY_CONSUMPTION)

        if self.__df_contains_network(df):
            summary_columns.network_bytes_kb_sum_sent_column = SummaryFieldsNativeVersion.NETWORK_SENT_TOTAL
            summary_columns.network_packets_sum_sent_column = SummaryFieldsNativeVersion.NETWORK_SENT_PACKET_COUNT
            summary_columns.network_bytes_kb_sum_received_column = SummaryFieldsNativeVersion.NETWORK_RECEIVED_TOTAL
            summary_columns.network_packets_sum_received_column = SummaryFieldsNativeVersion.NETWORK_RECEIVED_PACKET_COUNT

        return self._extract_data_from_df(df, summary_columns)

    def __df_contains_network(self, df: pd.DataFrame) -> bool:
        if SummaryFieldsNativeVersion.NETWORK_SENT_TOTAL in df and \
                SummaryFieldsNativeVersion.NETWORK_SENT_PACKET_COUNT in df and \
                SummaryFieldsNativeVersion.NETWORK_RECEIVED_TOTAL in df and \
                SummaryFieldsNativeVersion.NETWORK_RECEIVED_PACKET_COUNT in df:
            return True
        return False
