import pandas as pd

from aggregative_results.DTOs.aggregators_features.energy_model_features.system_energy_model_features import \
    SystemEnergyModelFeatures
from measurements_model.config import SummaryFieldsNativeVersion
from measurements_model.dataset_creation.data_extractors.summary_extractors.abstract_summary_extractor import \
    AbstractSummaryExtractor

INDEX_COLUMN = "Metric"

class NativeSummaryExtractor(AbstractSummaryExtractor):
    def extract_system_data(self, summary_file_path: str) -> SystemEnergyModelFeatures:
        df = pd.read_excel(summary_file_path)
        df = df.set_index(INDEX_COLUMN)

        network_bytes_sent_system = None
        network_packets_sent_system = None
        network_bytes_received_system = None
        network_packets_received_system = None

        if self.__df_contains_network(df):
            network_bytes_sent_system = df[SummaryFieldsNativeVersion.NETWORK_SENT_TOTAL]
            network_packets_sent_system = df[SummaryFieldsNativeVersion.NETWORK_SENT_PACKET_COUNT]
            network_bytes_received_system = df[SummaryFieldsNativeVersion.NETWORK_RECEIVED_TOTAL]
            network_packets_received_system = df[SummaryFieldsNativeVersion.NETWORK_RECEIVED_PACKET_COUNT]

        return SystemEnergyModelFeatures(
            cpu_usage_system=df[SummaryFieldsNativeVersion.CPU],
            memory_gb_usage_system=df[SummaryFieldsNativeVersion.MEMORY],
            disk_read_bytes_kb_usage_system=df[SummaryFieldsNativeVersion.DISK_IO_READ_BYTES],
            disk_read_count_usage_system=df[SummaryFieldsNativeVersion.DISK_IO_READ_COUNT],
            disk_write_bytes_kb_usage_system=df[SummaryFieldsNativeVersion.DISK_IO_WRITE_BYTES],
            disk_write_count_usage_system=df[SummaryFieldsNativeVersion.DISK_IO_WRITE_COUNT],
            disk_read_time_system_ms_sum=df[SummaryFieldsNativeVersion.DISK_IO_READ_TIME],
            disk_write_time_system_ms_sum=df[SummaryFieldsNativeVersion.DISK_IO_WRITE_TIME],
            network_bytes_kb_sum_sent_system=network_bytes_sent_system,
            network_packets_sum_sent_system=network_packets_sent_system,
            network_bytes_kb_sum_received_system=network_bytes_received_system,
            network_packets_sum_received_system=network_packets_received_system,
            number_of_page_faults_system=SummaryFieldsNativeVersion.PAGE_FAULTS,
            duration_system=SummaryFieldsNativeVersion.DURATION,
            total_energy_consumption_system_mWh=SummaryFieldsNativeVersion.ENERGY_CONSUMPTION
        )

    def __df_contains_network(self, df: pd.DataFrame) -> bool:
        if SummaryFieldsNativeVersion.NETWORK_SENT_TOTAL in df and \
                SummaryFieldsNativeVersion.NETWORK_SENT_PACKET_COUNT in df and \
                SummaryFieldsNativeVersion.NETWORK_RECEIVED_TOTAL in df and \
                SummaryFieldsNativeVersion.NETWORK_RECEIVED_PACKET_COUNT in df:
            return True
        return False
