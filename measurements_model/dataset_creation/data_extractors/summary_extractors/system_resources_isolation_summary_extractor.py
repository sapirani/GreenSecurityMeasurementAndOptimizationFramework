import pandas as pd

from aggregative_results.DTOs.aggregators_features.energy_model_features.system_energy_model_features import \
    SystemEnergyModelFeatures
from measurements_model.config import SummaryFieldsSystemResourcesIsolationVersion
from measurements_model.dataset_creation.data_extractors.summary_extractors.abstract_summary_extractor import \
    AbstractSummaryExtractor
from measurements_model.dataset_creation.data_extractors.summary_extractors.summary_columns import SummaryColumns

INDEX_COLUMN = "Metric"


class SystemResourcesIsolationSummaryExtractor(AbstractSummaryExtractor):
    def extract_system_data(self, summary_file_path: str) -> SystemEnergyModelFeatures:
        df = pd.read_excel(summary_file_path)
        df = df.set_index(INDEX_COLUMN)
        print("Network statistics are not available for this summary version.")

        summary_columns = SummaryColumns(
            total_column=SummaryFieldsSystemResourcesIsolationVersion.TOTAL_COLUMN,
            cpu_column=SummaryFieldsSystemResourcesIsolationVersion.CPU_SYSTEM,
            memory_column=SummaryFieldsSystemResourcesIsolationVersion.MEMORY_SYSTEM,
            disk_read_bytes_column=SummaryFieldsSystemResourcesIsolationVersion.IO_READ_BYTES_SYSTEM,
            disk_read_count_column=SummaryFieldsSystemResourcesIsolationVersion.IO_READ_COUNT_SYSTEM,
            disk_write_bytes_column=SummaryFieldsSystemResourcesIsolationVersion.IO_WRITE_BYTES_SYSTEM,
            disk_write_count_column=SummaryFieldsSystemResourcesIsolationVersion.IO_WRITE_COUNT_SYSTEM,
            disk_read_time_column=SummaryFieldsSystemResourcesIsolationVersion.DISK_IO_READ_TIME,
            disk_write_time_column=SummaryFieldsSystemResourcesIsolationVersion.DISK_IO_WRITE_TIME,
            network_bytes_kb_sum_sent_column=None,
            network_packets_sum_sent_column=None,
            network_bytes_kb_sum_received_column=None,
            network_packets_sum_received_column=None,
            number_of_page_faults_column=SummaryFieldsSystemResourcesIsolationVersion.PAGE_FAULTS,
            duration_column=SummaryFieldsSystemResourcesIsolationVersion.DURATION,
            total_energy_consumption_column=SummaryFieldsSystemResourcesIsolationVersion.ENERGY_CONSUMPTION)

        return self._extract_data_from_df(df, summary_columns)
