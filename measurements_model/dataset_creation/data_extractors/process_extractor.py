import pandas as pd

from DTOs.aggregators_features.energy_model_features.process_energy_model_features import ProcessEnergyModelFeatures
from measurements_model.config import AllProcessesFileFields


class ProcessExtractor:
    def extract(self, process_file_path: str, process_name: str) -> ProcessEnergyModelFeatures:
        df_all_processes = pd.read_csv(process_file_path)
        df_specific_process = df_all_processes[
            df_all_processes[AllProcessesFileFields.PROCESS_NAME_COL] == process_name]

        network_bytes_sent_process = None
        network_packets_sent_process = None
        network_bytes_received_process = None
        network_packets_received_process = None
        if self.__df_contains_network(df_all_processes):
            network_bytes_sent_process = df_all_processes[AllProcessesFileFields.NETWORK_BYTES_SENT].sum()
            network_packets_sent_process = df_all_processes[AllProcessesFileFields.NETWORK_PACKETS_SENT].sum()
            network_bytes_received_process = df_all_processes[AllProcessesFileFields.NETWORK_BYTES_RECEIVED].sum()
            network_packets_received_process = df_all_processes[AllProcessesFileFields.NETWORK_PACKETS_RECEIVED].sum()

        return ProcessEnergyModelFeatures(
            cpu_time_process=df_specific_process[AllProcessesFileFields.CPU].mean(),
            memory_mb_relative_process=df_specific_process[AllProcessesFileFields.MEMORY].mean(),
            disk_read_kb_process=df_specific_process[AllProcessesFileFields.DISK_READ_BYTES].sum(),
            disk_write_kb_process=df_specific_process[AllProcessesFileFields.DISK_WRITE_BYTES].sum(),
            disk_read_count_process=df_specific_process[AllProcessesFileFields.DISK_READ_COUNT].sum(),
            disk_write_count_process=df_specific_process[AllProcessesFileFields.DISK_WRITE_COUNT].sum(),
            network_kb_sent_process=network_bytes_sent_process,
            network_packets_sent_process=network_packets_sent_process,
            network_kb_received_process=network_bytes_received_process,
            network_packets_received_process=network_packets_received_process
        )

    def __df_contains_network(self, df: pd.DataFrame) -> bool:
        if AllProcessesFileFields.NETWORK_BYTES_SENT in df and \
                AllProcessesFileFields.NETWORK_PACKETS_SENT in df and \
                AllProcessesFileFields.NETWORK_PACKETS_RECEIVED in df and \
                AllProcessesFileFields.NETWORK_BYTES_RECEIVED in df:
            return True
        return False
