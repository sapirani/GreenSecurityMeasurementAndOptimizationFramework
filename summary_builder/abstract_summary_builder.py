from abc import ABC, abstractmethod

import pandas as pd

from utils.general_consts import ProcessesColumns, DiskIOColumns, NetworkIOColumns
from utils.general_functions import calc_delta_capacity, convert_mwh_to_other_metrics


def slice_df(df, percent):
    num = int(len(df.index) * (percent / 100))
    return df[num: len(df.index) - num]


def get_ratio(numerator, denominator):
    """
    Simple division, if denominator is 0 avoid crashing
    """
    return None if denominator == 0 else numerator / denominator


def get_all_df_by_id(processes_df, processes_ids):
    """
    Filter the processes dataframe so it will contain only the main and background processes specified by the user
    """
    return [processes_df[processes_df[ProcessesColumns.PROCESS_ID] == id] for id in processes_ids]


class AbstractSummaryBuilder(ABC):
    @abstractmethod
    def prepare_summary_csv(self, processes_df, cpu_df, memory_df, disk_io_each_moment_df, network_io_each_moment_df,
                            battery_df, processes_names, finished_scanning_time, processes_ids):
        pass

    @abstractmethod
    def colors_func(self, df):
        pass

    @staticmethod
    def add_general_info(summary_df, num_of_processes, battery_df, sub_disk_df, sub_network_df, sub_all_processes_df):
        # TODO: merge cells to one

        none_list = ["X" for _ in range(num_of_processes - 1)]

        total_disk_read_time = sub_disk_df[DiskIOColumns.READ_TIME].sum()
        total_disk_write_time = sub_disk_df[DiskIOColumns.WRITE_TIME].sum()
        summary_df.loc[len(summary_df.index)] = ["Disk IO Read Time (ms - sum)", *none_list, total_disk_read_time]
        summary_df.loc[len(summary_df.index)] = ["Disk IO Write Time (ms - sum)", *none_list, total_disk_write_time]

        # Network IO Sent Bytes
        all_process_network_size_sent_kb = [pd.to_numeric(df[ProcessesColumns.BYTES_SENT]).sum()
                                            for df in sub_all_processes_df]
        total_network_size_sent = sub_network_df[NetworkIOColumns.KB_SENT].sum()
        summary_df.loc[len(summary_df.index)] = ["Network Size Sent (KB - sum)",
                                                 *all_process_network_size_sent_kb,
                                                 total_network_size_sent]

        # Network IO Sent Packets
        all_process_network_packets_sent = [pd.to_numeric(df[ProcessesColumns.PACKETS_SENT]).sum()
                                            for df in sub_all_processes_df]
        total_network_packets_sent = sub_network_df[NetworkIOColumns.PACKETS_SENT].sum()
        summary_df.loc[len(summary_df.index)] = ["Network Packets Sent (# - sum)",
                                                 *all_process_network_packets_sent,
                                                 total_network_packets_sent]

        # Network IO Received Bytes
        all_process_network_size_received_kb = [pd.to_numeric(df[ProcessesColumns.BYTES_RECEIVED]).sum()
                                                for df in sub_all_processes_df]
        total_network_size_received = sub_network_df[NetworkIOColumns.KB_RECEIVED].sum()
        summary_df.loc[len(summary_df.index)] = ["Network Size Received (KB - sum)",
                                                 *all_process_network_size_received_kb,
                                                 total_network_size_received]

        # Network IO Received Packets
        all_process_network_packets_received = [pd.to_numeric(df[ProcessesColumns.PACKETS_RECEIVED]).sum()
                                                for df in sub_all_processes_df]
        total_network_packets_received = sub_network_df[NetworkIOColumns.PACKETS_RECEIVED].sum()
        summary_df.loc[len(summary_df.index)] = ["Network Packets Received (# - sum)",
                                                 *all_process_network_packets_received,
                                                 total_network_packets_received]

        battery_drop = calc_delta_capacity(battery_df)
        summary_df.loc[len(summary_df.index)] = ["Energy consumption - total energy(mwh)", *none_list, battery_drop[0]]
        summary_df.loc[len(summary_df.index)] = ["Battery Drop (%)", *none_list, battery_drop[1]]
        other_metrics = convert_mwh_to_other_metrics(battery_drop[0])
        summary_df.loc[len(summary_df.index)] = ["Trees (KG)", *none_list, other_metrics[3]]

        return summary_df
