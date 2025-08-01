import itertools
from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from utils.general_functions import EnvironmentImpact, BatteryDeltaDrain


def slice_df(df: pd.DataFrame, percent: float):
    num = int(len(df.index) * (percent / 100))
    return df[num: len(df.index) - num]


def get_ratio(numerator, denominator):
    """
    Simple division, if denominator is 0 avoid crashing
    """
    return None if denominator == 0 else numerator / denominator


def get_all_df_by_id(processes_df: pd.DataFrame, processes_ids: List[int]):
    """
    Filter the processes dataframe so it will contain only the main and background processes specified by the user
    """
    return [processes_df[processes_df["pid"] == processes_id] for processes_id in processes_ids]


class AbstractSummaryBuilder(ABC):
    _DEFAULT_SLICE_PERCENT = 3

    @abstractmethod
    def prepare_summary_csv(
            self, processes_df: pd.DataFrame, cpu_df: pd.DataFrame, memory_df: pd.DataFrame,
            disk_io_each_moment_df: pd.DataFrame, network_io_each_moment_df: pd.DataFrame,
            battery_df: pd.DataFrame, processes_names: List[str], finished_scanning_time: List[float],
            processes_ids: List[int]
    ):
        pass

    @abstractmethod
    def get_rows_colors(self) -> List[List[str]]:
        pass

    def colors_func(self, df: pd.DataFrame):
        return [f"background-color: {color}" for color in itertools.chain.from_iterable(self.get_rows_colors())]

    @staticmethod
    def add_general_resource_metrics_info(
            summary_df: pd.DataFrame, num_of_processes: int, sub_disk_df: pd.DataFrame,
            sub_network_df: pd.DataFrame, sub_all_processes_df: List[pd.DataFrame]
    ):
        # TODO: merge cells to one

        none_list = ["X" for _ in range(num_of_processes - 1)]

        total_disk_read_time = sub_disk_df["disk_read_time"].sum()
        total_disk_write_time = sub_disk_df["disk_write_time"].sum()
        summary_df.loc[len(summary_df.index)] = ["Disk IO Read Time (ms - sum)", *none_list, total_disk_read_time]
        summary_df.loc[len(summary_df.index)] = ["Disk IO Write Time (ms - sum)", *none_list, total_disk_write_time]

        # Network IO Sent Bytes
        all_process_network_size_sent_kb = [pd.to_numeric(df["network_kb_sent"]).sum()
                                            for df in sub_all_processes_df]
        total_network_size_sent = sub_network_df["network_kb_sent"].sum()
        summary_df.loc[len(summary_df.index)] = ["Network Size Sent (KB - sum)",
                                                 *all_process_network_size_sent_kb,
                                                 total_network_size_sent]

        # Network IO Sent Packets
        all_process_network_packets_sent = [pd.to_numeric(df["packets_sent"]).sum()
                                            for df in sub_all_processes_df]
        total_network_packets_sent = sub_network_df["packets_sent"].sum()
        summary_df.loc[len(summary_df.index)] = ["Network Packets Sent (# - sum)",
                                                 *all_process_network_packets_sent,
                                                 total_network_packets_sent]

        # Network IO Received Bytes
        all_process_network_size_received_kb = [pd.to_numeric(df["network_kb_received"]).sum()
                                                for df in sub_all_processes_df]
        total_network_size_received = sub_network_df["network_kb_received"].sum()
        summary_df.loc[len(summary_df.index)] = ["Network Size Received (KB - sum)",
                                                 *all_process_network_size_received_kb,
                                                 total_network_size_received]

        # Network IO Received Packets
        all_process_network_packets_received = [pd.to_numeric(df["packets_received"]).sum()
                                                for df in sub_all_processes_df]
        total_network_packets_received = sub_network_df["packets_received"].sum()
        summary_df.loc[len(summary_df.index)] = ["Network Packets Received (# - sum)",
                                                 *all_process_network_packets_received,
                                                 total_network_packets_received]

        return summary_df

    @staticmethod
    def add_energy_info(summary_df: pd.DataFrame, num_of_processes: int, battery_df: pd.DataFrame):
        none_list = ["X" for _ in range(num_of_processes - 1)]

        battery_drain = BatteryDeltaDrain.from_battery_drain(battery_df)
        summary_df.loc[len(summary_df.index)] = ["Energy consumption - total energy(mwh)", *none_list, battery_drain.mwh_drain]
        summary_df.loc[len(summary_df.index)] = ["Battery Drop (%)", *none_list, battery_drain.percent_drain]
        environment_impact = EnvironmentImpact.from_mwh(battery_drain.mwh_drain)
        summary_df.loc[len(summary_df.index)] = ["Trees (KG)", *none_list, environment_impact.kg_of_woods_burned]

        return summary_df
