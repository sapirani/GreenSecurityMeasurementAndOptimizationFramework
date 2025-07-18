from typing import List

import pandas as pd

from utils.general_consts import ProcessesColumns, CPUColumns, MemoryColumns, KB, DiskIOColumns
from summary_builder.abstract_summary_builder import AbstractSummaryBuilder, slice_df, get_all_df_by_id


class NativeSummaryBuilder(AbstractSummaryBuilder):
    def prepare_summary_csv(
            self, processes_df: pd.DataFrame, cpu_df: pd.DataFrame, memory_df: pd.DataFrame,
            disk_io_each_moment_df: pd.DataFrame, network_io_each_moment_df: pd.DataFrame,
            battery_df: pd.DataFrame, processes_names: List[str], finished_scanning_time: List[float],
            processes_ids: List[int]
    ):
        total_finishing_time = finished_scanning_time[-1]

        num_of_processes = len(processes_ids) + 1

        sub_cpu_df = slice_df(cpu_df, self.DEFAULT_SLICE_PERCENT).astype(float)
        sub_memory_df = slice_df(memory_df, self.DEFAULT_SLICE_PERCENT).astype(float)
        sub_disk_df = slice_df(disk_io_each_moment_df, self.DEFAULT_SLICE_PERCENT).astype(float)
        sub_network_df = slice_df(network_io_each_moment_df, self.DEFAULT_SLICE_PERCENT).astype(float)

        all_processes_df = get_all_df_by_id(processes_df, processes_ids)
        sub_all_processes_df = [slice_df(df, self.DEFAULT_SLICE_PERCENT) for df in all_processes_df]
        summary_df = pd.DataFrame(
            columns=["Metric", *processes_names, "Total"])

        summary_df.loc[len(summary_df.index)] = ["Duration", *([total_finishing_time for i in range(num_of_processes)])]

        # CPU
        cpu_all_processes = [pd.to_numeric(df[ProcessesColumns.CPU_CONSUMPTION]).mean() for df in sub_all_processes_df]
        cpu_total = sub_cpu_df[CPUColumns.USED_PERCENT].mean()
        summary_df.loc[len(summary_df.index)] = ["CPU", *cpu_all_processes, cpu_total]

        # Memory
        all_process_memory = [pd.to_numeric(df[ProcessesColumns.USED_MEMORY]).mean() for df in sub_all_processes_df]
        total_memory = sub_memory_df[MemoryColumns.USED_MEMORY].mean() * KB
        summary_df.loc[len(summary_df.index)] = ["Memory (MB)", *all_process_memory, total_memory]

        # Disk IO Read Bytes
        all_process_read_bytes = [pd.to_numeric(df[ProcessesColumns.READ_BYTES]).sum() for df in all_processes_df]
        total_read_bytes = sub_disk_df[DiskIOColumns.READ_BYTES].sum()
        summary_df.loc[len(summary_df.index)] = ["Disk IO Read (KB - sum)", *all_process_read_bytes, total_read_bytes]

        # Disk IO Read Count
        all_process_read_count = [pd.to_numeric(df[ProcessesColumns.READ_COUNT]).sum() for df in all_processes_df]
        total_read_count = sub_disk_df[DiskIOColumns.READ_COUNT].sum()
        summary_df.loc[len(summary_df.index)] = ["Disk IO Read Count (# - sum)", *all_process_read_count, total_read_count]

        # Disk IO Write Bytes
        all_process_write_bytes = [pd.to_numeric(df[ProcessesColumns.WRITE_BYTES]).sum() for df in all_processes_df]
        total_write_bytes = sub_disk_df[DiskIOColumns.WRITE_BYTES].sum()
        summary_df.loc[len(summary_df.index)] = ["Disk IO Write (KB - sum)", *all_process_write_bytes, total_write_bytes]

        # Disk IO Write Count
        all_process_write_count = [pd.to_numeric(df[ProcessesColumns.WRITE_COUNT]).sum() for df in all_processes_df]
        total_write_count = sub_disk_df[DiskIOColumns.WRITE_COUNT].sum()
        summary_df.loc[len(summary_df.index)] = ["Disk IO Write Count (# - sum)", *all_process_write_count, total_write_count]

        summary_df = AbstractSummaryBuilder.add_general_resource_metrics_info(
            summary_df, num_of_processes, sub_disk_df, sub_network_df, sub_all_processes_df
        )

        # Page Faults
        my_processes_page_faults = [pd.to_numeric(df[ProcessesColumns.PAGE_FAULTS]).sum() for df in all_processes_df]
        page_faults_all_processes = pd.to_numeric(processes_df[ProcessesColumns.PAGE_FAULTS]).sum()
        summary_df.loc[len(summary_df.index)] = ["Page Faults", *my_processes_page_faults, page_faults_all_processes]

        summary_df = AbstractSummaryBuilder.add_energy_info(summary_df, num_of_processes, battery_df)
        return summary_df

    def get_rows_colors(self) -> List[List[str]]:
        return [
            ['#FFFFFF'] * 1,    # Scan Duration Rows
            ['#ffff00'] * 1,    # CPU Consumption Rows
            ['#9CC2E5'] * 1,    # Memory Consumption Rows
            ['#66ff66'] * 2,    # I/O Read Rows
            ['#70ad47'] * 2,    # I/O Write Rows
            ['#cc66ff'] * 2,    # Disk I/O time Rows
            ['#00FFFF'] * 4,    # Network Consumption Rows
            ['#FFCC99'] * 1,    # Page Faults Rows
            ['#ffc000'] * 2,    # Energy Consumption Rows
            ['#FFFFFF'] * 1,    # Trees Translation Rows
        ]
