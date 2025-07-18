import pandas as pd

from utils.general_consts import ProcessesColumns, CPUColumns, MemoryColumns, KB, DiskIOColumns
from summary_builder.abstract_summary_builder import AbstractSummaryBuilder, slice_df, get_all_df_by_id


class SystemResourceIsolationSummaryBuilder(AbstractSummaryBuilder):
    def prepare_summary_csv(self, processes_df, cpu_df, memory_df, disk_io_each_moment_df, network_io_each_moment_df,
                            battery_df, processes_names, finished_scanning_time, processes_ids):
        total_finishing_time = finished_scanning_time[-1]

        num_of_processes = len(processes_ids) + 1

        sub_cpu_df = slice_df(cpu_df, 5).astype(float)
        sub_memory_df = slice_df(memory_df, 5).astype(float)
        sub_disk_df = slice_df(disk_io_each_moment_df, 0).astype(float)
        sub_network_df = slice_df(network_io_each_moment_df, 0).astype(float)

        all_processes_df = get_all_df_by_id(processes_df, processes_ids)
        sub_all_processes_df = [slice_df(df, 5) for df in all_processes_df]
        summary_df = pd.DataFrame(
            columns=["Metric", *processes_names, "System (total - all processes)"])

        summary_df.loc[len(summary_df.index)] = ["Duration", *([total_finishing_time for i in range(num_of_processes)])]

        # CPU
        cpu_all_processes = [pd.to_numeric(df[ProcessesColumns.CPU_CONSUMPTION]).mean() for df in sub_all_processes_df]
        cpu_total = sub_cpu_df[CPUColumns.USED_PERCENT].mean()
        cpu_system = cpu_total - sum(cpu_all_processes)
        cpu_total_without_process = [cpu_total - process_cpu for process_cpu in cpu_all_processes]
        summary_df.loc[len(summary_df.index)] = ["CPU Process", *cpu_all_processes, "X"]
        summary_df.loc[len(summary_df.index)] = ["CPU System (total - process)", *cpu_total_without_process, cpu_system]

        # Memory
        all_process_memory = [pd.to_numeric(df[ProcessesColumns.USED_MEMORY]).mean() for df in sub_all_processes_df]
        total_memory = sub_memory_df[MemoryColumns.USED_MEMORY].mean() * KB
        system_memory = total_memory - sum(all_process_memory)
        memory_total_without_process = [total_memory - process_memory for process_memory in all_process_memory]
        summary_df.loc[len(summary_df.index)] = ["Memory Process (MB)", *all_process_memory, "X"]
        summary_df.loc[len(summary_df.index)] = ["Memory Total (total - process) (MB)", *memory_total_without_process,
                                                 system_memory]

        # IO Read Bytes
        all_process_read_bytes = [pd.to_numeric(df[ProcessesColumns.READ_BYTES]).sum() for df in all_processes_df]
        total_read_bytes = sub_disk_df[DiskIOColumns.READ_BYTES].sum()
        system_read_bytes = total_read_bytes - sum(all_process_read_bytes)
        read_bytes_total_without_process = [total_read_bytes - process_read_bytes for process_read_bytes in
                                            all_process_read_bytes]
        summary_df.loc[len(summary_df.index)] = ["IO Read Process (KB - sum)", *all_process_read_bytes, "X"]
        summary_df.loc[len(summary_df.index)] = ["IO Read System (total - process) (KB - sum)",
                                                 *read_bytes_total_without_process, system_read_bytes]

        # IO Read Count
        all_process_read_count = [pd.to_numeric(df[ProcessesColumns.READ_COUNT]).sum() for df in all_processes_df]
        total_read_count = sub_disk_df[DiskIOColumns.READ_COUNT].sum()
        system_read_count = total_read_count - sum(all_process_read_count)
        read_count_total_without_process = [total_read_count - process_read_count for process_read_count in
                                            all_process_read_count]
        summary_df.loc[len(summary_df.index)] = ["IO Read Count Process (# - sum)", *all_process_read_count, "X"]
        summary_df.loc[len(summary_df.index)] = ["IO Read Count System (total - process) (# - sum)",
                                                 *read_count_total_without_process, system_read_count]

        # IO Write Bytes
        all_process_write_bytes = [pd.to_numeric(df[ProcessesColumns.WRITE_BYTES]).sum() for df in all_processes_df]
        total_write_bytes = sub_disk_df[DiskIOColumns.WRITE_BYTES].sum()
        system_write_bytes = total_write_bytes - sum(all_process_write_bytes)
        write_bytes_total_without_process = [total_write_bytes - process_write_bytes for process_write_bytes in
                                             all_process_write_bytes]
        summary_df.loc[len(summary_df.index)] = ["IO Write Process (KB - sum)", *all_process_write_bytes, "X"]
        summary_df.loc[len(summary_df.index)] = ["IO Write System (total - process) (KB - sum)",
                                                 *write_bytes_total_without_process, system_write_bytes]

        # IO Write Count
        all_process_write_count = [pd.to_numeric(df[ProcessesColumns.WRITE_COUNT]).sum() for df in all_processes_df]
        total_write_count = sub_disk_df[DiskIOColumns.WRITE_COUNT].sum()
        system_write_count = total_write_count - sum(all_process_write_count)
        write_count_total_without_process = [total_write_count - process_write_count for process_write_count in
                                             all_process_write_count]
        summary_df.loc[len(summary_df.index)] = ["IO Write Count Process (# - sum)", *all_process_write_count, "X"]
        summary_df.loc[len(summary_df.index)] = ["IO Write Count System (total - process) (# - sum)",
                                                 *write_count_total_without_process, system_write_count]

        summary_df = AbstractSummaryBuilder.add_general_resource_metrics_info(
            summary_df, num_of_processes, sub_disk_df, sub_network_df, sub_all_processes_df
        )

        # Page Faults
        my_processes_page_faults = [pd.to_numeric(df[ProcessesColumns.PAGE_FAULTS]).sum() for df in all_processes_df]
        page_faults_all_processes = pd.to_numeric(processes_df[ProcessesColumns.PAGE_FAULTS]).sum()
        page_faults_system = page_faults_all_processes - sum(my_processes_page_faults)
        summary_df.loc[len(summary_df.index)] = ["Page Faults", *my_processes_page_faults, page_faults_system]

        summary_df = AbstractSummaryBuilder.add_energy_info(summary_df, num_of_processes, battery_df)
        return summary_df

    def get_colors(self):
        return [
            ['#FFFFFF'] * 1,    # Scan Duration Rows
            ['#ffff00'] * 2,    # CPU Consumption Rows
            ['#9CC2E5'] * 2,    # Memory Consumption Rows
            ['#66ff66'] * 4,    # I/O Read Rows
            ['#70ad47'] * 4,    # I/O Write Rows
            ['#cc66ff'] * 2,    # Disk I/O time Rows
            ['#00FFFF'] * 4,    # Network Consumption Rows
            ['#FFCC99'] * 1,    # Page Faults Rows
            ['#ffc000'] * 2,    # Energy Consumption Rows
            ['#FFFFFF'] * 1,    # Trees Translation Rows
        ]
