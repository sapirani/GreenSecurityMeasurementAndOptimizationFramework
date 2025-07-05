import pandas as pd

from general_consts import ProcessesColumns, CPUColumns, MemoryColumns, KB, DiskIOColumns, NetworkIOColumns
from general_functions import calc_delta_capacity, convert_mwh_to_other_metrics


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


class SummaryBuilderInterface:
    def prepare_summary_csv(self, processes_df, cpu_df, memory_df, disk_io_each_moment_df, network_io_each_moment_df,
                            battery_df, processes_names, finished_scanning_time, processes_ids):
        return None

    def colors_func(self, df):
        return None

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


class DuduSummary(SummaryBuilderInterface):
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

        # Page Faults
        my_processes_page_faults = [pd.to_numeric(df[ProcessesColumns.PAGE_FAULTS]).sum() for df in all_processes_df]
        page_faults_all_processes = pd.to_numeric(processes_df[ProcessesColumns.PAGE_FAULTS]).sum()
        page_faults_system = page_faults_all_processes - sum(my_processes_page_faults)
        summary_df.loc[len(summary_df.index)] = ["Page Faults", *my_processes_page_faults, page_faults_system]

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

        return SummaryBuilderInterface.add_general_info(summary_df, num_of_processes, battery_df, sub_disk_df,
                                                        sub_network_df, sub_all_processes_df)

    def colors_func(self, df):
        return ['background-color: #FFFFFF'] + \
               ['background-color: #ffff00' for _ in range(2)] + ['background-color: #9CC2E5' for _ in range(3)] + \
               ['background-color: #66ff66' for _ in range(4)] + ['background-color: #70ad47' for _ in range(4)] + \
               ['background-color: #cc66ff' for _ in range(2)] + \
               ['background-color: #00FFFF' for _ in range(4)] + \
               ['background-color: #ffc000' for _ in range(2)] + \
               ['background-color: #FFFFFF']


class OtherSummary(SummaryBuilderInterface):
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

        # Page Faults
        my_processes_page_faults = [pd.to_numeric(df[ProcessesColumns.PAGE_FAULTS]).sum() for df in all_processes_df]
        page_faults_all_processes = pd.to_numeric(processes_df[ProcessesColumns.PAGE_FAULTS]).sum()
        summary_df.loc[len(summary_df.index)] = ["Page Faults", *my_processes_page_faults, page_faults_all_processes]

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

        return SummaryBuilderInterface.add_general_info(summary_df, num_of_processes, battery_df, sub_disk_df,
                                                        sub_network_df, sub_all_processes_df)

    def colors_func(self, df):
        return ['background-color: #FFFFFF'] + \
               ['background-color: #ffff00' for _ in range(1)] + ['background-color: #9CC2E5' for _ in range(2)] + \
               ['background-color: #66ff66' for _ in range(2)] + ['background-color: #70ad47' for _ in range(2)] + \
               ['background-color: #cc66ff' for _ in range(2)] + \
               ['background-color: #00FFFF' for _ in range(4)] + \
               ['background-color: #ffc000' for _ in range(2)] + \
               ['background-color: #FFFFFF']
