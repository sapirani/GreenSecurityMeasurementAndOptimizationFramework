from datetime import timedelta
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd


class BatteryColumns(Enum):
    TIME = "Time(sec)"
    BATTERY = "REMAINING BATTERY(%)"
    CAPACITY = "REMAINING CAPACITY(mWh)"
    VOLTAGE = "Voltage(mV)"


class MemoryColumns(Enum):
    TIME = "Time(sec)"
    USED_MEMORY = "Used(GB)"
    USED_PERCENT = "Percentage"


class DiskIOColumns(Enum):
    TIME = "Time(sec)"
    READ_COUNT = "READ(#)"
    WRITE_COUNT = "WRITE(#)"
    READ_BYTES = "READ(KB)"
    WRITE_BYTES = "WRITE(KB)"


class ProcessesColumns(Enum):
    TIME = "Time(sec)"
    PROCESS_ID = "PID"
    PROCESS_NAME = "PNAME"
    CPU_CONSUMPTION = "CPU(%)"
    NUMBER_OF_THREADS = "NUM THREADS"
    USED_MEMORY = "MEMORY(MB)"
    MEMORY_PERCENT = "MEMORY(%)"
    READ_COUNT = "read_count"
    WRITE_COUNT = "write_count"
    READ_BYTES = "read_bytes"
    WRITE_BYTES = "write_bytes"


DISK_FILE = "disk_io_each_moment.csv"
MEMORY_FILE = "total_memory_each_moment.csv"
PROCESSES_FILE = "processes_data.csv"
BATTERY_FILE = "battery_status.csv"
GB = 2 ** 30
MB = 2 ** 20
KB = 2 ** 10
ONE_GRAPH = 1
COMBINED_GRAPH = 2
DEFAULT_Y_LABLE = "DEFAULT"


def read_file_to_dataframe(path):
    df = pd.read_csv(path)

    # if len(columns_names) == 0:
    #     return df
    #
    # df_new = df[columns_names]
    # df_new.plot()
    return df


def draw_graph(df, x_col, y_cols, x_name, graph_name, should_define_y=True, y_name=DEFAULT_Y_LABLE, display_legend=False):
    # define the x, y-axis and remove the legend from display

    if should_define_y:
        y_cols = [col.value for col in y_cols]
        df.plot(x=x_col.value, y=y_cols, legend=display_legend)
    else:
        df.plot(x=x_col.value, legend=display_legend)

    # naming the x-axis
    plt.xlabel(x_name)
    # naming the y-axis
    if y_name != DEFAULT_Y_LABLE:
        plt.ylabel(y_name)

    # giving a title to my graph
    plt.title(graph_name)

    # change x to display time
    #plt.autofmt_xdate()
    #plt.set_xlim(0, timedelta(seconds=100))

    # save graph as picture
    plt.savefig(graph_name)

    # function to show the plot
    plt.show()


def display_battery_graphs():
    battery_df = read_file_to_dataframe(BATTERY_FILE)
    draw_graph(battery_df, BatteryColumns.TIME, [BatteryColumns.CAPACITY], "Time (sec)", "Battery drop (mWh)",
               y_name="Remaining Capacity (mWh)")
    draw_graph(battery_df, BatteryColumns.TIME, [BatteryColumns.VOLTAGE], "Time (sec)", "Battery drop (mV)",
               y_name="Voltage (mV)")


def display_memory_graphs():
    memory_df = read_file_to_dataframe(MEMORY_FILE)
    draw_graph(memory_df, MemoryColumns.TIME, [MemoryColumns.USED_MEMORY], "Time (sec)", "Total Memory Consumption",
               y_name="Used Memory (GB)")


def display_disk_io_graphs():
    disk_io_df = read_file_to_dataframe(DISK_FILE)
    draw_graph(disk_io_df, DiskIOColumns.TIME, [DiskIOColumns.READ_COUNT, DiskIOColumns.WRITE_COUNT], "Time (sec)",
               "Count of Disk IO actions", "Number of Accesses", display_legend=True)
    draw_graph(disk_io_df, DiskIOColumns.TIME, [DiskIOColumns.READ_BYTES, DiskIOColumns.WRITE_BYTES], "Time (sec)",
               "Number of bytes of Disk IO actions", y_name="Number of Bytes (KB)", display_legend=True)


def group_highest_processes(df, grouped_df, sort_by):
    sorted_mean_values = grouped_df.mean().sort_values(by=sort_by.value)
    top_processes = sorted_mean_values[-10:]
    all_top_processes = df.loc[df[ProcessesColumns.PROCESS_NAME.value].isin(top_processes.index)]
    all_top_processes_grouped = all_top_processes.groupby(ProcessesColumns.PROCESS_NAME.value)
    return all_top_processes_grouped


def display_processes_graphs():
    processes_df = read_file_to_dataframe(PROCESSES_FILE)
    processes_df_grouped = processes_df.groupby(ProcessesColumns.PROCESS_NAME.value)

    # display CPU consumption
    all_top_processes_grouped_cpu = group_highest_processes(processes_df, processes_df_grouped, ProcessesColumns.CPU_CONSUMPTION)
    draw_graph(all_top_processes_grouped_cpu[ProcessesColumns.CPU_CONSUMPTION.value], ProcessesColumns.TIME, [],
               "Time (sec)", "CPU consumption per process", False, "CPU Consumption", True)

    # display Memory
    all_top_processes_grouped_memory = group_highest_processes(processes_df, processes_df_grouped, ProcessesColumns.USED_MEMORY)
    draw_graph(all_top_processes_grouped_memory[ProcessesColumns.USED_MEMORY.value], ProcessesColumns.TIME, [],
               "Time (sec)", "Memory consumption per process", False, "Memory Consumption", True)


def main():
    # battery table
    display_battery_graphs()

    # total memory table
    display_memory_graphs()

    # total disk io table
    display_disk_io_graphs()

    # processes table
    display_processes_graphs()


if __name__ == '__main__':
    main()
