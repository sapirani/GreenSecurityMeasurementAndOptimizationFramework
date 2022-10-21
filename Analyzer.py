# from datetime import timedelta
# from enum import Enum
import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
from configurations import *


class AxisInfo:
    def __init__(self, label, unit, axis):
        self.axis = axis
        self.label = label
        self.unit = unit


class Units:
    TIME = "Seconds"
    PERCENT = "% out of 100"
    CAPACITY = "mWatt/hour"
    VOLTAGE = "mVolt"
    COUNT = "#"
    MEMORY_TOTAL = "GB"
    MEMORY_PROCESS = "MB"
    IO_BYTES = "KB"


"""class BatteryColumns(Enum):
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
"""

"""DISK_FILE = "disk_io_each_moment.csv"
MEMORY_FILE = "total_memory_each_moment.csv"
PROCESSES_FILE = "processes_data.csv"
BATTERY_FILE = "battery_status.csv"""
GB = 2 ** 30
MB = 2 ** 20
KB = 2 ** 10
ONE_GRAPH = 1
COMBINED_GRAPH = 2
DEFAULT_Y_LABLE = "DEFAULT"


def draw_graph(df, graph_name, x_info, y_info):
    y_cols = [col for col in y_info.axis]
    df.columns.intersection(y_cols).plot(x=x_info.axis, y=y_cols, legend=len(y_cols) > 1)
    # naming the y-axis
    plt.ylabel(y_info.label + " (in " + y_info.unit + ")", color='crimson', labelpad=10,
               fontname="Comic Sans MS")  # naming the y-axis

    # naming the x-axis
    plt.xlabel(x_info.label + " (in " + x_info.unit + ")", color='crimson', labelpad=10, fontname="Comic Sans MS")

    # giving a title to the graph, changing it's font, size and color
    plt.title(graph_name, color="darkblue", fontsize=20, fontname="Times New Roman", fontweight="bold")  # Set the



    # changing legend to display processes names if necessary
    # if not is_total_table:  # meaning, processes table
    #   plt.legend(df.index.values(), fancybox=True, framealpha=1, shadow=True, borderpad=1)

    # change x to display time
    # plt.autofmt_xdate()
    # plt.set_xlim(0, timedelta(seconds=100))

    # design graph
    plt.rc('axes', labelsize=12)  # Set the axes labels font size
    plt.rc('legend', fontsize=6)  # Set the legend font size
    plt.xticks(fontsize=8, color='darkgray')  # change x ticks color
    plt.yticks(fontsize=8, color='darkgray')  # change y ticks color
    plt.subplots_adjust(left=0.1, right=0.90, top=0.93, bottom=0.2)
    plt.rcParams["figure.figsize"] = (25, 5)  # change figure size

    # save graph as picture
    plt.savefig(os.path.join(GRAPHS_DIR, graph_name))

    # design graph

    # function to show the plot
    plt.show()


def display_battery_graphs():
    battery_df = pd.read_csv(BATTERY_STATUS_CSV)

    # display capacity drain
    x_info_capacity = AxisInfo("Time", Units.TIME, BatteryColumns.TIME)
    y_info_capacity = AxisInfo("Remaining Capacity", Units.CAPACITY, [BatteryColumns.CAPACITY])
    draw_graph(battery_df, "Battery drop (mWh)", x_info_capacity, y_info_capacity)

    # display voltage drain
    x_info_voltage = AxisInfo("Time", Units.TIME, BatteryColumns.TIME)
    y_info_voltage = AxisInfo("Voltage", Units.VOLTAGE, [BatteryColumns.VOLTAGE])
    draw_graph(battery_df, "Battery drop (mV)", x_info_voltage, y_info_voltage)


def display_memory_graphs():
    memory_df = pd.read_csv(TOTAL_MEMORY_EACH_MOMENT_CSV)
    x_info = AxisInfo("Time", Units.TIME, MemoryColumns.TIME)
    y_info = AxisInfo("Used Memory", Units.MEMORY_TOTAL, [MemoryColumns.USED_MEMORY])
    draw_graph(memory_df, "Total Memory Consumption", x_info, y_info)


def display_disk_io_graphs():
    disk_io_df = pd.read_csv(DISK_IO_EACH_MOMENT, index_col=DiskIOColumns.TIME)

    # display number of io reads and writes
    x_info_count = AxisInfo("Time", Units.TIME, DiskIOColumns.TIME)
    y_info_count = AxisInfo("Number of Accesses", Units.COUNT, [DiskIOColumns.READ_COUNT, DiskIOColumns.WRITE_COUNT])
    draw_graph(disk_io_df, "Count of Disk IO actions", x_info_count, y_info_count)

    # display number of bytes in io reads and writes
    x_info_bytes = AxisInfo("Time", Units.TIME, DiskIOColumns.TIME)
    y_info_bytes = AxisInfo("Number of Bytes", Units.IO_BYTES, [DiskIOColumns.READ_BYTES, DiskIOColumns.WRITE_BYTES])
    draw_graph(disk_io_df, "Number of bytes of Disk IO actions", x_info_bytes, y_info_bytes)


def group_highest_processes(df, grouped_df, sort_by, group_by):
    sorted_mean_values = grouped_df.mean().sort_values(by=sort_by)
    top_processes = sorted_mean_values[-10:]
    all_top_processes = df.loc[df[group_by].isin(top_processes.index)]
    all_top_processes_grouped = all_top_processes.groupby(group_by)
    return all_top_processes_grouped


def display_processes_graphs():
    processes_df = pd.read_csv(PROCESSES_CSV, index_col=ProcessesColumns.TIME)
    processes_df_grouped = processes_df.groupby(ProcessesColumns.PROCESS_ID)

    # display CPU consumption
    all_top_processes_grouped_cpu = group_highest_processes(processes_df, processes_df_grouped,
                                                            ProcessesColumns.CPU_CONSUMPTION,
                                                            ProcessesColumns.PROCESS_ID)

    x_info_cpu = AxisInfo("Time", Units.TIME, ProcessesColumns.TIME)
    y_info_cpu = AxisInfo("CPU consumption", Units.PERCENT, [ProcessesColumns.CPU_CONSUMPTION])
    draw_graph(all_top_processes_grouped_cpu, "CPU consumption per process",
               x_info_cpu, y_info_cpu)

    # display Memory
    all_top_processes_grouped_memory = group_highest_processes(processes_df, processes_df_grouped,
                                                               ProcessesColumns.USED_MEMORY,
                                                               ProcessesColumns.PROCESS_ID)

    x_info_memory = AxisInfo("Time", Units.TIME, ProcessesColumns.TIME)
    y_info_memory = AxisInfo("Memory consumption", Units.MEMORY_PROCESS, [])
    draw_graph(all_top_processes_grouped_memory[ProcessesColumns.USED_MEMORY], "Memory consumption per process",
               x_info_memory, y_info_memory)

    # display IO read bytes
    all_top_processes_grouped_read = group_highest_processes(processes_df, processes_df_grouped,
                                                             ProcessesColumns.READ_BYTES,
                                                             ProcessesColumns.PROCESS_ID)

    x_info_read = AxisInfo("Time", Units.TIME, ProcessesColumns.TIME)
    y_info_read = AxisInfo("IO Read bytes", Units.IO_BYTES, [])
    draw_graph(all_top_processes_grouped_read[ProcessesColumns.READ_BYTES], "IO read bytes per process",
               x_info_read, y_info_read)

    # display IO write bytes
    all_top_processes_grouped_write = group_highest_processes(processes_df, processes_df_grouped,
                                                              ProcessesColumns.WRITE_BYTES,
                                                              ProcessesColumns.PROCESS_ID)

    x_info_write = AxisInfo("Time", Units.TIME, ProcessesColumns.TIME)
    y_info_write = AxisInfo("IO Write bytes", Units.IO_BYTES, [])
    draw_graph(all_top_processes_grouped_write[ProcessesColumns.WRITE_BYTES], "IO write bytes per process",
               x_info_write, y_info_write)

    # display io read count
    all_top_processes_grouped_num_of_read = group_highest_processes(processes_df, processes_df_grouped,
                                                                    ProcessesColumns.READ_COUNT,
                                                                    ProcessesColumns.PROCESS_ID)

    x_info_read_count = AxisInfo("Time", Units.TIME, ProcessesColumns.TIME)
    y_info_read_count = AxisInfo("IO Read count", Units.COUNT, [])
    draw_graph(all_top_processes_grouped_num_of_read[ProcessesColumns.READ_COUNT], "IO read count per process",
               x_info_read_count, y_info_read_count)

    # display io write count
    all_top_processes_grouped_num_of_write = group_highest_processes(processes_df, processes_df_grouped,
                                                                     ProcessesColumns.WRITE_COUNT,
                                                                     ProcessesColumns.PROCESS_ID)

    x_info_write_count = AxisInfo("Time", Units.TIME, ProcessesColumns.TIME)
    y_info_write_count = AxisInfo("IO Write count", Units.COUNT, [])
    draw_graph(all_top_processes_grouped_num_of_write[ProcessesColumns.WRITE_COUNT], "IO write count per process",
               x_info_write_count, y_info_write_count)


def main():
    # battery table
    # display_battery_graphs()

    # total memory table
    # display_memory_graphs()

    # total disk io table
    #display_disk_io_graphs()

    # processes table
    display_processes_graphs()


if __name__ == '__main__':
    main()
