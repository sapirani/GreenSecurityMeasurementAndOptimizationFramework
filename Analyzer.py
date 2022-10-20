from datetime import timedelta
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class AxisInfo:
    def __init__(self, label, unit, axis):
        self.axis = axis
        self.label = label
        self.unit = unit


class Units(Enum):
    TIME = "Seconds"
    PERCENT = "% out of 100"
    CAPACITY = "mWatt/hour"
    VOLTAGE = "mVolt"
    COUNT = "#"
    MEMORY_TOTAL = "GB"
    MEMORY_PROCESS = "MB"
    DISK_TOTAL = "KB"


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


def draw_graph(df, graph_name, x_info, y_info, should_define_y=True, display_legend=False):

    if should_define_y: # if there is only one y-axis
        y_cols = [col.value for col in y_info.axis]
        df.plot(x=x_info.axis.value, y=y_cols, legend=display_legend)
        plt.ylabel(y_info.label) # naming the y-axis
    else:
        df.plot(x=x_info.axis.value, legend=display_legend)

    # naming the x-axis
    plt.xlabel({x_info.label, })

    # giving a title to the graph
    plt.title(graph_name)

    # change x to display time
    # plt.autofmt_xdate()
    # plt.set_xlim(0, timedelta(seconds=100))

    # save graph as picture
    plt.savefig(graph_name)

    # design graph

    # function to show the plot
    plt.show()


def display_battery_graphs():
    battery_df = pd.read_csv(BATTERY_FILE)

    # display capacity drain
    x_info_capacity = AxisInfo("Time ", Units.TIME, BatteryColumns.TIME)
    y_info_capacity = AxisInfo("Remaining Capacity ", Units.CAPACITY, [BatteryColumns.CAPACITY])
    draw_graph(battery_df, "Battery drop (mWh)", x_info_capacity, y_info_capacity)

    # display voltage drain
    x_info_voltage = AxisInfo("Time ", Units.TIME, BatteryColumns.TIME)
    y_info_voltage = AxisInfo("Voltage ", Units.VOLTAGE, [BatteryColumns.VOLTAGE])
    draw_graph(battery_df, "Battery drop (mV)", x_info_voltage, y_info_voltage)


def display_memory_graphs():
    memory_df = pd.read_csv(MEMORY_FILE)
    x_info = AxisInfo("Time ", Units.TIME, MemoryColumns.TIME)
    y_info = AxisInfo("Used Memory ", Units.MEMORY_TOTAL, [MemoryColumns.USED_MEMORY])
    draw_graph(memory_df, "Total Memory Consumption", x_info, y_info)


def display_disk_io_graphs():
    disk_io_df = pd.read_csv(DISK_FILE)

    # display number of io reads and writes
    x_info_count = AxisInfo("Time ", Units.TIME, DiskIOColumns.TIME)
    y_info_count = AxisInfo("Number of Accesses ", Units.COUNT, [DiskIOColumns.READ_COUNT, DiskIOColumns.WRITE_COUNT])
    draw_graph(disk_io_df, "Count of Disk IO actions", x_info_count, y_info_count, display_legend=True)

    # display number of bytes in io reads and writes
    x_info_bytes = AxisInfo("Time ", Units.TIME, DiskIOColumns.TIME)
    y_info_bytes = AxisInfo("Number of Bytes ", Units.DISK_TOTAL, [DiskIOColumns.READ_BYTES, DiskIOColumns.WRITE_BYTES])
    draw_graph(disk_io_df, "Number of bytes of Disk IO actions", x_info_bytes, y_info_bytes, display_legend=True)


def group_highest_processes(df, grouped_df, sort_by):
    sorted_mean_values = grouped_df.mean().sort_values(by=sort_by.value)
    top_processes = sorted_mean_values[-10:]
    all_top_processes = df.loc[df[ProcessesColumns.PROCESS_NAME.value].isin(top_processes.index)]
    all_top_processes_grouped = all_top_processes.groupby(ProcessesColumns.PROCESS_NAME.value)
    return all_top_processes_grouped


def display_processes_graphs():
    processes_df = pd.read_csv(PROCESSES_FILE)
    processes_df_grouped = processes_df.groupby(ProcessesColumns.PROCESS_NAME.value)

    # display CPU consumption
    all_top_processes_grouped_cpu = group_highest_processes(processes_df, processes_df_grouped,
                                                            ProcessesColumns.CPU_CONSUMPTION)

    x_info_cpu = AxisInfo("Time ", Units.TIME, ProcessesColumns.TIME)
    y_info_cpu = AxisInfo("CPU consumption ", Units.PERCENT, [])
    draw_graph(all_top_processes_grouped_cpu[ProcessesColumns.CPU_CONSUMPTION.value], "CPU consumption per process",
               x_info_cpu, y_info_cpu, should_define_y=False, display_legend=True)

    # display Memory
    all_top_processes_grouped_memory = group_highest_processes(processes_df, processes_df_grouped,
                                                               ProcessesColumns.USED_MEMORY)

    x_info_memory = AxisInfo("Time ", Units.TIME, ProcessesColumns.TIME)
    y_info_memory = AxisInfo("Memory consumption ", Units.MEMORY_PROCESS,
                             [DiskIOColumns.READ_COUNT, DiskIOColumns.WRITE_COUNT])
    draw_graph(all_top_processes_grouped_memory[ProcessesColumns.USED_MEMORY.value], "Memory consumption per process",
               x_info_memory, y_info_memory, should_define_y=False, display_legend=True)


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
