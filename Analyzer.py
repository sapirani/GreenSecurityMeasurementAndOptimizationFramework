import matplotlib.pyplot as plt
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


DEFAULT = "default"
ANTIVIRUS_PROCESS_NAME = "MsMpEng.exe"


def design_and_plot(x_info, y_info, graph_name, total_path=DEFAULT, total_index=DEFAULT, total_column=DEFAULT):
    # naming the x-axis
    plt.xlabel(x_info.label + " (in " + x_info.unit + ")", color='crimson', labelpad=10, fontname="Comic Sans MS")

    # naming the y-axis
    plt.ylabel(y_info.label + " (in " + y_info.unit + ")", color='crimson', labelpad=10,
               fontname="Comic Sans MS")  # naming the y-axis

    # giving a title to the graph, changing it's font, size and color
    plt.title(graph_name, color="darkblue", fontsize=20, fontname="Times New Roman", fontweight="bold")

    # design graph
    plt.rc('axes', labelsize=12)  # Set the axes labels font size
    plt.rc('legend', fontsize=6)  # Set the legend font size
    plt.xticks(fontsize=8, color='darkgray')  # change x ticks color
    plt.yticks(fontsize=8, color='darkgray')  # change y ticks color
    plt.rcParams["figure.figsize"] = (10, 5)  # change figure size

    # save graph as picture
    plt.savefig(os.path.join(GRAPHS_DIR, graph_name))

    # function to show the plot
    plt.show()


def check_plot(ax, total_path=DEFAULT, total_index=DEFAULT, total_column=DEFAULT):
    if total_path != DEFAULT:
        df_total = pd.read_csv(total_path, index_col=total_index)

        if total_column != DEFAULT:
            df_total = df_total[total_column]
            print(df_total)

        df_total.plot(color='black', ax=ax).legend(labels=["Total Consumption"])


def draw_grouped_dataframe(df, graph_name, x_info, y_info, total_path=DEFAULT, total_index=DEFAULT,
                           total_column=DEFAULT):
    fig, ax = plt.subplots(figsize=(18, 6))

    check_plot(ax, total_path, total_index, total_column)

    for group_name, group in df:
        proc_name = group_name[1]
        group.plot(y=y_info.axis, ax=ax, label=proc_name, linewidth=(5 if proc_name == ANTIVIRUS_PROCESS_NAME
                                                                         else 2))

    design_and_plot(x_info, y_info, graph_name)


def draw_dataframe(df, graph_name, x_info, y_info):
    df[y_info.axis].plot(legend=len(y_info.axis) > 1)
    design_and_plot(x_info, y_info, graph_name)


def display_battery_graphs():
    battery_df = pd.read_csv(BATTERY_STATUS_CSV, index_col=BatteryColumns.TIME)

    # display capacity drain
    x_info_capacity = AxisInfo("Time", Units.TIME, BatteryColumns.TIME)
    y_info_capacity = AxisInfo("Remaining Capacity", Units.CAPACITY, [BatteryColumns.CAPACITY])
    draw_dataframe(battery_df, "Battery drop (mWh)", x_info_capacity, y_info_capacity)

    # display voltage drain
    x_info_voltage = AxisInfo("Time", Units.TIME, BatteryColumns.TIME)
    y_info_voltage = AxisInfo("Voltage", Units.VOLTAGE, [BatteryColumns.VOLTAGE])
    draw_dataframe(battery_df, "Battery drop (mV)", x_info_voltage, y_info_voltage)


def display_cpu_graphs():
    memory_df = pd.read_csv(TOTAL_CPU_CSV, index_col=CPUColumns.TIME)
    x_info = AxisInfo("Time", Units.TIME, CPUColumns.TIME)
    y_info = AxisInfo("Used CPU", Units.PERCENT, [CPUColumns.USED_PERCENT])
    draw_dataframe(memory_df, "Total CPU Consumption", x_info, y_info)


def display_memory_graphs():
    memory_df = pd.read_csv(TOTAL_MEMORY_EACH_MOMENT_CSV, index_col=MemoryColumns.TIME)
    x_info = AxisInfo("Time", Units.TIME, MemoryColumns.TIME)
    y_info = AxisInfo("Used Memory", Units.MEMORY_TOTAL, [MemoryColumns.USED_MEMORY])
    draw_dataframe(memory_df, "Total Memory Consumption", x_info, y_info)


def display_disk_io_graphs():
    disk_io_df = pd.read_csv(DISK_IO_EACH_MOMENT, index_col=DiskIOColumns.TIME)

    # display number of io reads and writes
    x_info_count = AxisInfo("Time", Units.TIME, DiskIOColumns.TIME)
    y_info_count = AxisInfo("Number of Accesses", Units.COUNT, [DiskIOColumns.READ_COUNT, DiskIOColumns.WRITE_COUNT])
    draw_dataframe(disk_io_df, "Count of Disk IO actions", x_info_count, y_info_count)

    # display number of bytes in io reads and writes
    x_info_bytes = AxisInfo("Time", Units.TIME, DiskIOColumns.TIME)
    y_info_bytes = AxisInfo("Number of Bytes", Units.IO_BYTES, [DiskIOColumns.READ_BYTES, DiskIOColumns.WRITE_BYTES])
    draw_dataframe(disk_io_df, "Number of bytes of Disk IO actions", x_info_bytes, y_info_bytes)


def group_highest_processes(df, grouped_df, sort_by, group_by):
    sorted_mean_values = grouped_df.mean().sort_values(by=sort_by)
    top_processes = sorted_mean_values[-10:]
    all_top_processes = df.loc[df[group_by[0]].isin(top_processes.index)]
    all_top_processes_grouped = all_top_processes.groupby(group_by)
    return all_top_processes_grouped


def display_processes_graphs():
    processes_df = pd.read_csv(PROCESSES_CSV, index_col=ProcessesColumns.TIME)
    processes_df_grouped = processes_df.groupby(ProcessesColumns.PROCESS_ID)

    # display CPU consumption
    all_top_processes_grouped_cpu = group_highest_processes(processes_df, processes_df_grouped,
                                                            ProcessesColumns.CPU_CONSUMPTION,
                                                            [ProcessesColumns.PROCESS_ID,
                                                             ProcessesColumns.PROCESS_NAME])

    x_info_cpu = AxisInfo("Time", Units.TIME, ProcessesColumns.TIME)
    y_info_cpu = AxisInfo("CPU consumption", Units.PERCENT, ProcessesColumns.CPU_CONSUMPTION)
    draw_grouped_dataframe(all_top_processes_grouped_cpu, "CPU consumption per process",
                           x_info_cpu, y_info_cpu, TOTAL_CPU_CSV, CPUColumns.TIME)

    # display Memory
    all_top_processes_grouped_memory = group_highest_processes(processes_df, processes_df_grouped,
                                                               ProcessesColumns.USED_MEMORY,
                                                               [ProcessesColumns.PROCESS_ID,
                                                                ProcessesColumns.PROCESS_NAME])

    x_info_memory = AxisInfo("Time", Units.TIME, ProcessesColumns.TIME)
    y_info_memory = AxisInfo("Memory consumption", Units.MEMORY_PROCESS, ProcessesColumns.USED_MEMORY)
    draw_grouped_dataframe(all_top_processes_grouped_memory, "Memory consumption per process",
                           x_info_memory, y_info_memory, TOTAL_MEMORY_EACH_MOMENT_CSV, MemoryColumns.TIME,
                           MemoryColumns.USED_MEMORY)

    # display IO read bytes
    all_top_processes_grouped_read = group_highest_processes(processes_df, processes_df_grouped,
                                                             ProcessesColumns.READ_BYTES,
                                                             [ProcessesColumns.PROCESS_ID,
                                                              ProcessesColumns.PROCESS_NAME])

    x_info_read = AxisInfo("Time", Units.TIME, ProcessesColumns.TIME)
    y_info_read = AxisInfo("IO Read bytes", Units.IO_BYTES, ProcessesColumns.READ_BYTES)
    draw_grouped_dataframe(all_top_processes_grouped_read, "IO read bytes per process",
                           x_info_read, y_info_read)

    # display IO write bytes
    all_top_processes_grouped_write = group_highest_processes(processes_df, processes_df_grouped,
                                                              ProcessesColumns.WRITE_BYTES,
                                                              [ProcessesColumns.PROCESS_ID,
                                                               ProcessesColumns.PROCESS_NAME])

    x_info_write = AxisInfo("Time", Units.TIME, ProcessesColumns.TIME)
    y_info_write = AxisInfo("IO Write bytes", Units.IO_BYTES, ProcessesColumns.WRITE_BYTES)
    draw_grouped_dataframe(all_top_processes_grouped_write, "IO write bytes per process",
                           x_info_write, y_info_write)

    # display io read count
    all_top_processes_grouped_num_of_read = group_highest_processes(processes_df, processes_df_grouped,
                                                                    ProcessesColumns.READ_COUNT,
                                                                    [ProcessesColumns.PROCESS_ID,
                                                                     ProcessesColumns.PROCESS_NAME])

    x_info_read_count = AxisInfo("Time", Units.TIME, ProcessesColumns.TIME)
    y_info_read_count = AxisInfo("IO Read count", Units.COUNT, ProcessesColumns.READ_COUNT)
    draw_grouped_dataframe(all_top_processes_grouped_num_of_read, "IO read count per process",
                           x_info_read_count, y_info_read_count)

    # display io write count
    all_top_processes_grouped_num_of_write = group_highest_processes(processes_df, processes_df_grouped,
                                                                     ProcessesColumns.WRITE_COUNT,
                                                                     [ProcessesColumns.PROCESS_ID,
                                                                      ProcessesColumns.PROCESS_NAME])

    x_info_write_count = AxisInfo("Time", Units.TIME, ProcessesColumns.TIME)
    y_info_write_count = AxisInfo("IO Write count", Units.COUNT, ProcessesColumns.WRITE_COUNT)
    draw_grouped_dataframe(all_top_processes_grouped_num_of_write, "IO write count per process",
                           x_info_write_count, y_info_write_count)


def main():
    # battery table
    display_battery_graphs()

    # total cpu table
    display_cpu_graphs()

    # total memory table
    display_memory_graphs()

    # total disk io table
    display_disk_io_graphs()

    # processes table
    display_processes_graphs()


if __name__ == '__main__':
    main()
