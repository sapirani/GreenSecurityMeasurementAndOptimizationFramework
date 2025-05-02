import os
from typing import List

import pandas as pd

from general_consts import BatteryColumns, CPUColumns, MemoryColumns, DiskIOColumns, ProcessesColumns, KB
from results_analyzer.analyzer_constants import AxisInfo, Units, DEFAULT, MINIMAL_REQUIRED_RECORDS
from results_analyzer.graphing_utils import draw_dataframe, draw_subplots, draw_processes_and_total

NUM_OF_PATHS = 12
MAX_PROCESSES_TO_PLOT = 10


class GraphsGenerator:
    def __init__(self, measurement_dir: str, container_name: str):
        self.__container_name = container_name
        self.__base_dir = measurement_dir
        self.__graphs_output_dir = os.path.join(measurement_dir, "graphs")
        self.__stdout_files_dir = os.path.join(measurement_dir, "stdouts")
        self.__stderr_files_dir = os.path.join(measurement_dir, "stderrs")
        self.__processes_results_csv = os.path.join(measurement_dir, 'processes_data.csv')
        self.__total_memory_each_moment_csv = os.path.join(measurement_dir, 'total_memory_each_moment.csv')
        self.__disk_io_each_moment_csv = os.path.join(measurement_dir, 'disk_io_each_moment.csv')
        self.__network_io_each_moment_csv = os.path.join(measurement_dir, 'network_io_each_moment.csv')
        self.__battery_status_csv = os.path.join(measurement_dir, 'battery_status.csv')
        self.__general_information_csv = os.path.join(measurement_dir, 'general_information.txt')
        self.__total_cpu_csv = os.path.join(measurement_dir, 'total_cpu.csv')
        self.__summary_csv = os.path.join(measurement_dir, 'summary.xlsx')

        if not os.path.exists(self.__graphs_output_dir):
            os.makedirs(self.__graphs_output_dir)

    def get_container_name(self) -> str:
        return self.__container_name

    def get_graphs_output_dir(self) -> str:
        return self.__graphs_output_dir

    def get_base_dir(self) -> str:
        return self.__base_dir

    def __get_graph_name(self, title: str, for_comparison: bool = False) -> str:
        if for_comparison:
            title = f"{title} - Comparison"
        return f"{title} Graph for {self.__container_name}"

    def display_battery_graphs(self):
        if os.path.exists(self.__battery_status_csv):
            battery_df = pd.read_csv(self.__battery_status_csv, index_col=BatteryColumns.TIME)

            # display capacity drain
            x_info_capacity = AxisInfo("Time", Units.TIME, BatteryColumns.TIME)
            y_info_capacity = AxisInfo("Remaining Capacity", Units.CAPACITY, [BatteryColumns.CAPACITY])
            draw_dataframe(battery_df, self.__graphs_output_dir, self.__get_graph_name("Battery drop (mWh)"),
                           x_info_capacity, y_info_capacity)

            # display voltage drain
            x_info_voltage = AxisInfo("Time", Units.TIME, BatteryColumns.TIME)
            y_info_voltage = AxisInfo("Voltage", Units.VOLTAGE, [BatteryColumns.VOLTAGE])
            draw_dataframe(battery_df, self.__graphs_output_dir, self.__get_graph_name("Battery drop (mV)"),
                           x_info_voltage, y_info_voltage)
        else:
            print(f"The battery status file {self.__battery_status_csv} does not exist.")

    def display_cpu_graphs(self):
        if os.path.exists(self.__total_cpu_csv):
            cpu_df = pd.read_csv(self.__total_cpu_csv)  # , index_col=CPUColumns.TIME)
            x_info = AxisInfo("Time", Units.TIME, [CPUColumns.TIME])
            y_info = AxisInfo("Used CPU", Units.PERCENT, cpu_df.columns.tolist()[1:])
            draw_subplots(cpu_df, self.__graphs_output_dir, x_info, y_info,
                          self.__get_graph_name("Total CPU Consumption per core"))

            # one graph with all plots
            emphasis_column = y_info.axis[0] if y_info.axis else None
            draw_dataframe(cpu_df, self.__graphs_output_dir,
                           self.__get_graph_name("Total CPU Consumption for ALL Cores"), x_info, y_info,
                           column_to_emphasis=emphasis_column)

        else:
            print(f"The total CPU usage file {self.__total_cpu_csv} does not exist.")

    def display_memory_graphs(self):
        if os.path.exists(self.__total_memory_each_moment_csv):
            memory_df = pd.read_csv(self.__total_memory_each_moment_csv, index_col=MemoryColumns.TIME)
            x_info = AxisInfo("Time", Units.TIME, MemoryColumns.TIME)
            y_info = AxisInfo("Used Memory", Units.MEMORY_TOTAL, [MemoryColumns.USED_MEMORY])
            draw_dataframe(memory_df, self.__graphs_output_dir, self.__get_graph_name("Total Memory Consumption"),
                           x_info, y_info)
        else:
            print(f"The total memory usage file {self.__total_memory_each_moment_csv} does not exist.")

    def display_disk_io_graphs(self):
        if os.path.exists(self.__disk_io_each_moment_csv):
            disk_io_df = pd.read_csv(self.__disk_io_each_moment_csv, index_col=DiskIOColumns.TIME)

            # display number of io reads and writes
            x_info_count = AxisInfo("Time", Units.TIME, DiskIOColumns.TIME)
            y_info_count = AxisInfo("Number of Accesses", Units.COUNT,
                                    [DiskIOColumns.READ_COUNT, DiskIOColumns.WRITE_COUNT])
            draw_dataframe(disk_io_df, self.__graphs_output_dir, self.__get_graph_name("Count of Disk IO actions"),
                           x_info_count, y_info_count)

            # display number of bytes in io reads and writes
            x_info_bytes = AxisInfo("Time", Units.TIME, DiskIOColumns.TIME)
            y_info_bytes = AxisInfo("Number of Bytes", Units.IO_BYTES,
                                    [DiskIOColumns.READ_BYTES, DiskIOColumns.WRITE_BYTES])
            draw_dataframe(disk_io_df, self.__graphs_output_dir,
                           self.__get_graph_name("Number of bytes of Disk IO actions"), x_info_bytes, y_info_bytes)
        else:
            print(f"The total disk io usage file {self.__disk_io_each_moment_csv} does not exist.")

    def display_processes_graphs(self, processes_ids_to_emphasize: List[int]):
        if os.path.exists(self.__processes_results_csv):
            processes_df = pd.read_csv(self.__processes_results_csv)
            top_pids = (
                processes_df
                .groupby(ProcessesColumns.PROCESS_ID)[ProcessesColumns.CPU_CONSUMPTION]
                .sum()
                .nlargest(MAX_PROCESSES_TO_PLOT)
                .index
            )

            # Step 3: Filter the DataFrame
            filtered_processes_df = processes_df[processes_df[ProcessesColumns.PROCESS_ID].isin(top_pids)]

            self.__create_source_graph_per_processes_graph(processes_df=filtered_processes_df,
                                                           resource_type=ProcessesColumns.CPU_CONSUMPTION,
                                                           label_for_y="CPU consumption",
                                                           units_for_y=Units.PERCENT,
                                                           graph_name="CPU consumption per process")

            self.__create_source_graph_per_processes_graph(processes_df=filtered_processes_df,
                                                           resource_type=ProcessesColumns.MEMORY_PERCENT,
                                                           label_for_y="Memory consumption",
                                                           units_for_y=Units.MEMORY_PROCESS,
                                                           graph_name="Memory consumption per process")

            self.__create_source_graph_per_processes_graph(processes_df=filtered_processes_df,
                                                           resource_type=ProcessesColumns.READ_BYTES,
                                                           label_for_y="IO Read bytes",
                                                           units_for_y=Units.IO_BYTES,
                                                           graph_name="IO Read bytes per process")

            self.__create_source_graph_per_processes_graph(processes_df=filtered_processes_df,
                                                           resource_type=ProcessesColumns.WRITE_BYTES,
                                                           label_for_y="IO Write bytes",
                                                           units_for_y=Units.IO_BYTES,
                                                           graph_name="IO Write bytes per process")

            self.__create_source_graph_per_processes_graph(processes_df=filtered_processes_df,
                                                           resource_type=ProcessesColumns.READ_COUNT,
                                                           label_for_y="IO Read count",
                                                           units_for_y=Units.COUNT,
                                                           graph_name="IO Read count per process")

            self.__create_source_graph_per_processes_graph(processes_df=filtered_processes_df,
                                                           resource_type=ProcessesColumns.WRITE_COUNT,
                                                           label_for_y="IO Write count",
                                                           units_for_y=Units.COUNT,
                                                           graph_name="IO Write count per process")

            if processes_ids_to_emphasize is not None and len(processes_ids_to_emphasize) > 0:
                relevant_processes_df = processes_df.loc[processes_df[ProcessesColumns.PROCESS_ID].isin(processes_ids_to_emphasize)]
                self.__display_process_and_total_resource(processes_df=relevant_processes_df,
                                                          path_to_resource_df=self.__total_cpu_csv,
                                                          column_from_resource=CPUColumns.USED_PERCENT,
                                                          time_column_from_resource=CPUColumns.TIME,
                                                          column_of_resource_in_processes=ProcessesColumns.CPU_CONSUMPTION,
                                                          graph_name="CPU consumption",
                                                          label_for_y="CPU consumption",
                                                          units_for_y=Units.PERCENT,
                                                          processes_to_plot_id=processes_ids_to_emphasize)

                # TODO: CHANGE TOTAL DF TO BE TOTAL DF * KB IN THE MEMORY USAGE COLUM
                self.__display_process_and_total_resource(processes_df=relevant_processes_df,
                                                          path_to_resource_df=self.__total_memory_each_moment_csv,
                                                          column_from_resource=MemoryColumns.USED_MEMORY,
                                                          time_column_from_resource=MemoryColumns.TIME,
                                                          column_of_resource_in_processes=ProcessesColumns.USED_MEMORY,
                                                          graph_name="Memory consumption",
                                                          label_for_y="Memory consumption",
                                                          units_for_y=Units.MEMORY_PROCESS,
                                                          processes_to_plot_id=processes_ids_to_emphasize)

        else:
            print(f"The processes usage file {self.__processes_results_csv} does not exist.")




    def __display_process_and_total_resource(self, processes_df: pd.DataFrame, path_to_resource_df: str,
                                             column_from_resource: str, time_column_from_resource: str,
                                             column_of_resource_in_processes: str, graph_name: str,
                                             label_for_y: str, units_for_y: str,
                                             processes_to_plot_id: List[int]):

        total_df = pd.read_csv(path_to_resource_df)
        x_info = AxisInfo(label='Time', unit=Units.TIME, axis=ProcessesColumns.TIME)
        y_info = AxisInfo(label=label_for_y, unit=units_for_y, axis=[])  # axis list is not needed here

        if len(processes_df) > MINIMAL_REQUIRED_RECORDS:
            # TODO: patch for now to receive the same units.
            if column_from_resource is MemoryColumns.USED_MEMORY:
                total_df[MemoryColumns.USED_MEMORY] = total_df[MemoryColumns.USED_MEMORY] * KB

            draw_processes_and_total(
                processes_df=processes_df,
                total_df=total_df,
                total_time_column=time_column_from_resource,
                total_resource_column=column_from_resource,
                process_resource_column=column_of_resource_in_processes,
                process_ids_to_plot=processes_to_plot_id,
                x_info=x_info,
                y_info=y_info,
                title=self.__get_graph_name(graph_name, for_comparison=True),
                graphs_output_dir=self.__graphs_output_dir
            )

    def __create_source_graph_per_processes_graph(self, processes_df: pd.DataFrame, resource_type: str,
                                                  label_for_y: str, units_for_y: str, graph_name: str):
        resource_pivot = processes_df.pivot(index=ProcessesColumns.TIME, columns=ProcessesColumns.PROCESS_ID,
                                            values=resource_type)

        resource_pivot.columns = [f'PID {pid}' for pid in resource_pivot.columns]

        x_info = AxisInfo("Time", Units.TIME, ProcessesColumns.TIME)
        y_info = AxisInfo(axis=resource_pivot.columns, label=label_for_y, unit=units_for_y)

        draw_dataframe(resource_pivot, path_for_graphs=self.__graphs_output_dir,
                       graph_name=self.__get_graph_name(graph_name), x_info=x_info, y_info=y_info)
