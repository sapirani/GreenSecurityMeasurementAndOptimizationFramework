import os

import pandas as pd

from general_consts import BatteryColumns, CPUColumns
from initialization_helper import result_paths
from results_analyzer.analyzer_constants import AxisInfo, Units
from results_analyzer.graphing_utils import draw_dataframe, draw_subplots

NUM_OF_PATHS = 12

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

    def __get_graph_name(self, title: str) -> str:
        return f"{title} Graph for {self.__container_name}"

    def display_battery_graphs(self):
        if os.path.exists(self.__battery_status_csv):
            battery_df = pd.read_csv(self.__battery_status_csv, index_col=BatteryColumns.TIME)

            # display capacity drain
            x_info_capacity = AxisInfo("Time", Units.TIME, BatteryColumns.TIME)
            y_info_capacity = AxisInfo("Remaining Capacity", Units.CAPACITY, [BatteryColumns.CAPACITY])
            draw_dataframe(battery_df, self.__graphs_output_dir, self.__get_graph_name("Battery drop (mWh)"), x_info_capacity, y_info_capacity)

            # display voltage drain
            x_info_voltage = AxisInfo("Time", Units.TIME, BatteryColumns.TIME)
            y_info_voltage = AxisInfo("Voltage", Units.VOLTAGE, [BatteryColumns.VOLTAGE])
            draw_dataframe(battery_df, self.__graphs_output_dir, self.__get_graph_name("Battery drop (mV)"), x_info_voltage, y_info_voltage)
        else:
            print(f"The battery status file {self.__battery_status_csv} does not exist.")

    def display_cpu_graphs(self):
        if os.path.exists(self.__total_cpu_csv):
            cpu_df = pd.read_csv(self.__total_cpu_csv) #, index_col=CPUColumns.TIME)
            x_info = AxisInfo("Time", Units.TIME, [CPUColumns.TIME])
            y_info = AxisInfo("Used CPU", Units.PERCENT, cpu_df.columns.tolist()[1:])
            draw_subplots(cpu_df, self.__graphs_output_dir, x_info, y_info, self.__get_graph_name("Total CPU Consumption per core"))
            # draw_dataframe(cpu_df, self.__graphs_output_dir, self.__get_graph_name("Total CPU Consumption"), x_info, y_info, column_to_emphasis=CPUColumns.USED_PERCENT)
        else:
            print(f"The total CPU usage file {self.__total_cpu_csv} does not exist.")
