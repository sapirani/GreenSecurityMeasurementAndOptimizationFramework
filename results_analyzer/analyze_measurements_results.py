import os
from typing import List

import pandas as pd
from numpy.core.defchararray import startswith

from general_consts import BatteryColumns
from results_analyzer.analyzer_constants import AxisInfo, Units
from results_analyzer.analyzer_utils import is_results_dir, draw_dataframe

results_main_dir = fr"C:\Users\Administrator\Desktop\GreenSecurityAll\results"
graphs_output_dir_name = fr"graphs"


def read_all_results_dirs(main_result_dir: str) -> List[str]:
    if os.path.exists(main_result_dir):
        return [os.path.join(main_result_dir, result_dir) for result_dir in os.listdir(main_result_dir) if is_results_dir(result_dir)]

    else:
        return []


def print_container_name(container_results_dir: str) -> str:
    container_name = os.path.basename(container_results_dir)[len("results_"):]
    print(f"****** Printing {container_name} Results: ******\n\n")
    return container_name


def display_battery_graphs(container_results_dir: str, graphs_output_dir: str, container_name: str) -> None:
    battery_results_path = os.path.join(container_results_dir, "battery.csv")
    if os.path.exists(battery_results_path):
        battery_df = pd.read_csv(battery_results_path, index_col=BatteryColumns.TIME)

        # display capacity drain
        x_info_capacity = AxisInfo("Time", Units.TIME, BatteryColumns.TIME)
        y_info_capacity = AxisInfo("Remaining Capacity", Units.CAPACITY, [BatteryColumns.CAPACITY])
        draw_dataframe(battery_df, graphs_output_dir, "Battery drop (mWh)", x_info_capacity, y_info_capacity)

        # display voltage drain
        x_info_voltage = AxisInfo("Time", Units.TIME, BatteryColumns.TIME)
        y_info_voltage = AxisInfo("Voltage", Units.VOLTAGE, [BatteryColumns.VOLTAGE])
        draw_dataframe(battery_df, graphs_output_dir, "Battery drop (mV)", x_info_voltage, y_info_voltage)


def print_results_graphs_per_container(container_results_dir: str):
    graphs_output_dir = os.path.join(container_results_dir, "graphs")
    container_name = print_container_name(container_results_dir)
    display_battery_graphs(container_results_dir, graphs_output_dir, container_name)


def print_results_graphs(results_dir: str):
    if os.path.exists(results_dir):
        for container_results_dir in os.listdir(results_dir):
            print_results_graphs_per_container(container_results_dir)


def main():
    results_dirs = read_all_results_dirs(results_main_dir)
    for results_dir in results_dirs:
        print_results_graphs(results_dir)


if __name__ == "__main__":
    main()
