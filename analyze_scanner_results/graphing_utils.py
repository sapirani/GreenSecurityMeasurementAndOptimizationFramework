import os
from typing import List

import pandas as pd
from matplotlib import pyplot as plt

from utils.general_consts import ProcessesColumns
from analyze_scanner_results.analyzer_constants import AxisInfo


def design_and_plot(x_info, y_info, graph_name, path_for_graphs):
    # naming the x-axis
    plt.xlabel(x_info.label + " (in " + x_info.unit + ")", color='crimson', labelpad=10, fontname="Comic Sans MS")

    # naming the y-axis
    plt.ylabel(y_info.label + " (in " + y_info.unit + ")", color='crimson', labelpad=10,
               fontname="Comic Sans MS")  # naming the y-axis

    # giving a title to the graph, changing its font, size and color
    plt.suptitle(graph_name, color="darkblue", fontsize=20, fontname="Times New Roman", fontweight="bold")

    # design graph
    plt.rc('axes', labelsize=12)  # Set the axes labels font size
    plt.rc('legend', fontsize=6)  # Set the legend font size
    plt.xticks(fontsize=8, color='darkgray')  # change x ticks color
    plt.yticks(fontsize=8, color='darkgray')  # change y ticks color
    plt.rcParams["figure.figsize"] = (10, 5)  # change figure size

    # save graph as picture
    plt.savefig(os.path.join(path_for_graphs, graph_name))

    # function to show the plot
    plt.show()


def draw_subplots(df: pd.DataFrame, path_for_graphs: str, x_info: AxisInfo, y_info: AxisInfo, title: str):
    # separate graph for each column
    for column in y_info.axis:
        single_y_info = AxisInfo(
            label=y_info.label,
            unit=y_info.unit,
            axis=[column]
        )
        graph_name = f"{title} - {column.replace(' ', '_')}"
        draw_dataframe(df, path_for_graphs, graph_name, x_info, single_y_info)


def draw_dataframe(df: pd.DataFrame, path_for_graphs: str, graph_name: str, x_info: AxisInfo, y_info: AxisInfo,
                   column_to_emphasis: str = None, do_subplots: bool = False):
    fig, ax = plt.subplots(figsize=(10, 5))

    if column_to_emphasis is not None:
        y_info.axis.remove(column_to_emphasis)
        df[column_to_emphasis].plot(ax=ax, legend=True, linewidth=5, subplots=do_subplots)

    df[y_info.axis].plot(ax=ax, legend=True)
    design_and_plot(x_info, y_info, graph_name, path_for_graphs)


def draw_processes_and_total(processes_df: pd.DataFrame, total_df: pd.DataFrame,
                             total_time_column: str, total_resource_column: str,
                             process_resource_column: str, process_ids_to_plot: List[int],
                             x_info: AxisInfo, y_info: AxisInfo, title: str, graphs_output_dir: str):

    plt.figure(figsize=(12, 6))

    has_lines = False  # Track if anything was plotted

    # Plot each process
    for pid in process_ids_to_plot:
        pid_data = processes_df[processes_df[ProcessesColumns.PROCESS_ID] == pid]
        if not pid_data.empty:
            plt.plot(
                pid_data[ProcessesColumns.TIME],
                pid_data[process_resource_column],
                label=f'PID {pid}'
            )
            has_lines = True

    # Plot total system resource
    if not total_df.empty:
        plt.plot(
            total_df[total_time_column],
            total_df[total_resource_column],
            label='Total',
            color='black',
            linewidth=2,
            linestyle='--'
        )
        has_lines = True

    if has_lines:
        print("HAS LINES")
        plt.legend(loc='best', fontsize='medium')


    design_and_plot(x_info, y_info, title, graphs_output_dir)
