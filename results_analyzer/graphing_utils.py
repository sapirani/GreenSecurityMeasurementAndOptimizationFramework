import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from general_consts import KB
from results_analyzer.analyzer_constants import AxisInfo, DEFAULT


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


# def draw_grouped_dataframe(grouped_df, path_for_graphs: str, x_info: AxisInfo, y_info: AxisInfo, title: str):
#     cpu_pivot = df.pivot(index='time', columns='process id', values='cpu usage')
#
#     # Optional: rename columns to include 'PID' prefix
#     cpu_pivot.columns = [f'PID {pid}' for pid in cpu_pivot.columns]
#
#     # Define AxisInfo
#     x_info = AxisInfo(axis='time', label='Time')
#     y_info = AxisInfo(axis=cpu_pivot.columns.tolist(), label='CPU Usage (%)')
#
#     # Call the plotting function
#     draw_dataframe(cpu_pivot, path_for_graphs='.', graph_name='cpu_usage_by_process', x_info=x_info, y_info=y_info)


def draw_dataframe(df: pd.DataFrame, path_for_graphs: str, graph_name: str, x_info: AxisInfo, y_info: AxisInfo,
                   column_to_emphasis: str = None, do_subplots: bool = False):
    fig, ax = plt.subplots(figsize=(10, 5))

    if column_to_emphasis is not None:
        y_info.axis.remove(column_to_emphasis)
        df[column_to_emphasis].plot(ax=ax, legend=True, linewidth=5, subplots=do_subplots)

    df[y_info.axis].plot(ax=ax, legend=True)
    design_and_plot(x_info, y_info, graph_name, path_for_graphs)


def draw_process_and_total(total_df: pd.DataFrame, process_df: pd.DataFrame, x: AxisInfo, y: AxisInfo, title: str,
                           path_for_graphs: str, ):
    fig, ax = plt.subplots(figsize=(15, 6))
    total_df.plot(color='black', ax=ax).legend(labels=["Total Consumption"])
    process_df.plot(y=y.axis, ax=ax, label=True, color="r")
    design_and_plot(x, y, title, path_for_graphs)
