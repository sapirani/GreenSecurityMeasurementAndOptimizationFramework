import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from results_analyzer.analyzer_constants import AxisInfo


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

    # one graph with all plots
    emphasis_column = y_info.axis[0] if y_info.axis else None
    graph_name = f"{title} - All Cores"
    draw_dataframe(df, path_for_graphs, graph_name, x_info, y_info, column_to_emphasis=emphasis_column)

    # plt.figure(figsize=(12, 6))
    # for col in y_info.axis:
    #     plt.plot(df[x_info.axis[0]], df[col], label=col)
    #
    # design_and_plot(x_info, y_info, title, path_for_graphs)
    # plt.xlabel(x_info.label)
    # plt.ylabel('Usage (%)')
    # plt.title(title)
    # plt.legend(loc='upper right')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # number_of_cols_to_plot = len(y_info.axis)
    # cols = round(math.sqrt(number_of_cols_to_plot))
    # rows = cols
    #
    # fig, ax = plt.subplots(rows, cols, figsize=(18, 10))
    # for i in range(rows):
    #     for j in range(cols):
    #         if i * cols + j >= number_of_cols_to_plot:
    #             break
    #         ax[i][j].plot(df[y_info.axis[i + j]])
    #         ax[i][j].set_title(y_info.axis[i * cols + j])
    #
    # fig.suptitle(title, color="darkblue", fontsize=30, fontname="Times New Roman", fontweight="bold")
    # fig.supxlabel(x_info.label, fontsize=20, color='crimson')
    # fig.supylabel(y_info.label, fontsize=20, color='crimson')
    # fig.tight_layout(pad=3.0)

    # save graph as picture
    # plt.savefig(os.path.join(path_for_graphs, title))
    #
    # # function to show the plot
    # plt.show()

def draw_dataframe(df, path_for_graphs, graph_name, x_info, y_info, column_to_emphasis=None, do_subplots=False):
    fig, ax = plt.subplots(figsize=(10, 5))

    if column_to_emphasis is not None:
        y_info.axis.remove(column_to_emphasis)
        df[column_to_emphasis].plot(ax=ax, legend=True, linewidth=5, subplots=do_subplots)
    df[y_info.axis].plot(ax=ax, legend=True)
    design_and_plot(x_info, y_info, graph_name, path_for_graphs)

