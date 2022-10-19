import psutil
import time
import numpy as np
from subprocess import call

import wmi
from prettytable import PrettyTable
from datetime import datetime
import matplotlib.pyplot as plt
import subprocess
from multiprocessing import Process
from threading import Thread
import time
import pandas as pd

DISK_FILE = "disk_io_each_moment.csv"
MEMORY_FILE = "total_memory.csv"
PROCESSES_FILE = "processes_data.csv"
GB = 2 ** 30
MB = 2 ** 20
KB = 2 ** 10
ONE_GRAPH = 1
COMBINED_GRAPH = 2


def read_file_to_dataframe(path, columns_names):
    df = pd.read_csv(path)
    df_new = df[columns_names]
    # df_new.plot()
    return df_new


def draw_graph(df, x_col, y_cols,  x_name, y_name, graph_name):
    # plotting the points

    df.plot(x=x_col, y=y_cols)

    # naming the x axis
    plt.xlabel(x_name)
    # naming the y axis
    plt.ylabel(y_name)

    # giving a title to my graph
    plt.title(graph_name)

    # function to show the plot
    plt.show()


def read_processes_file():
    processes_df = read_file_to_dataframe(PROCESSES_FILE, ["Time(sec)", "PNAME", "CPU(%)"])

    return processes_df


def main():
    # processes table
    processes_df = read_file_to_dataframe(PROCESSES_FILE, ["Time(sec)", "PNAME", "CPU(%)"])

    draw_graph(processes_df, "Time(sec)", "CPU(%)", "Time (sec)", "Total CPU use", "CPU consumption of all processes "
                                                                                   "in the system")

    memory_df = read_file_to_dataframe(MEMORY_FILE, ['Time(sec)', 'Used(GB)'])
    draw_graph(memory_df, "Time(sec)", "Used(GB)", "Time (sec)", "Total Memory Use", "Memory consumption of the system")


if __name__ == '__main__':
    main()

