import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ConstantsForGraphs import *

PATH = r"C:\Users\Administrator\Desktop\green security\documents\Signature Statistics\changed signatures.csv"
GRAPHS_DIR = r"C:\Users\Administrator\Desktop\green security\documents\Signature Statistics\Graphs"

SIGNATURE_NAME_COL = "signature"
DATABASE_COL = "database"
DATE_COL = "buildtime"
VERSION_COL = "version"
VIRUS_TYPE_COL = "virusType"
OPERATION_SYSTEM_COL = "system"

VIRUS_TYPE_INDEX = 1
OPERATION_SYSTEM_INDEX = 0

def preprocess_data(df):
    df = df[df[DATE_COL].str.contains("Jul")]
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format='mixed', dayfirst=True)
    df[DATE_COL] = df[DATE_COL].dt.date
    return df


def design_and_plot(x_info, y_info, graph_name, path_to_save=GRAPHS_DIR):
    today = datetime.now().strftime("%d-%m-%Y")
    graph_name = graph_name + " - " + today

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
    plt.xticks(fontsize=8, color='black')  # change x ticks color
    plt.yticks(fontsize=8, color='darkgray')  # change y ticks color
    plt.rcParams["figure.figsize"] = (10, 10)  # change figure size

    # save graph as picture
    plt.savefig(os.path.join(path_to_save, graph_name))

    # function to show the plot
    plt.show()


def get_statistics_per_date(df):
    # graph of number of signatures per date
    df_grouped_by_date = df.groupby(DATE_COL)[SIGNATURE_NAME_COL].count()
    df_grouped_by_date.plot(kind='bar')
    x_info = AxisInfo("Dates", "", DATE_COL)
    y_info = AxisInfo("Number of signatures", Units.COUNT, df_grouped_by_date.keys())
    design_and_plot(x_info, y_info, "Number of signatures per day")

    # graph of number of signatures per date and version
    df_grouped_by_date_version = df.groupby([DATE_COL, VERSION_COL])[SIGNATURE_NAME_COL].count()
    df_grouped_by_date_version.plot(kind='bar')
    x_info = AxisInfo("Dates and Versions", "", DATE_COL)
    y_info = AxisInfo("Number of signatures", Units.COUNT, df_grouped_by_date_version.keys())
    design_and_plot(x_info, y_info, "Number of signatures per day and version")

    # graph of number of signatures per date and database type
    df_grouped_by_date_database = df.groupby([DATE_COL, DATABASE_COL])[SIGNATURE_NAME_COL].count()
    df_grouped_by_date_database.plot(kind='bar')
    x_info = AxisInfo("Dates and Database", "", DATE_COL)
    y_info = AxisInfo("Number of signatures", Units.COUNT, df_grouped_by_date_database.keys())
    design_and_plot(x_info, y_info, "Number of signatures per day and database type")

def get_statistics_per_virus_type(df):
    # convert signature name to 2 new cols
    df[VIRUS_TYPE_COL] = df[SIGNATURE_NAME_COL].apply(lambda name: name.split('.')[VIRUS_TYPE_INDEX])
    df[OPERATION_SYSTEM_COL] = df[SIGNATURE_NAME_COL].apply(lambda name: name.split('.')[OPERATION_SYSTEM_INDEX])

    # graph of number of signatures per virus type
    df_grouped_by_virus = df.groupby(VIRUS_TYPE_COL)[SIGNATURE_NAME_COL].count()
    df_grouped_by_virus.plot(kind='bar')
    x_info = AxisInfo("Virus type", "", VIRUS_TYPE_COL)
    y_info = AxisInfo("Number of signatures", Units.COUNT, df_grouped_by_virus.keys())
    design_and_plot(x_info, y_info, "Number of signatures per virus type")

    # graph of number of signatures per virus operating system
    df_grouped_by_system = df.groupby(OPERATION_SYSTEM_COL)[SIGNATURE_NAME_COL].count()
    df_grouped_by_system.plot(kind='bar')
    x_info = AxisInfo("Operation system", "", OPERATION_SYSTEM_COL)
    y_info = AxisInfo("Number of signatures", Units.COUNT, df_grouped_by_system.keys())
    design_and_plot(x_info, y_info, "Number of signatures per system")

    # graph of number of signatures per virus type and operating system
    df_grouped_by_system = df.groupby([OPERATION_SYSTEM_COL, VIRUS_TYPE_COL])[SIGNATURE_NAME_COL].count()
    df_grouped_by_system.plot(kind='bar')
    x_info = AxisInfo("Operation system and virus type", "", OPERATION_SYSTEM_COL)
    y_info = AxisInfo("Number of signatures", Units.COUNT, df_grouped_by_system.keys())
    design_and_plot(x_info, y_info, "Number of signatures per system and virus type")


def main():

    # if there are already available statistics
    """if any(os.scandir(GRAPHS_DIR)):
        print("There are files in signature statistics directory. Replacing the files now.")
        [f.unlink() for f in Path(GRAPHS_DIR).glob("*") if f.is_file()]
"""

    df = pd.read_csv(PATH)  # read signatures csv
    print(df.columns)
    df = preprocess_data(df)  # preprocess the date column

    get_statistics_per_date(df)
    get_statistics_per_virus_type(df)


if __name__ == '__main__':
    main()
