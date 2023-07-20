import os
import re
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ConstantsForGraphs import *
from AnalyzeData.analyze_virus_signatures import design_and_plot

OUTPUT_FILE_PATH = r"C:\Users\Administrator\Desktop\green security\documents\Files Statistics\Summaries"
INPUT_FILE_PATH = r"C:\Users\Administrator\Desktop\green security\documents\Files Statistics\total_files_per_type.txt"
GRAPHS_DIR = r"C:\Users\Administrator\Desktop\green security\documents\Files Statistics\Graphs"

FILE_TYPE_COL = "FileType"
NUMBER_OF_FILES_COL = "FilesAmount"
FILES_RATIO_COL = "FilesAmountPrecent"
MAX_ROWS_TO_PLOT = 20

def read_file_to_df():
    df = pd.DataFrame(columns=[FILE_TYPE_COL, NUMBER_OF_FILES_COL])
    with open(INPUT_FILE_PATH, "r") as f:
        f.readline()
        f.readline()
        f.readline()
        for line in f.readlines():
            line = line.strip()
            name_and_count = [word for word in line.split(" ") if word is not ""]

            if len(name_and_count) == 2: # what to do with partial row?
                df = df._append({FILE_TYPE_COL: name_and_count[0], NUMBER_OF_FILES_COL: int(name_and_count[1])}, ignore_index=True)


    print(df)
    return df


def plot_graph(df, x_info, y_info, graph_title):
    df.plot.bar(x=x_info.axis, y=y_info.axis)
    design_and_plot(x_info, y_info, graph_title, GRAPHS_DIR)

def get_statistics_per_file(df):
    # get sum of files
    total_number_of_file_on_computer = df[NUMBER_OF_FILES_COL].sum()
    print("Number of total files on the computer: ")
    print(total_number_of_file_on_computer)
    # insert ratio column
    df[FILES_RATIO_COL] = df[NUMBER_OF_FILES_COL].apply(lambda x: int(x)/int(total_number_of_file_on_computer))

    # graph of files amount
    x_info = AxisInfo("File Type", "", FILE_TYPE_COL)
    y_info = AxisInfo("Number of files", Units.COUNT, NUMBER_OF_FILES_COL)
    plot_graph(df[[FILE_TYPE_COL, NUMBER_OF_FILES_COL]].head(MAX_ROWS_TO_PLOT), x_info, y_info, "Number of files per type")

    # graph of files ratio
    x_info = AxisInfo("File Type", "", FILE_TYPE_COL)
    y_info = AxisInfo("Number of type files out of total files ", Units.PERCENT, FILES_RATIO_COL)
    plot_graph(df[[FILE_TYPE_COL, FILES_RATIO_COL]].head(MAX_ROWS_TO_PLOT), x_info, y_info, "Number of files per type out of all existing files")


def main():

    # if there are already available statistics
    """if any(os.scandir(GRAPHS_DIR)):
        print("There are files in signature statistics directory. Replacing the files now.")
        [f.unlink() for f in Path(GRAPHS_DIR).glob("*") if f.is_file()]"""


    df = read_file_to_df()
    get_statistics_per_file(df)


if __name__ == '__main__':
    main()