import os
import re
from datetime import datetime
from os.path import isfile, join
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ConstantsForGraphs import *

GRAPHS_DIR = r"C:\Users\Administrator\Desktop\green security\documents\Signature Statistics\Graphs"
SIGNATURES_DIR = r"C:\Users\Administrator\Desktop\green security\documents\Signature Statistics\clamav-main\signatures"
SIGNATURE_DATASET_UPDATE_PATH = r"C:\Users\Administrator\Desktop\green security\documents\Signature Statistics\changed signatures.csv"
SIGNATURE_AND_TYPE_DATASET_PATH = r"Signatures_dataset.csv"
SIGNATURE_NAME_COL = "signature"
SIGNATURE_NAME_AFTER_MERGE_COL = "SubName"
DATABASE_COL = "database"
DATE_COL = "buildtime"
VERSION_COL = "version"
FILE_TYPE_COL = "fileType"


def preprocess_data():
    sig_df = pd.read_csv(SIGNATURE_DATASET_UPDATE_PATH)
    print(sig_df[DATE_COL])
    sig_df[DATE_COL] = pd.to_datetime(sig_df[DATE_COL], format="%d/%m/%Y %H:%M")
    #sig_df = sig_df.loc[(sig_df[DATE_COL] >= '2023-07-01')]
    #sig_df[sig_df[DATE_COL].contains("JUL")]

    sig_per_file_type_df = pd.read_csv(SIGNATURE_AND_TYPE_DATASET_PATH)

    pat_sig_name = f"({'|'.join(sig_per_file_type_df[SIGNATURE_NAME_COL])})"
    sig_df[SIGNATURE_NAME_AFTER_MERGE_COL] = sig_df[SIGNATURE_NAME_COL].str.extract(pat=pat_sig_name)
    combined_df = sig_df.merge(sig_per_file_type_df, left_on=[SIGNATURE_NAME_AFTER_MERGE_COL], right_on=[SIGNATURE_NAME_COL], how='left')

    """ for index1, sig_update in sig_df.iterrows():
        for index2, sig_details in sig_per_file_type_df.iterrows():
            if sig_details[SIGNATURE_NAME_COL] in sig_update[SIGNATURE_NAME_COL]:
                combined_df = combined_df._append({SIGNATURE_NAME_COL: sig_update[SIGNATURE_NAME_COL],
                                      DATE_COL: sig_update[DATE_COL],
                                      FILE_TYPE_COL: sig_per_file_type_df[FILE_TYPE_COL]},
                                    ignore_index=True)"""

    return combined_df[combined_df['SubName'].notna()]

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
    plt.rc('legend', fontsize=12)  # Set the legend font size
    plt.xticks(fontsize=8, color='black')  # change x ticks color
    plt.yticks(fontsize=12, color='black')  # change y ticks color
    plt.rcParams["figure.figsize"] = (10,10)  # change figure size

    # save graph as picture
    plt.savefig(os.path.join(path_to_save, graph_name))

    # function to show the plot
    plt.show()


def get_statistics_per_type_and_update(df):

    # for each file type count how many signatures where added in JULY.
    df_grouped_by_type = df.groupby(FILE_TYPE_COL)[SIGNATURE_NAME_AFTER_MERGE_COL].count()
    df_grouped_by_type.plot(kind='bar')
    x_info = AxisInfo("File type", "", FILE_TYPE_COL)
    y_info = AxisInfo("Number of signatures", Units.COUNT, df_grouped_by_type.keys())
    design_and_plot(x_info, y_info, "Number of signatures per file type in last month")

    # for each day, count how much signatures added for each file type
    # x - date, y - number of sigs, color - file type
    df_grouped_by_type_and_date = df.groupby([DATE_COL, FILE_TYPE_COL])[SIGNATURE_NAME_AFTER_MERGE_COL].count()
    df_grouped_by_type_and_date.unstack(FILE_TYPE_COL).plot(kind='bar', stacked=True)
    x_info = AxisInfo("Date", "days", DATE_COL)
    y_info = AxisInfo("Number of signatures", Units.COUNT, df_grouped_by_type_and_date.keys())
    design_and_plot(x_info, y_info, "Number of signatures per day and file type")

def get_statistics_per_year(df):
    df_grouped_by_type_and_date = df.groupby([DATE_COL, FILE_TYPE_COL])[SIGNATURE_NAME_AFTER_MERGE_COL].count()
    df_grouped_by_type_and_date.unstack(FILE_TYPE_COL).plot(kind='bar', stacked=True)
    x_info = AxisInfo("Date", "days", DATE_COL)
    y_info = AxisInfo("Number of signatures", Units.COUNT, df_grouped_by_type_and_date.keys())
    design_and_plot(x_info, y_info, "Number of signatures per file type in last year, which are not for any file")
    print(df_grouped_by_type_and_date.keys())
    with open(r'C:\Users\Administrator\Desktop\green security\documents\Signature Statistics\dates_without_generic.txt', 'w') as fp:
        fp.write("\n".join(str(item) for item in df_grouped_by_type_and_date.keys()))


def remove_dates_with_generic_sigs(df):
    df = df.loc[(df[DATE_COL] >= '2023-01-01')]
    df_any_file_sigs = df.loc[(df[FILE_TYPE_COL] == "ANY FILE")]
    df = df.loc[(~df[DATE_COL].isin(df_any_file_sigs[DATE_COL]))]
    return df


def main():
    # if there are already available statistics
    """if any(os.scandir(GRAPHS_DIR)):
        print("There are files in signature statistics directory. Replacing the files now.")
        [f.unlink() for f in Path(GRAPHS_DIR).glob("*") if f.is_file()]
"""

    if os.path.exists(r"C:\Users\Administrator\Desktop\green security\code\AnalyzeData\Signatures_dataset_after_merge_with_all_year.csv"):
        df = pd.read_csv("Signatures_dataset_after_merge_with_all_year.csv")
    else:
        df = preprocess_data()  # preprocess the date column
        df.to_csv("Signatures_dataset_after_merge_with_all_year.csv")

    print(df.columns)
    print(df)
    print("Getting statistics")

    df = remove_dates_with_generic_sigs(df)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.date
    #get_statistics_per_type_and_update(df)
    get_statistics_per_year(df)

if __name__ == '__main__':
    main()
