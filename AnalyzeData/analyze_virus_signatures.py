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

SIGNATURE_NAME_COL = "signature"
FILE_TYPE_COL = "fileType"
VIRUS_NAME_STRING = "VIRUS NAME: "


class TargetString:
    first = "TARGET TYPE:"
    second = "Target:"


class TargetType:
    any = "ANY FILE"
    portable_executable = "PE"
    ole = "OLE2"
    html = "HTML"
    mail = "MAIL"
    graphics = "GRAPHICS"
    elf = "ELF"
    ascii = "NORMALIZED ASCII TEXT"
    unused = "UNUSED"
    macho = "MACHO"
    pdf = "PDF"
    flash = "FLASH"
    java = "JAVA CLASS"


def convert_file_type_to_str(type):
    if bool(re.search(r'\d', type)):
        match int(type):
            case 0:
                return TargetType.any
            case 1:
                return TargetType.portable_executable
            case 2:
                return TargetType.ole
            case 3:
                return TargetType.html
            case 4:
                return TargetType.mail
            case 5:
                return TargetType.graphics
            case 6:
                return TargetType.elf
            case 7:
                return TargetType.ascii
            case 8:
                return TargetType.unused
            case 9:
                return TargetType.macho
            case 10:
                return TargetType.pdf
            case 11:
                return TargetType.flash
            case 12:
                return TargetType.java
    else:
        return type


def preprocess_data():
    df = pd.DataFrame(columns=[FILE_TYPE_COL, SIGNATURE_NAME_COL])
    print("Start reading files")
    for f in os.listdir(SIGNATURES_DIR):
        path = join(SIGNATURES_DIR, f)
        if isfile(path):
            try:
                with open(path, "r") as file:
                    lines = file.readlines()
                    line_with_target = list(filter(lambda l: TargetString.first in l or TargetString.second in l, lines))
                    line_split = re.split(TargetString.first + "|" + TargetString.second, line_with_target[0].strip())
                    target_type = convert_file_type_to_str(line_split[1]).strip()
                    virus_type_name = re.split(VIRUS_NAME_STRING, lines[0].strip())[1]
                    df = df._append({FILE_TYPE_COL: target_type, SIGNATURE_NAME_COL: virus_type_name},
                               ignore_index=True)
            except:
                continue

    print("End of reading files")

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
    plt.rcParams["figure.figsize"] = (20, 20)  # change figure size

    # save graph as picture
    plt.savefig(os.path.join(path_to_save, graph_name))

    # function to show the plot
    plt.show()


def get_statistics_per_file_type(df):
    df_grouped_by_virus = df.groupby(FILE_TYPE_COL)[SIGNATURE_NAME_COL].count()
    df_grouped_by_virus.plot(kind='bar')
    x_info = AxisInfo("File type", "", FILE_TYPE_COL)
    y_info = AxisInfo("Number of signatures", Units.COUNT, df_grouped_by_virus.keys())
    design_and_plot(x_info, y_info, "Number of signatures per file type")


def main():
    # if there are already available statistics
    """if any(os.scandir(GRAPHS_DIR)):
        print("There are files in signature statistics directory. Replacing the files now.")
        [f.unlink() for f in Path(GRAPHS_DIR).glob("*") if f.is_file()]
"""

    df = preprocess_data()  # preprocess the date column
    print(df.columns)
    print(df)
    print("Getting statistics")
    get_statistics_per_file_type(df)


if __name__ == '__main__':
    main()
