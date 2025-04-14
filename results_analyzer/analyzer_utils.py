import os

from matplotlib import pyplot as plt


def is_results_dir(dir_name: str) -> bool:
    return dir_name is not None and dir_name.startswith("results")

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

def draw_dataframe(df, path_for_graphs, graph_name, x_info, y_info, column_to_emphasis=None, do_subplots=False):
    fig, ax = plt.subplots(figsize=(10, 5))

    if column_to_emphasis is not None:
        y_info.axis.remove(column_to_emphasis)
        df[column_to_emphasis].plot(ax=ax, legend=True, linewidth=5, subplots=do_subplots)

    df[y_info.axis].plot(ax=ax, legend=len(y_info.axis) > 1)
    design_and_plot(x_info, y_info, graph_name, path_for_graphs)
