import sys
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_graph(stas_file_path, conv_axis_path, save_path, title, normalize=False):
    # read data of statistics value
    value_lst = list(pd.read_csv(stas_file_path, header=None)[0])

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(1,1,1)
    plt.title(title)

    # set value of x axis
    value_x = []
    for i in range(len(value_lst)):
        value_x.append(i)
    value_x_lim_min = value_x[0]
    value_x_lim_max = value_x[-1]

    if normalize:
        # normalize and set axis range to 0-1
        value_lst = np.array(value_lst)/max(value_lst)
        value_y_lim_min = 0
        value_y_lim_max = 1
    else:
        value_y_lim_min = 0
        value_y_lim_max = max(value_lst)*1.01

    plt.xlim(value_x_lim_min, value_x_lim_max)
    plt.ylim(value_y_lim_min, value_y_lim_max)
    plt.tick_params(labelsize=10)
    plt.xticks(rotation=90)
    plt.xlabel("Frame Number")
    plt.ylabel("Value [pixel]")

    # convert axis (frame number to something)
    # ex) frame number to one day time (30 fps video)
    if conv_axis_path is not None:
        conv_df = pd.read_csv(conv_axis_path)
        ax.set_xticks(conv_df["frame"])
        ax.set_xticklabels(conv_df["time"])

    plt.grid(True)
    plt.plot(value_x, value_lst)
    plt.savefig(save_path)
    plt.close(fig)
    print("SAVE: {0}".format(save_path))


if __name__ == "__main__":
    conv_axis_path = None

    plot_graph("../data/stats/mean.csv", conv_axis_path, "../data/graph/mean.eps", "mean graph", False)
    plot_graph("../data/stats/var.csv", conv_axis_path, "../data/graph/var.eps", "var graph", False)
    plot_graph("../data/stats/max.csv", conv_axis_path, "../data/graph/max.eps", "max graph", False)
