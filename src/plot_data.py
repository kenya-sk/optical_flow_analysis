#! /usr/bin/env python
# coding: utf-8

import sys
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt


def plot_graph(direc_path, normalize=False):
    # read data from csv file and store it in list
    def read_data(file_path):
        value_lst = []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print("Not Found {0} ! ".format(file_path))
            sys.exit(1)

        for line in lines:
            value_lst.append(float(line.strip('\n')))

        return value_lst

    value_lst = []
    file_path_lst = glob.glob(direc_path+"*.csv")
    for path in file_path_lst:
        value_lst.extend(read_data(path))

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(1,1,1)
    # title of graph ex) mean value (total)
    statistic_type = str(direc_path.split('/')[-2])
    plt.title(statistic_type + " value (total)")
    # set value of x axis
    value_x = []
    for i in range(len(value_lst)):
        value_x.append(i)
    value_x_lim_min = value_x[0]
    value_x_lim_max = value_x[-1]
    print(len(value_x))

    if normalize:
        # normalize and set axis range to 0-1
        value_lst = np.array(value_lst)/max(value_lst)
        value_y_lim_min = 0
        value_y_lim_max = 1
    else:
        value_y_lim_min = 0
        value_y_lim_max = max(value_lst)*1.01

    value_y_lim_max = 150
    plt.xlim(value_x_lim_min, value_x_lim_max)
    plt.ylim(value_y_lim_min, value_y_lim_max)
    plt.tick_params(labelsize=8)
    plt.xticks(rotation=90)

    # one day data
    ax.set_xticks([0, 8990, 17980, 26970, 35960, 44950,
                    53939, 62929, 71919, 80909, 89899, 98889,
                    107878, 116868, 125858, 134848, 143838, 152828,
                    161817, 170807, 179797, 188787, 197777, 206767,
                    215756, 224746, 233736, 242726, 251716, 260706,
                    269696, 278686, 287676, 296666, 305656, 314646,
                    323636, 332626, 341616, 350606, 359596, 368586,
                    377576, 386566, 395556, 404546, 413536, 422526
                    ])
    ax.set_xticklabels(["09:00", "09:10", "09:20", "09:30", "09:40", "09:50",
                        "10:00", "10:10", "10:20", "10:30", "10:40", "10:50",
                        "11:00", "11:10", "11:20", "11:30", "11:40", "11:50",
                        "12:00", "12:10", "12:20", "12:30", "12:40", "12:50",
                        "13:00", "13:10", "13:20", "13:30", "13:40", "13:50",
                        "14:00", "14:10", "14:20", "14:30", "14:40", "14:50",
                        "15:00", "15:10", "15:20", "15:30", "15:40", "15:50",
                        "16:00", "16:10", "16:20", "16:30", "16:40", "16:50"
                        ])

    plt.grid(True)
    plt.plot(value_x, value_lst)
    plt.savefig("/Users/sakka/optical_flow_analysis/image/graph/2017-04-21/out_" + statistic_type + "_total.png" )
    plt.close(fig)
    print("DONE: {}".format("/".join(direc_path.split('/')[-3:-1])))


if __name__ == "__main__":
    plot_graph("/Users/sakka/optical_flow_analysis/data/2017-04-21/mean/", False)
    plot_graph("/Users/sakka/optical_flow_analysis/data/2017-04-21/var/", False)
    plot_graph("/Users/sakka/optical_flow_analysis/data/2017-04-21/max/", False)
    plot_graph("/Users/sakka/optical_flow_analysis/data/2017-04-21/human/", False)
