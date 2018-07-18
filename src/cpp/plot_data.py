#! /usr/bin/env python
# coding: utf-8

import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt


def plot_graph(direcPath, normalize=False):
    # get path of csv file from directory
    def get_filePath(direcPath):
        filePath_lst = os.listdir(direcPath)
        filePath_lst = list(filter(lambda filePath: ".csv" in filePath, filePath_lst))

        return filePath_lst

    # read data from csv file and store it in list
    def read_data(filePath):
        readValue_lst = []
        try:
            with open(filePath, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print("Not Found {0} ! ".format(filePath))
            sys.exit(1)

        for line in lines:
            readValue_lst.append(float(line.strip('\n')))

        return readValue_lst

    value_lst = []
    filePath_lst = get_filePath(direcPath)
    for filePath in filePath_lst:
        value_lst.extend(read_data(direcPath + filePath))

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(1,1,1)
    # title of graph ex) mean value (total)
    type = str(direcPath.split('/')[-2])
    plt.title(type + " value (total)")
    # set value of x axis
    valueX = []
    for i in range(len(value_lst)):
        valueX.append(i)
    valueXLimMin = valueX[0]
    valueXLimMax = valueX[-1]

    if normalize:
        # normalize and set axis range to 0-1
        value_lst = np.array(value_lst)/max(value_lst)
        valueYLimMin = 0
        valueYLimMax = 1
    else:
        valueYLimMin = 0
        valueYLimMax = max(value_lst)*1.01

    valueYLimMax = 1.0
    plt.xlim(valueXLimMin, valueXLimMax)
    plt.ylim(valueYLimMin, valueYLimMax)
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
    plt.plot(valueX, value_lst)
    plt.savefig("../image/graph/2017-04-28/out_" + type + "_total.png" )
    plt.close(fig)
    print("DONE: {}".format("/".join(direcPath.split('/')[-3:-1])))


if __name__ == "__main__":
    #plot_graph("../data/2017-04-28/mean/", False)
    #plot_graph("../data/2017-04-28/var/", False)
    #plot_graph("../data/2017-04-28/max/", False)
    plot_graph("../data/2017-04-28/human/", False)
