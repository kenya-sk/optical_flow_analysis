#! /usr/bin/env python
#coding: utf-8

import sys
import numpy as np
import os
import cv2
import re
import draw_opt
import get_frameList
import plot_kernelDensity
import plot_graph

def batch_processing():
    #--------------------------------------------------
    # load inputDate.txt
    #--------------------------------------------------
    try:
        with open('inputDate.txt','r') as f:
            date = f.readline().strip()
    except FileNotFoundError:
        print('Not Found: inputDate.txt')
        print('\tPlease soecify file directory in ./inputDate.txt')
        sys.exit(1)

    #get flush event time list
    #flushList = get_frameList.get_frameList("./flushSec.txt")

    #-------------------------------------------------
    #processing all data
    #------------------------------------------------
    #direc = '/Volumes/HDD-IO/Tuna_conv/'+date
    direc = "../movie/cut_version"
    file_lst = os.listdir(direc)
    pattern = r'^(?!._).*(.mp4)$'
    repattern = re.compile(pattern)
    norDensity_arr = np.zeros((720, 1280))
    total_mean, total_var, total_max = [], [], []
    for window in [30]:
        tmpMean_lst = [0 for i in range(window - 1)]
        tmpVar_lst = [0 for i in range(window - 1)]
        tmpMax_lst = [0 for i in range(window - 1)]
        for file_name in file_lst:
            if re.search(repattern,file_name):
                inputFile = direc + '/' + file_name
                #draw_opt.main(inputFile, window)

                #1h分繋げたグラフを作成
                mean_lst, var_lst, max_lst, tmpMean_lst, tmpVar_lst, tmpMax_lst = draw_opt.calc_flow(inputFile,tmpMean_lst, tmpVar_lst, tmpMax_lst, window, False)
                total_mean.extend(mean_lst)
                total_var.extend(var_lst)
                total_max.extend(max_lst)
        plot_graph.mean_var_plot(total_mean, total_var, window, "1h")
        plot_graph.max_plot(total_max, window, "1h")

if __name__ == '__main__':
     batch_processing()
