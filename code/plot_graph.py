# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def fill_region(frameList):
    if len(frameList) != 0:
        for i in range(0, len(frameList), 2):
            plt.axvspan(frameList[i], frameList[i+1], facecolor = 'y', alpha = 0.5)
    else:
        pass

def set_graph(value_lst):
    valueX = []
    for i in range(len(value_lst)):
        valueX.append(i)
    valueXLimMin = valueX[0]
    valueXLimMax = valueX[-1]
    valueYLimMin = 0
    valueYLimMax = max(value_lst[:len(value_lst)-100]) + 5
    plt.xlim(valueXLimMin, valueXLimMax)
    plt.ylim(valueYLimMin, valueYLimMax)
    x_lst = [x for x in range(0, len(valueX), 152)]
    xTicks_lst = [x*10 for x in range(0, len(x_lst), 1)]
    plt.xticks(x_lst ,xTicks_lst)
    plt.tick_params(labelsize=8)
    plt.grid(True)
    plt.plot(valueX, value_lst)

def mean_var_plot(mean_lst, var_lst, filePath):
    '''
    plot mean and variance
    mean_lst: mean value list
    val_lst: variance value list
    frameList: two dimention list
    '''

    plt.figure(figsize=(12, 9))
    # mean list
    plt.subplot(2, 1, 1)
    plt.title('Mean of optical flow')
    plt.xlabel('time [s]')
    plt.ylabel('mean')
    set_graph(mean_lst)
    #fill_region(frameList)
    # variance list
    plt.subplot(2, 1, 2)
    plt.title('Variance of optical flow')
    plt.xlabel('time [s]')
    plt.ylabel('variance')
    set_graph(var_lst)
    #fill_region(frameList)
    fileName = filePath.split('/')[-1].split('.')[0] + '_mean_var.png'
    #plt.savefig('../image/test/' + fileName)
    plt.savefig("./" + fileName)
    print('\n'+fileName + ' graph success !\n')

def max_plot(max_lst, filePath):
    plt.figure(figsize=(12,9))
    plt.title('max of optical flow')
    plt.xlabel('time [s]')
    plt.ylabel('max')
    set_graph(max_lst)
    fileName = filePath.split('/')[-1].split('.')[0] + '_max.png'
    #plt.savefig('../image/test/' + fileName)
    plt.savefig("./" + fileName)
    print('\n'+fileName + ' graph success !\n')
