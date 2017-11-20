#! /usr/bin/env python
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
    valueYLimMax = max(value_lst) + 10
    plt.xlim(valueXLimMin, valueXLimMax)
    plt.ylim(valueYLimMin, valueYLimMax)
    #x_lst = [x for x in range(0, len(valueX), 152)]
    #xTicks_lst = [x*10 for x in range(0, len(x_lst), 1)]
    #plt.xticks(x_lst ,xTicks_lst)
    plt.tick_params(labelsize=8)
    plt.grid(True)
    plt.plot(valueX, value_lst)

def mean_var_plot(mean_lst, var_lst, window, filePath):
    '''
    plot mean and variance
    mean_lst: mean value list
    val_lst: variance value list
    frameList: two dimention list
    '''

    # mean list
    plt.figure()
    plt.figure(figsize=(20, 6))
    plt.title('Mean of optical flow')
    plt.xlabel('frame number')
    plt.ylabel('mean')
    set_graph(mean_lst)
    fileName = filePath.split('/')[-1].split('.')[0] + '_mean_{}.png'.format(window)
    plt.savefig('../image/total_17_18/' + fileName)
    #fill_region(frameList)

    # variance list
    plt.figure()
    plt.figure(figsize=(20, 6))
    plt.title('Variance of optical flow')
    plt.xlabel('frame number')
    plt.ylabel('variance')
    set_graph(var_lst)
    #fill_region(frameList)
    fileName = filePath.split('/')[-1].split('.')[0] + '_var_{}.png'.format(window)
    plt.savefig('../image/total_17_18/' + fileName)
    print('\n'+fileName + ' graph success !\n')

def max_plot(max_lst, window, filePath):
    plt.figure()
    plt.figure(figsize=(20, 6))
    plt.title('max of optical flow')
    plt.xlabel('frame number')
    plt.ylabel('max')
    set_graph(max_lst)
    fileName = filePath.split('/')[-1].split('.')[0] + '_max_{}.png'.format(window)
    plt.savefig('../image/total_17_18/' + fileName)
    print('\n'+fileName + ' graph success !\n')
