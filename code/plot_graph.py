# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


def mean_var_plot(mean_lst, var_lst, filePath):
    '''
    plot mean and variance
    mean_lst: mean value list
    val_lst: variance value list
    frameList: two dimention list
    '''
    def fill_region(frameList):
        if len(frameList) != 0:
            for i in range(0, len(frameList), 2):
                plt.axvspan(frameList[i], frameList[i+1], facecolor = 'y', alpha = 0.5)
        else:
            pass

    plt.figure(figsize=(12, 9))
    # mean list
    meanX = []
    for i in range(len(mean_lst)):
        meanX.append(i)
    plt.subplot(2, 1, 1)
    plt.title('Mean of optical flow')
    plt.xlabel('time [s]')
    plt.ylabel('mean')
    meanXLimMin = 0.0
    meanXLimMax = 300.0
    meanYLimMin = min(mean_lst[16:]) - 5
    meanYLimMax = max(mean_lst[16:]) + 5
    plt.axis([0.0, 300.0, 0.0, 20])
    plt.grid(True)
    plt.plot(meanX, mean_lst)
    #fill_region(frameList)
    # variance list
    varX = []
    for i in range(len(var_lst)):
        varX.append(i)
    plt.subplot(2, 1, 2)
    plt.title('Variance of optical flow')
    plt.xlabel('time [s]')
    plt.ylabel('variance')
    varXLimMin = 0.0
    varXLimMax = 300.0
    varYLimMin = min(var_lst[16:]) - 5
    varYLimMax = max(var_lst[16:]) + 5
    plt.axis([0.0, 300.0, 0.0, 40.0])
    plt.grid(True)
    plt.plot(varX, var_lst)
    #fill_region(frameList)
    fileName = filePath.split('/')[-1].split('.')[0] + '_mean_var.png'
    plt.savefig('../image/20170429_17_18/' + fileName)
    print('\n'+fileName + ' graph success !\n')

def max_plot(max_lst, filePath):
    plt.figure(figsize=(12,9))
    maxX = []
    for i in range(len(max_lst)):
        maxX.append(i)
    plt.title('max of optical flow')
    plt.xlabel('time [s]')
    plt.ylabel('max')
    maxXLimMin = 0.0
    maxXLimMax = 300.0
    maxYLimMin = min(max_lst[16:]) - 5
    maxYLimMax = max(max_lst[16:]) + 5
    plt.axis([0.0, 300.0, 0.0, 80.0])
    plt.grid(True)
    plt.plot(maxX, max_lst)
    fileName = filePath.split('/')[-1].split('.')[0] + '_max.png'
    plt.savefig('../image/20170429_17_18/' + fileName)
    print('\n'+fileName + ' graph success !\n')
