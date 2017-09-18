#! /usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


def mean_val_plot(meanList, valList, filePath, frameList, fps):
    '''
    plot mean and variance
    meanList: mean value list
    valList: variance value list
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
    for i in range(len(meanList)):
        meanX.append(i / fps)
    plt.subplot(2, 1, 1)
    plt.title('Mean of optical flow')
    plt.xlabel('time [s]')
    plt.ylabel('mean')
    plt.xlim(0.0, 300.0)
    plt.ylim(0.0, 2.0)
    plt.plot(meanX, meanList)
    fill_region(frameList)
    # variance list
    valX = []
    for i in range(len(valList)):
        valX.append(i / fps)
    plt.subplot(2, 1, 2)
    plt.title('Variance of optical flow')
    plt.xlabel('time [s]')
    plt.ylabel('variance')
    plt.xlim(0.0, 300.0)
    plt.ylim(0, 10)
    plt.plot(valX, valList)
    fill_region(frameList)
    fileName = filePath.split('/')[-1].split('.')[0] + '.png'
    plt.savefig('../image/20170429_2/' + fileName)
    print('\n'+fileName + ' graph success !\n')
