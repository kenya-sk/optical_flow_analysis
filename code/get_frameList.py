#! /usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import sys

def get_frameList(timeFilePath):
    subList = []
    frameList = []
    with open(timeFilePath, "r") as f:
        time = f.readlines()
        for i in range(len(time)):
            if len(time[i]) >= 9:
                continue
            elif len(time[i]) >= 3:
                subList.append(int(time[i].strip("¥n").split("-")[0]))
                subList.append(int(time[i].strip("¥n").split("-")[1]))
            else:
                frameList.append(subList)
                subList = []
        frameList.append(subList)
    return frameList
