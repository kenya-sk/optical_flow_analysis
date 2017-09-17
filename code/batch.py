#! /usr/bin/env python
#coding: utf-8

import sys
import numpy as np
import os
import cv2
import re
import draw_opt
import get_frameList

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
    flushList = get_frameList.get_frameList(14, "./flushSec.txt")

    #-------------------------------------------------
    #processing all data
    #------------------------------------------------
    direc = '/Volumes/HDD-IO/Tuna_conv/'+date
    fileList = os.listdir(direc)
    pattern = r'^(?!._).*(.mp4)$'
    repattern = re.compile(pattern)
    count = 0
    for file_name in fileList:
        if re.search(repattern,file_name):
            inputFile = direc + '/' + file_name
            draw_opt.main(inputFile,flushList[count])
            count += 1

if __name__ == '__main__':
     batch_processing()
