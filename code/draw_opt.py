#! /usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import sys
import argparse

import get_frameList
import plot_graph

FOURCC = None
FPS = None
HEIGHT = None
WIDTH = None
TOTALFRAME = None


def partical_draw_flow(img, gray, flow, active, typeNum=1):
    '''
    describe optical flow in target area
    img: original image
    gray: gray scale image
    flow: optical flow value array
    active: target region
    typeNum: 1: one color 2: two color
    '''
    x, y = active[:, 0], active[:, 1]
    fx, fy = flow[y, x].T
    # change amount array
    opt_size = np.empty(len(fx))
    for i in range(len(fx)):
        opt_size[i] = (fx[i]**2 + fy[i]**2)**0.5

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    # 1: only one color(red)
    if typeNum == 1:
        col = (0, 0, 255)
        for (x1, y1), (x2, y2) in lines:
            cv2.line(img, (x1, y1), (x2, y2), col, 1)
    # 2: decide accleration(red)/decleration(blue)
    elif typeNum == 2:
        i = 0
        prev = 0
        for (x1, y1), (x2, y2) in lines:
            if opt_size[i] > prev:
                col = (0, 0, 255)
            else:
                col = (255, 0, 0)
            cv2.line(img, (x1, y1), (x2, y2), col, 1)
            if (i + 1) % width == 1:
                prev = 0
            else:
                prev = opt_size[i]
            i += 1
    return img


def count_pixcel(img, step):
    '''
    count target area cell in binary image by step interval
    img: original image
    step: step interval
    '''
    active = []
    count = 0
    for x in range(int(step / 2), WIDTH, step):
        for y in range(int(step / 2), HEIGHT, step):
            if img[y, x][0] == 255:
                active.append([x, y])
                count += 1
    active = np.array(active).reshape(-1, 2)
    return active, count


def calc_mean(active, flow, num_pixcel):
    '''
    calculate mean (optical flow value) in target area
    active: target area
    num_pixcel: amount of pixcel in target area
    '''
    total = 0.0
    for x, y in active:
        total += ((flow[y, x][0])**2 + (flow[y, x][1]**2))**0.5
    return (total / num_pixcel)


def calc_val(active, flow, num_pixcel, mean):
    '''
    calculate variance (optical flow value) in target area
    active: target area
    flow: optical flow value array
    num_pixcel: amount of pixcel in target area
    mean: mean value in target area
    '''
    val = 0.0
    for x, y in active:
        val += (mean - (flow[y, x][0]**2 + flow[y, x][1]**2)**0.5)**2
    val = val / num_pixcel
    return val


def show_gage(pointList, num):
    '''
    display loading gage
    pointList: frame number list to change gage
    num: amount of total frame
    '''
    if num in pointList:
        numIdx = pointList.index(num) + 1
        sys.stderr.write('\rWriting Rate:[{0}{1}] {2}%'.format(
            '*' * numIdx * 2, ' ' * (20 - numIdx), numIdx * 10))


def main(filePath, frameList):
    global FOURCC
    global FPS
    global HEIGHT
    global WIDTH
    global TOTALFRAME
    #-------------------------------------------------------
    # pre processing
    #------------------------------------------------------
    # capture movie and data
    cap = cv2.VideoCapture(filePath)
    FOURCC = int(cv2.VideoWriter_fourcc(*'avc1'))
    FPS = int(cap.get(cv2.CAP_PROP_FPS)/2)
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    TOTALFRAME = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # preprocessing for loding gage
    point = int(TOTALFRAME / 20)
    pointList = []
    for i in range(point, TOTALFRAME, point):
        pointList.append(i)
    sys.stderr.write('\rWriting Rate:[{0}] {1}%'.format(' ' * 20, 0))
    # output movie
    #out = cv2.VideoWriter('../movie/out_' + filePath.split('/')[-1],
    #FOURCC, FPS, (WIDTH, HEIGHT))
    # initial frame
    cap.set(cv2.CAP_PROP_POS_MSEC, 3 * 1000)
    ret, prev = cap.read()
    prevGray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # average value list per frame
    meanList = []
    # variance value list per frame
    valList = []
    # store pixcel of the target area and their number
    mask = cv2.imread('../image/image_data/mask.png', 0)
    mask = cv2.merge((mask, mask, mask))
    active, num_pixcel = count_pixcel(mask, 16)

    #-------------------------------------------------------
    # caluculate optical flow per frame
    #------------------------------------------------------
    frameNum = 0
    while (cap.isOpened()):
        ret,img = cap.read()
        frameNum += 1
        if ret == True:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prevGray,gray,None,0.5,3,15,3,5,1.2,0)
            prevGray = gray
            mean = calc_mean(active,flow,num_pixcel)
            meanList.append(mean)
            val = calc_val(active,flow,num_pixcel,mean)
            valList.append(val)
            # make optical flow image
            #flow_img = partical_draw_flow(img,gray,flow,active,1)
            # restore and display
            #out.write(flow_img)
            show_gage(pointList,frameNum)
            if cv2.waitKey(1)&0xff == 27:
                break
        else:
            break
    plot_graph.mean_val_plot(meanList, valList, filePath, frameList, FPS)
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def make_parse():
    '''
    make argparse
    no argument
    '''
    parser = argparse.ArgumentParser(prog='flow_opt.py',
                        usage='draw optical flow and mean/val graphs',
                        description='description',
                        epilog='end',
                        add_help=True,
                        )

    parser.add_argument('Arg1: input file path',help='string',type=argparse.FileType('r'))

    args = parser.parse_args()
"""
if __name__ == '__main__':
    make_parse()
    start = time.time()
    args = sys.argv
    main(args[1])
    #-------------------------------------------------------
    # Disply time and result
    #-------------------------------------------------------
    elapsed_time = time.time() - start
    minute = int(elapsed_time/60)
    sec = int(elapsed_time - minute*60)
    print('\nelapsed_time: {0}分{1}秒'.format(minute,sec))
"""
