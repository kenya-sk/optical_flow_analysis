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

FPS = None
HEIGHT = None
WIDTH = None

def set_capture(filePath):
    cap = cv2.VideoCapture(filePath)
    fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))
    fps = int(cap.get(cv2.CAP_PROP_FPS)/2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fourcc, fps, height, width, totalFrame

def draw_flow(img, maskFlow, step=8):
    '''
    describe optical flow in masked area
    img: original image
    flow: masked optical flow value array
    '''
    x, y = np.mgrid[step/2:WIDTH:step, step/2:HEIGHT:step].reshape(2,-1).astype(int)
    fx, fy = maskFlow[y, x].T

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return img

def show_gage(pointList, num):
    '''
    display loading gage
    pointList: frame number list to change gage
    num: amount of total frame
    '''
    if num in pointList:
        numIdx = pointList.index(num) + 1
        sys.stderr.write('\rWriting Rate:[{0}] {1}%'.format(
            '*' * numIdx, numIdx * 10))


def density_flow(filePath, output=False):
    global HEIGHT
    global WIDTH
    global FPS

    def init_gage(totalFrame):
        point = int(totalFrame / 10)
        pointList = []
        for i in range(point, totalFrame, point):
            pointList.append(i)
        sys.stderr.write('\rWriting Rate:[{0}] {1}%'.format(' ', 0))
        return pointList

    def load_mask(filepath):
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        mask = cv2.merge((mask, mask))
        mask[mask==255] = 1
        return mask

    def calc_flow(prevGray, gray):
        flow = cv2.calcOpticalFlowFarneback(prevGray,gray,None,0.5,3,15,3,5,1.2,0)
        return flow

    #-------------------------------------------------------
    # preprocessing
    #-------------------------------------------------------
    cap, fourcc, FPS, HEIGHT, WIDTH, totalFrame = set_capture(filePath)
    pointList = init_gage(totalFrame)
    if output:
        out = cv2.VideoWriter('../movie/out_' + filePath.split('/')[-1],
        fourcc, FPS, (WIDTH, HEIGHT))
    else:
        pass
    cap.set(cv2.CAP_PROP_POS_MSEC, 3 * 1000)    # initial frame
    ret, prev = cap.read()
    prevGray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    meanList = []
    varList = []
    mask = load_mask('../image/image_data/mask.png')
    #-------------------------------------------------------
    # caluculate optical flow per frame
    #-------------------------------------------------------
    frameNum = 0
    while (cap.isOpened()):
        ret,img = cap.read()
        frameNum += 1
        if ret == True:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            flow = calc_flow(prevGray, gray)
            maskFlow = flow*mask
            mean = np.mean(maskFlow)
            meanList.append(mean)
            var = np.var(maskFlow)
            varList.append(var)
            if output:
                flow_img = draw_flow(img, maskFlow, 8)
                out.write(flow_img)
                cv2.imshow("flow_img", flow_img)
            else:
                pass
            show_gage(pointList,frameNum)
            prevGray = gray
            if cv2.waitKey(1)&0xff == 27:
                break
            if frameNum == 150:
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("meanList = {}".format(meanList))
    print("varList = {}".format(varList))

    return meanList, varList

def main(filePath):
    meanList, varList = density_flow(filePath, True)
    plot_graph.mean_val_plot(meanList, varList, filePath, FPS)


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
