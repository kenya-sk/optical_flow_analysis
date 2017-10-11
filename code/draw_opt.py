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

class FeatureError(Exception):
    def __init__(self, value):
        self.value = value

def set_capture(filePath):
    cap = cv2.VideoCapture(filePath)
    fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))
    fps = int(cap.get(cv2.CAP_PROP_FPS)/2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fourcc, fps, height, width, totalFrame

def draw_dense_flow(img, maskFlow, step=8):
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
        sys.stderr.write('\rWriting Rate:[{0}] {1}%'.format('*' * numIdx, numIdx * 10))


def calc_flow(filePath, output=False):
    global HEIGHT
    global WIDTH
    global FPS

    def init_gage(totalFrame):
        point = int(totalFrame / 20)
        pointList = []
        for i in range(point, totalFrame, point):
            pointList.append(i)
        sys.stderr.write('\rWriting Rate:[{0}] {1}%'.format(' ', 0))
        return pointList

    def load_mask(filepath):
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        #mask = cv2.merge((mask, mask))
        mask[mask==255] = 1
        return mask

    def calc_dense_flow(prevGray, gray):
        flow = cv2.calcOpticalFlowFarneback(prevGray,gray,None,0.5,3,15,3,5,1.2,0)
        return flow

    def set_sparse_parm():
        #corner detection parameter of Shi-Tomasi
        feature_params = dict(maxCorners = 200,
                                qualityLevel = 0.001,
                                minDistance = 10,
                                blockSize = 5)
        #parameter of Lucas-Kanade method
        lk_params = dict(winSize = (20, 20),
                            maxLevel = 5,
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        return feature_params, lk_params

    def get_feature(prevGray, gray, feature_params, lk_params, mask=None):
        prevFeatureFiltered = np.array([0])
        nextFeatureFiltered = np.array([0])
        if mask is None:
            prevFeature = cv2.goodFeaturesToTrack(prevGray, mask=None, **feature_params)
        else:
            prevFeature = cv2.goodFeaturesToTrack(prevGray, mask=mask, **feature_params)
        nextFeature, status, err = cv2.calcOpticalFlowPyrLK(prevGray, gray, prevFeature, None, **lk_params)
        prevFeatureFiltered = prevFeature[status == 1]
        nextFeatureFiltered = nextFeature[status == 1]
        return prevFeatureFiltered, nextFeatureFiltered


    def calc_sparse_flow(prevFeatureFiltered, nextFeatureFiltered):
        sparseFlow = np.zeros((nextFeatureFiltered.shape[0],3))
        try:
            if prevFeatureFiltered.shape[0] == 0:
                raise FeatureError("Not detect feature")
            for i, (prevPoint, nextPoint) in enumerate(zip(prevFeatureFiltered, nextFeatureFiltered)):
                prevX, prevY = prevPoint.ravel()
                nextX, nextY = nextPoint.ravel()
                sparseFlow[i][0] = nextX - prevX
                sparseFlow[i][1] = nextY - prevY
        except FeatureError:
            pass
        return sparseFlow

    def make_spase_flow_image(img, flowMask, prevFeatureFiltered, nextFeatureFiltered):
        for i, (nextPoint, prevPoint) in enumerate(zip(nextFeatureFiltered, prevFeatureFiltered)):
            prevX, prevY = prevPoint.ravel()
            nextX, nextY = nextPoint.ravel()
            flowMask = cv2.line(flowMask, (nextX, nextY), (prevX, prevY), (0, 0, 255), 2)
            img = cv2.circle(img, (nextX, nextY), 3, (0, 0, 255), -1)
        flowImg = cv2.add(img, flowMask)
        return flowImg

    def calc_norm(flow):
        normFx = flow[:,0]**2
        normFy = flow[:, 1]**2
        flowNorm = (normFx + normFy)**0.5
        flowNorm = flowNorm[flowNorm > 0]
        return flowNorm

    #-------------------------------------------------------
    # preprocessing
    #-------------------------------------------------------
    cap, fourcc, FPS, HEIGHT, WIDTH, totalFrame = set_capture(filePath)
    print("fps = {}".format(FPS))
    pointList = init_gage(totalFrame)
    if output:
        out = cv2.VideoWriter('../movie/out_' + filePath.split('/')[-1],
        fourcc, FPS, (WIDTH, HEIGHT))
    else:
        pass
    cap.set(cv2.CAP_PROP_POS_MSEC, 3 * 1000)    # initial frame
    ret, prev = cap.read()
    prevGray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    maxList = []
    meanList = []
    varList = []
    tmpMean = 0
    tmpVar = 0
    tmpMax = 0
    feature_params, lk_params = set_sparse_parm()
    mask = load_mask('../image/image_data/mask.png')
    flowMask = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    #-------------------------------------------------------
    # caluculate optical flow per frame
    #-------------------------------------------------------
    frameNum = 0
    while (cap.isOpened()):
        ret,img = cap.read()
        frameNum += 1
        if ret == True:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            """
            flow = calc_dense_flow(prevGray, gray)
            maskFlow = flow*mask
            flowNorm = calc_norm(maskFlow, 8)
            try:
                mean = np.mean(flowNorm)
            except RuntimeWarning:
                print(frameNum)
            meanList.append(mean)
            var = np.var(flowNorm)
            varList.append(var)
            """
            prevFeatureFiltered, nextFeatureFiltered = get_feature(prevGray, gray, feature_params, lk_params, mask)
            sparseFlow = calc_sparse_flow(prevFeatureFiltered, nextFeatureFiltered)
            flowNorm = calc_norm(sparseFlow)
            try:
                tmpMax += max(flowNorm)
            except ValueError:
                pass
            if output:
                #flowImg = draw_dense_flow(img, maskFlow, 8)
                try:
                    mean = np.mean(flowNorm)
                except RuntimeWarning:
                    print(frameNum)
                tmpMean += mean
                var = np.var(flowNorm)
                tmpVar += var
                if frameNum % int(FPS) == 0:
                    flowMask = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                    meanList.append(tmpMean/int(FPS))
                    varList.append(tmpVar/int(FPS))
                    maxList.append(tmpMax)
                    tmpMax = 0
                    tmpMean, tmpVar = 0, 0
                else:
                    pass
                flowImg = make_spase_flow_image(img, flowMask, prevFeatureFiltered, nextFeatureFiltered)
                out.write(flowImg)
                #cv2.imshow("flow img", flowImg)
            else:
                pass
            show_gage(pointList,frameNum)
            prevGray = gray
            if cv2.waitKey(1)&0xff == 27:
                break
        else:
            break
    cap.release()
    if output:  out.release()
    cv2.destroyAllWindows()
    print("meanList length: {}".format(len(meanList)))
    print("varList length: {}".format(len(varList)))

    return meanList, varList, maxList

def main(filePath):
    meanList, varList, maxList = calc_flow(filePath, True)
    #plot_graph.mean_val_plot(meanList, varList, filePath)
    plot_graph.max_plot(maxList, filePath)



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
