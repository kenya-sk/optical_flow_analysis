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
    fps = cap.get(cv2.CAP_PROP_FPS)
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

def show_gage(point_lst, num):
    '''
    display loading gage
    point_lst: frame number list to change gage
    num: amount of total frame
    '''
    if num in point_lst:
        numIdx = point_lst.index(num) + 1
        sys.stderr.write('\rWriting Rate:[{0}] {1}%'.format('*' * numIdx, numIdx * 10))


def calc_flow(filePath, tmpMean_lst, tmpVar_lst, tmpMax_lst, window=30, output=False):
    global HEIGHT
    global WIDTH
    global FPS

    def init_gage(totalFrame):
        point = int(totalFrame / 10)
        point_lst = []
        for i in range(point, totalFrame, point):
            point_lst.append(i)
        sys.stderr.write('\rWriting Rate:[{0}] {1}%'.format(' ', 0))
        return point_lst

    def load_mask(filepath):
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        #mask = cv2.merge((mask, mask))
        mask[mask==255] = 1
        return mask

    def make_coordinate_lst(prevFeatureFiltered, coordinateX_lst, coordinateY_lst):
        for prevPoint in prevFeatureFiltered:
            prevX, prevY = prevPoint.ravel()
            coordinateX_lst.append(prevX)
            coordinateY_lst.append(prevY)
        return coordinateX_lst, coordinateY_lst

    def calc_dense_flow(prevGray, gray, mask=None):
        flow = cv2.calcOpticalFlowFarneback(prevGray,gray,None,0.5,3,15,3,5,1.2,0)
        maskFlow = mask * flow
        return maskFlow

    def set_sparse_parm():
        #corner detection parameter of Shi-Tomasi
        feature_params = dict(maxCorners = 150,
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
        feature_count(frameNum, prevFeatureFiltered, nextFeatureFiltered)
        return prevFeatureFiltered, nextFeatureFiltered


    def calc_sparse_flow(prevFeatureFiltered, nextFeatureFiltered):
        sparseFlow = np.zeros((nextFeatureFiltered.shape[0],3))
        try:
            if prevFeatureFiltered.shape[0] == 0:
                raise FeatureError("Not detect feature")
            for i, (prevPoint, nextPoint) in enumerate(zip(prevFeatureFiltered, nextFeatureFiltered)):
                prevX, prevY = prevPoint.ravel()
                nextX, nextY = nextPoint.ravel()
                flowX = nextX - prevX
                flowY = nextY - prevY
                if flowX > 1 or flowY > 1:
                    sparseFlow[i][0] = nextX - prevX
                    sparseFlow[i][1] = nextY - prevY
                else:
                    pass
        except FeatureError:
            pass
        return sparseFlow

    def make_spase_flow_image(img, flowMask, prevFeatureFiltered, nextFeatureFiltered):
        for nextPoint, prevPoint in zip(nextFeatureFiltered, prevFeatureFiltered):
            prevX, prevY = prevPoint.ravel()
            nextX, nextY = nextPoint.ravel()
            flowX = nextX - prevX
            flowY = nextY - prevY
            flowMask = cv2.line(flowMask, (nextX, nextY), (prevX, prevY), (0, 0, 255), 2)
            img = cv2.circle(img, (nextX, nextY), 3, (0, 0, 255), -1)

        flowImg = cv2.add(img, flowMask)
        return flowImg

    def calc_norm(flow):
        normFx = flow[:,0]**2
        normFy = flow[:, 1]**2
        flowNorm = (normFx + normFy)**0.5
        #flowNorm = flowNorm[flowNorm > 5]
        return flowNorm

    def feature_count(frameNum, prevFeatureFiltered, nextFeatureFiltered):
        with open("./feature.txt", 'a') as f:
            text = "NUM:{0}  prev:{1}  next:{2}\n".format(frameNum, prevFeatureFiltered.shape[0], nextFeatureFiltered.shape[0])
            f.write(text)

    #-------------------------------------------------------
    # preprocessing
    #-------------------------------------------------------
    cap, fourcc, FPS, HEIGHT, WIDTH, totalFrame = set_capture(filePath)
    print("fps = {}".format(FPS))
    print("start calculation of optical flow")
    point_lst = init_gage(totalFrame)
    if output:
        out = cv2.VideoWriter('../movie/out/out_' + filePath.split('/')[-1],
        fourcc, FPS, (WIDTH, HEIGHT))
    else:
        pass
    ret, prev = cap.read()
    prevGray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    coordinateX_lst = []
    coordinateY_lst = []
    max_lst = []
    mean_lst = []
    var_lst = []
    windowSize = window  #int(FPS) #shift caluculate region
    #tmpMean_lst = [0 for i in range(windowSize - 1)]
    #tmpVar_lst = [0 for i in range(windowSize - 1)]
    #tmpMax_lst = [0 for i in range(windowSize - 1)]
    #tmpMax_lst = []
    #tmpVar_lst = []
    #tmpMean_lst = []
    feature_params, lk_params = set_sparse_parm()
    mask = load_mask('../image/image_data/mask.png')
    flowMask = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    #dens_mask = cv2.imread("../image/image_data/density.png")
    #-------------------------------------------------------
    # caluculate optical flow per frame
    #-------------------------------------------------------
    frameNum = 0
    while (cap.isOpened()):
        ret,img = cap.read()
        frameNum += 1
        if ret == True:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            prevFeatureFiltered, nextFeatureFiltered = get_feature(prevGray, gray, feature_params, lk_params, mask)
            coordinateX_lst, coordinateY_lst = make_coordinate_lst(prevFeatureFiltered, coordinateX_lst, coordinateY_lst)
            sparseFlow = calc_sparse_flow(prevFeatureFiltered, nextFeatureFiltered)
            flowNorm = calc_norm(sparseFlow)

            #make list per frame
            try:
                flowMax = max(flowNorm)
            except ValueError:
                flowMax = 0
            if flowNorm.shape[0] != 0:
                flowMean = np.mean(flowNorm)
                flowVar = np.var(flowNorm)
            else:
                flowMean = 0
                flowVar = 0

            #remove disturbance
            if flowVar > 200:
                flowMean = 0
                flowVar = 0
                flowMax = 0
            else:
                pass

            tmpMean_lst.append(flowMean)
            tmpVar_lst.append(flowVar)
            tmpMax_lst.append(flowMax)
            assert len(tmpMax_lst) == windowSize, "tmpMax_lst length is not windowSize"
            assert len(tmpMean_lst) == windowSize, "tmpMean_lst length is not windowSize"
            assert len(tmpVar_lst) == windowSize, "tmpVar_lst length is not windowSize"
            #add the element of the current frame
            #if frameNum % int(FPS) == 0:
            max_lst.append(sum(tmpMax_lst))
            mean_lst.append(sum(tmpMean_lst))
            var_lst.append(sum(tmpVar_lst))
                #tmpMax_lst = []
                #tmpVar_lst = []
                #tmpMean_lst = []
            #delete the first element
            tmpMax_lst.pop(0)
            tmpMean_lst.pop(0)
            tmpVar_lst.pop(0)
            assert len(tmpMax_lst) == windowSize - 1, "tmpMax_lst element is not deleted"
            assert len(tmpMean_lst) == windowSize - 1, "tmpMean_lst element is not deleted"
            assert len(tmpVar_lst) == windowSize - 1, "tmpVar_lst element is not deleted"

            if output:
                #make cumulative image
                flowImg = make_spase_flow_image(img, flowMask, prevFeatureFiltered, nextFeatureFiltered)
                #test densuty map
                #densImg = cv2.addWeighted(flowImg, 0.5, dens_mask, 0.5, 0)
                #write frame number
                text = "[ Frame Number: {0:04d} ]".format(frameNum)
                cv2.putText(flowImg, text, (850, 680), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                cv2.imshow("flow img", flowImg)
                out.write(flowImg)
                if frameNum % int(FPS) == 0:
                    flowMask = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            else:
                pass
            show_gage(point_lst,frameNum)
            prevGray = gray
            if cv2.waitKey(1)&0xff == 27:
                break
        else:
            break
    cap.release()
    if output:  out.release()
    cv2.destroyAllWindows()

    np.save("./mean.npy", np.array(mean_lst))
    np.save("./var.npy", np.array(var_lst))
    np.save("./max.npy", np.array(max_lst))

    print("\nend calculation optical flow")
    print("\nmeanList length: {}".format(len(mean_lst)))
    print("\nvarList length: {}".format(len(var_lst)))
    print("\nmaxList length: {}".format(len(max_lst)))

    #fileName = filePath.split('/')[-1].split('.')[0]
    #np.save("../data/coordinate/{}_coordinateX.npy".format(fileName), np.array(coordinateX_lst))
    #np.save("../data/coordinate/{}_coordinateY.npy".format(fileName), np.array(coordinateY_lst))

    plot_graph.mean_var_plot(mean_lst, var_lst, window, filePath)
    plot_graph.max_plot(max_lst, window, filePath)

    return mean_lst, var_lst, max_lst, tmpMean_lst, tmpVar_lst, tmpMax_lst

def main(filePath, window=30):
    tmpMean_lst = [0 for i in range(window - 1)]
    tmpVar_lst = [0 for i in range(window - 1)]
    tmpMax_lst = [0 for i in range(window - 1)]
    mean_lst, var_lst, max_lst, _, _, _ = calc_flow(filePath, tmpMean_lst, tmpVar_lst, tmpMax_lst, window, True)
    plot_graph.mean_var_plot(mean_lst, var_lst, window, filePath)
    plot_graph.max_plot(max_lst, window, filePath)



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
