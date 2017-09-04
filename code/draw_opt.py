#! /usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import sys
import argparse


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
    height, width = img.shape[:2]
    active = []
    count = 0
    for x in range(int(step / 2), width, step):
        for y in range(int(step / 2), height, step):
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


def mean_val_plot(meanList, valList, filePath, frameList):
    '''
    plot mean and variance
    meanList: mean value list
    valList: variance value list
    frameList: two dimention list
    '''
    def fill_region(frameList):
        if len(frameList) != 0:
            for fram in frameList:
                plt.axvspan(fram[0], fram[1], facecolor = 'y', alpha = 0.5)

    plt.figure(figsize=(12, 9))
    # mean list
    meanX = []
    for i in range(len(meanList)):
        meanX.append(i / 15.3)
    plt.subplot(2, 1, 1)
    plt.title('Mean of optical flow')
    plt.xlabel('time [s]')
    plt.ylabel('mean')
    plt.ylim(0.0, 2.0)
    plt.plot(meanX, meanList)
    fill_region(frameList)
    # variance list
    valX = []
    for i in range(len(valList)):
        valX.append(i / 15.3)
    plt.subplot(2, 1, 2)
    plt.title('Variance of optical flow')
    plt.xlabel('time [s]')
    plt.ylabel('variance')
    plt.ylim(0, 10)
    plt.plot(valX, valList)
    fill_region(frameList)
    fileName = filePath.split('/')[-1].split('.')[0] + '.png'
    plt.savefig('../image/20170429_13_15/' + fileName)
    print('\n'+fileName + ' graph success !\n')


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


def time_to_frame(fps,timeFilePath):
    subList = []
    frameList = []
    with open (timeFilePath,"r") as f:
        time = f.readlines()
        for i in range(len(time)):
            if time[i] != "\n":
                subList.append(int(time[i].strip("\n")))
            else:
                frameList.append(subList)
                subList = []
        frameList.append(subList)                
    return(frameList)


def main(filePath):
    #-------------------------------------------------------
    # pre processing
    #------------------------------------------------------
    # capture movie and data
    cap = cv2.VideoCapture(filePath)
    fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))
    fps = int(cap.get(cv2.CAP_PROP_FPS)/2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # preprocessing for loding gage
    point = int(totalFrame / 20)
    pointList = []
    for i in range(point, totalFrame, point):
        pointList.append(i)
    sys.stderr.write('\rWriting Rate:[{0}] {1}%'.format(' ' * 20, 0))
    # output movie
    out = cv2.VideoWriter('../movie/out_' + filePath.split('/')[-1],
    fourcc, fps, (width, height))
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
    active, num_pixcel = count_pixcel(mask, 2)

    #----------------------------------------------------
    # pre processing for sparse optical flow
    #----------------------------------------------------
    # corner detection parameter of Shi-Tomasi
    feature_params = dict(maxCorners = 200,
                            qualityLevel = 0.001,
                            minDistance = 10,
                            blockSize = 5)
    # parameter of Lucas-Kanade method
    lk_params = dict(winSize = (20,20),
                     maxLevel = 5,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
    color = np.random.randint(0,255, 255)
    prevFeature = cv2.goodFeaturesToTrack(prevGray,mask=None,**feature_params)

	#-------------------------------------------------------
	# caluculate optical flow per frame
	#------------------------------------------------------
    framNum = 0
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
            flow_img = partical_draw_flow(img,gray,flow,active,1)
            # restore and display
            out.write(flow_img)
            show_gage(pointList,frameNum)
            
            """
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            nextFeature, status, err = cv2.calcOpticalFlowPyrLK(prevGray, gray, prevFeature, None, **lk_params)
            
            prevGood = prevFeature[status == 1]
            nextGood = nextFeature[status == 1]

            for i, (next_point,prev_point) in enumerate(zip(nextGood, prevGood)):
                prevX, prevY = prev_point.ravel()
                nextX, nextY = next_point.ravel()
                img = cv2.circle(img, (nextX,nextY), 5, (0,0,255),-1)
            out.write(img)

            prevGray = gray.copy()
            prevFeature = nextGood.reshape(-1,1,2)
            """

            if cv2.waitKey(1)&0xff == 27:
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
	
    framList = [[0,0]]
    mean_val_plot(meanList,valList,filePath,framList)



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
	# parser.add_argument('Arg2: output file path',help='string',type=argparse.FileType('w'))

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

