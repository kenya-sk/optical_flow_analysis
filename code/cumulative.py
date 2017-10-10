#! /usr/bin/env python
#coding: utf-8


import numpy as np
import cv2
import time
import sys

import draw_opt

FPS = None
HEIGHT = None
WIDTH = None
WINDOWSIZE = None


def make_cumulative_video(filePath, windowSize=5):
    global FPS
    global HEIGHT
    global WIDTH
    global WINDOWSIZE

    def calc_cumulative_flow(cmlFlow_arr, flow):
        flow_arr = np.zeros((1,720, 1280, 2))
        for x in range(WIDTH):
            for y in range(HEIGHT):
                i = y
                j = x
                #assert flowData.shape == (720, 1280, 2)
                for flow in cmlFlow_arr:
                    ni = i + flow[int(i)][int(j)][0]
                    nj = j + flow[int(i)][int(j)][1]
                    i,j = ni, nj
                flow_arr[0][y][x][0] = i - y
                flow_arr[0][y][x][0] = j - x
        cmlFlow_arr = np.vstack((cmlFlow_arr, flow_arr))
        print("end caluculate cumulative")
        return cmlFlow_arr

    def draw_cumulative_flow(img, cmlFlow_arr, step=8):
        x, y = np.mgrid[step/2:WIDTH:step, step/2:HEIGHT:step].reshape(2,-1).astype(int)
        for time in range(WINDOWSIZE):
            fx, fy = cmlFlow_arr[time][y, x].T
            lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
            lines = np.int32(lines)
            for (x1, y1), (x2, y2) in lines:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        return img

    cap, fourcc, FPS, HEIGHT, WIDTH, totalFrame = draw_opt.set_capture(filePath)
    output = cv2.VideoWriter("../movie/cumulative.mp4", fourcc, FPS, (WIDTH, HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, 3 * 1000)    # initial frame
    ret, prev = cap.read()
    prevGray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    WINDOWSIZE = windowSize
    img_arr = np.zeros((WINDOWSIZE, HEIGHT, WIDTH, 3))
    cmlFlow_arr = np.zeros((WINDOWSIZE + 1, HEIGHT, WIDTH, 2))
    frameNum  = 0


    while(cap.isOpened()):
        ret, img = cap.read()
        frameNum += 1
        if ret == True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prevGray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            if frameNum < 20:
                cmlFlow_arr = calc_cumulative_flow(cmlFlow_arr, flow)
                cmlImg = draw_cumulative_flow(img_arr[0], cmlFlow_arr[:5,:,:,:], step=8)
                output.write(cmlImg)
                cv2.imshow("cmlImg", cmlImg)
                cmlFlow_arr = np.delete(cmlFlow_arr, 0, 0)
                img_arr = np.delete(img_arr, 0, 0)
                img_arr = np.vstack((img_arr, img.reshape(1,HEIGHT, WIDTH, 3)))
            prevGray = gray
        else:
            break
    cap.release()
    output.release()
    cv2.destroyAllWindows()

def main(filePath):
    make_cumulative_video(filePath, windowSize=5)


if __name__ == "__main__":
    start = time.time()
    args = sys.argv
    main(args[1])
    elapsed_time = time.time() - start
    minute = int(elapsed_time / 60)
    second = int(elapsed_time - minute*60)
    print("\nelapsed time: {0}分{1}秒".format(minute, second))
