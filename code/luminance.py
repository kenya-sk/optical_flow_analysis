#! /usr/bin/env python
# conding: utf-8

import numpy as np
import cv2
import sys
import time

def region():
    """
    calculate the area with a pixel value of 255 in the grayscale image
    """
    mask = cv2.imread("../image/image_data/flash_Mask.png")
    grayMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    height, width = mask.shape[:2]
    activeArray = np.array([],np.uint8)
    for x in range(0, width):
        for y in range(0, height):
            if grayMask[y, x] == 255:
                activeArray = np.append(activeArray,[x, y])
    activeArray = activeArray.reshape(-1, 2)
    return activeArray


def diff_luminance(prevImg,img,activeArray):
    """
    calculate the difference in luminance value between two frames
    """
    def get_luminance(img, activeArray):
        luminanceArray = np.array([],np.float16)
        for x, y in activeArray:
            b, g, r = img[y, x][:3]
            l = 0.299 * r + 0.587 * g + 0.114 * b
            luminanceArray = np.append(luminanceArray, l)
        return luminanceArray

    prevLuminanceArray = get_luminance(prevImg,activeArray)
    LuminanceArray = get_luminance(img,activeArray)
    diffArray = LuminanceArray - prevLuminanceArray
    return diffArray


def elapsed_time(start):
    elapsed = time.time() - start
    minute = int(elapse/60)
    sec = int(elapse - minute*60)
    print("\nelapsed time: {0}分{1}秒".format(minute,sec))

def main(filePath):
    #-------------------------------------------------------
    # Pre processing
    #-------------------------------------------------------
    # capture movie and data
    cap = cv2.VideoCapture(filePath)
    totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_MSEC, 3 * 1000)
    ret, prevImg = cap.read()
    activeArray = region()
    frameNum = 0
    flashList = []

    #-----------------------------------------------------
    # Caluculate luminance
    #----------------------------------------------------
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret:
            frameNum += 1
            diff_luminance(prevImg, img, activeArray)
            prevImg = img
            sys.stderr.write("\rProcessing Rate: {0}/{1}".format(frameNum,totalFrame))
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return flashList


if __name__ == "__main__":
    start = time.time()
    args=sys.argv
    main(args[1])
    elapsed_time(start)
    
