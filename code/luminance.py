#! /usr/bin/env python
#conding: utf-8

import numpy as np
import cv2
import sys

def get_luminance(img,active,frameNum):
	maxL=0.0
	lumThresh=200
	for y,x in active:
		b,g,r=img[y,x][:3]
		l=0.299*r+0.587*g+0.114*b 
		if maxL<l: maxL=l
	if maxL>lumThresh: return frameNum

def region():
	mask=cv2.imread('../image/image_data/flash_Mask.png')
	grayMask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
	height,width=mask.shape[:2]
	active=[]
	for x in range(0,width):
		for y in range(0,height):
			if grayMask[y,x][0]==255: active.append([x,y])
	active = np.array(active).reshape(-1,2)
	return active
	
