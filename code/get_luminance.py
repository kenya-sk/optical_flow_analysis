#! /usr/bin/env python
#conding: utf-8

import numpy as np
import cv2
import sys

def get_luminance(img):
	maxY = 0.0
	for y,x in active:
		b,g,r = img[y,x][:3]
		y = 0.299*r+0.587*g+0.114*b 
		if maxY < y:
			maxY = y
			

def main(fileName):
	cap = cv2.VideoCapture(fileName)
	fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))
	fps = cap.get(cv2.CAP_PROP_FPS)
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	out = cv2.VideoWriter('../movie/gray_output.mp4',fourcc,15.3,(width,height))
	#initial frame
	cap.set(cv2.

if __name__ == '__main__':
	get_luminance()

