#! /usr/bin/env python
#coding: utf-8

import numpy as np
import cv2
import pylab as plt
import time
import sys
import argparse


#drow optical flow
'''
img: original image
gray: gray scale image
flow: calculate optical flow array
step: adjust step interval 
'''

def draw_flow(img,gray,flow,step=16):
	height,width = img.shape[:2]
	y,x = np.mgrid[step/2:height:step,step/2:width:step].reshape(2,-1).astype(int)
	fx,fy = flow[y,x].T
	#change amount array
	opt_size = np.empty(len(fx))
	for i in range(len(fx)):
		opt_size[i] = (fx[i]**2 + fy[i]**2)**0.5
	lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
	lines = np.int32(lines)
	#ベクトル描写用のマスク
	#mask = np.zeros((height,width,3),np.uint8)
	#cv2.polylines(mask,lines,False,(0,0,255),2)
	#色固定で描写
	#cv2.polylines(img,lines,False,(0,0,255),1)
	#加減速の判断 フレームより変化が大きければ赤、小さければ青
	i = 0
	prev = 0
	for (x1,y1),(x2,y2) in lines:
		if opt_size[i] > prev:
			col = (0,0,255)
		else:
			col = (255,0,0)	
	
		cv2.line(img,(x1,y1),(x2,y2),col,1)
		if (i+1)%width == 1:
			prev = 0
		else:	
			prev = opt_size[i]
		i += 1
	return img	
	#変化をドットで描く
	'''
	rad = int(step/2)
	i = 0	#ループカウンタ
	for (x1,y1),(x2,y2) in lines:
		pv = img[y1,x1]
		col = (int(pv[0]),int(pv[1]),int(pv[2]))
		#ドットの半径を移動に応じて小さくする
		r = rad - int(rad*abs(fx[i]*.2))
		cv2.circle(mask,(x1,y1),abs(r),col,-1)
			i += 1
	return mask
	'''

#describe optical flow in target area
def partical_draw_flow(img,gray,flow,active):
	x,y = active[:,0],active[:,1]
	fx,fy = flow[y,x].T
	#change amount array
	opt_size = np.empty(len(fx))
	for i in range(len(fx)):
		opt_size[i] = (fx[i]**2 + fy[i]**2)**0.5
	
	lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
	lines = np.int32(lines)
	#decide accleration/decleration
	i = 0
	prev = 0
	for (x1,y1),(x2,y2) in lines:
		if opt_size[i] > prev:
			col = (0,0,255)
		else:
			col = (255,0,0)	
	
		cv2.line(img,(x1,y1),(x2,y2),col,1)
		if (i+1)%width == 1:
			prev = 0
		else:	
			prev = opt_size[i]
		i += 1
	return img

#image segmentation
def cut_img(img,y1,y2,x1,x2):
	img = img[y1:y2,x1:x2]
	return img


#count target area cell in binary image by sstep interval
def count_pixcel(img,step):
	height,width = img.shape[:2]
	active = []
	count = 0
	for x in range(int(step/2),width,step):
		for y in range(int(step/2),height,step):
			if img[y,x][0] == 255:
				active.append([x,y])
				count += 1
	active = np.array(active).reshape(-1,2)			
	return active,count		


#calculate mean (optival flow value) in target area
def calc_mean(active,flow,num_pixcel):
	total = 0.0
	for x,y in active:
			total += ((flow[y,x][0])**2 + (flow[y,x][1]**2))**0.5 
	return (total/num_pixcel)

#calculate variance (optical flow value) in target area
def calc_val(active,flow,num_pixcel,mean):
	val = 0.0
	for x,y in active:
		val += (mean - (flow[y,x][0]**2 + flow[y,x][1]**2)**0.5)**2
	val = val/num_pixcel
	return val
		
#display loding gage
def show_gage(pointList,num):
	if num in pointList:
		numIdx = pointList.index(num)+1
		sys.stderr.write('\rWriting Rate:[{0}{1}] {2}%'.format('*'*numIdx,' '*(20-numIdx),numIdx*5))


#plot mean and variance 
def mean_val_plot(meanList,valList):
	plt.figure(figsize=(12,9))
	#mean list
	meanX = []
	for i in range(len(meanList)):
		meanX.append(i/15.3)
	plt.subplot(2,1,1)
	plt.title('Mean of optical flow')
	plt.xlabel('time [s]')
	plt.ylabel('mean')
	plt.plot(meanX,meanList)
	#variance list
	valX = []
	for i in range(len(valList)):
		valX.append(i/15.3)
	plt.subplot(2,1,2)
	plt.title('Variance of optical flow')
	plt.xlabel('time [s]')
	plt.ylabel('variance')
	plt.plot(valX,valList)
	plt.savefig('../image/opt.png')
	plt.show()
		
#make argparse
def make_parse():
	parser = argparse.ArgumentParser(prog='flow_opt.py',
									usage='draw optical flow and mean/val graphs',
									description='description',
									epilog='end',
									add_help=True,
									)

	parser.add_argument('Arg1: input file path',help='string',type=argparse.FileType('r'))
	parser.add_argument('Arg2: output file path',help='string',type=argparse.FileType('w'))

	args = parser.parse_args()

if __name__ == '__main__':
	make_parse()
	#accept input and output file by argment
	args = sys.argv
	#recode processing time
	start = time.time()
	#capture movie and data
	cap = cv2.VideoCapture(str(args[1]))
	fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))
	fps = cap.get(cv2.CAP_PROP_FPS)
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	#preprocessing for loding gage
	point = int(totalFrame/20)
	pointList = []
	for i in range(point,totalFrame,point):
		pointList.append(i)
	frameNum = 0
	sys.stderr.write('\rWriting Rate:[{0}] {1}%'.format(' '*20,0))
	#output movie
	out = cv2.VideoWriter(str(args[2]),fourcc,15.3,(width,height))
	#initial frame
	cap.set(cv2.CAP_PROP_POS_MSEC,3*1000)
	ret,prev = cap.read()
	prevgray = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
	#average value list per frame
	meanList = []
	#variance value list per frame
	valList = []
	#store pixcel of the target area and their number 
	mask = cv2.imread('../image/mask.png',0)
	mask = cv2.merge((mask,mask,mask))
	active,num_pixcel = count_pixcel(mask,8)
	#main process
	while (cap.isOpened()):
		ret,img = cap.read()
		frameNum += 1
		if ret == True:
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			flow = cv2.calcOpticalFlowFarneback(prevgray,gray,None,0.5,3,15,3,5,1.2,0)
			prevgray = gray
			mean = calc_mean(active,flow,num_pixcel)
			meanList.append(mean)
			val = calc_val(active,flow,num_pixcel,mean)
			valList.append(val)
			#make optical flow image
			flow_img = partical_draw_flow(img,gray,flow,active)
			#restore and display
			out.write(flow_img)
			#cv2.imshow('optical flow',flow_img)
			show_gage(pointList,frameNum)
			el_time = time.time()
			if el_time - start > 600:
				break

			if cv2.waitKey(1)&0xff == 27:
				break
		else:
			break

	cap.release()
	out.release()
	cv2.destroyAllWindows()

	
	elapsed_time = time.time() - start
	minute = int(elapsed_time/60)
	sec = int(elapsed_time - minute*60)
	print('\nelapsed_time: {0}分{1}秒'.format(minute,sec))
	
	mean_val_plot(meanList,valList)
