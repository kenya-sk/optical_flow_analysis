#! /usr/bin/env python
#coding: utf-8

import numpy as np
import cv2
import pylab as plt
import time
import sys


#opticalflowを描写する関数
'''
img:元画像
gray:グレースケール変換画像
flow:計算されたflow画像
step:間隔を調節
'''

def draw_flow(img,gray,flow,step=16):
	height,width = img.shape[:2]
	y,x = np.mgrid[step/2:height:step,step/2:width:step].reshape(2,-1).astype(int)
	fx,fy = flow[y,x].T
	#変化量を配列に格納	
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

#active領域内のみに対してオプティカルフローを描写
def partical_draw_flow(img,gray,flow,active):
	x,y = active[:,0],active[:,1]
	fx,fy = flow[y,x].T
	#変化量を配列に格納	
	opt_size = np.empty(len(fx))
	for i in range(len(fx)):
		opt_size[i] = (fx[i]**2 + fy[i]**2)**0.5
	
	lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
	lines = np.int32(lines)
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

#画像の切り出し
def cut_img(img,y1,y2,x1,x2):
	img = img[y1:y2,x1:x2]
	return img


#２値化された画像に対して対象となるピクセル数,step間隔でカウント
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


#対象領域内のオプティカルフローの平均を計算
def calc_mean(active,flow,num_pixcel):
	total = 0.0
	for x,y in active:
			total += ((flow[y,x][0])**2 + (flow[y,x][1]**2))**0.5 
	return (total/num_pixcel)

#対象領域の分散を計算
def calc_val(active,flow,num_pixcel,mean):
	val = 0.0
	for x,y in active:
		val += (mean - (flow[y,x][0]**2 + flow[y,x][1]**2)**0.5)**2
	val = val/num_pixcel
	return val
		
#動画読み込みのゲージ表示
def show_gage(pointList,num):
	if num in pointList:
		numIdx = pointList.index(num)+1
		sys.stderr.write('\rWriting Rate:[{0}{1}] {2}%'.format('*'*numIdx,' '*(20-numIdx),numIdx*5))

#平均、分散の時間推移をプロット
def mean_val_plot(meanList,valList):
	#平均、分散のグラフの描写
	plt.figure(figsize=(12,9))
	#平均
	meanX = []
	for i in range(len(meanList)):
		meanX.append(i/15.3)
	plt.subplot(2,1,1)
	plt.title('Mean of optical flow')
	plt.xlabel('time [s]')
	plt.ylabel('mean')
	plt.plot(meanX,meanList)
	#分散	
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
		

if __name__ == '__main__':
	#入力ファイルと出力先を引数から受け取る
	args = sys.argv
	#処理時間を計測
	start = time.time()
	#入力動画の取得
	cap = cv2.VideoCapture(str(args[1]))
	fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))
	fps = cap.get(cv2.CAP_PROP_FPS)
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	#ゲージ出力のための前処理
	point = int(totalFrame/20)
	pointList = []
	for i in range(point,totalFrame,point):
		pointList.append(i)
	frameNum = 0
	sys.stderr.write('\rWriting Rate:[{0}] {1}%'.format(' '*20,0))
	#出力先
	out = cv2.VideoWriter(str(args[2]),fourcc,15.3,(width,height))
	#最初のフレーム処理
	cap.set(cv2.CAP_PROP_POS_MSEC,3*1000)
	ret,prev = cap.read()
	#prevgray = cv2.bitwise_and(prev,mask)
	prevgray = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
	#各フレームでの平均値を格納したリスト
	meanList = []
	#各フレームでの分散を格納したリスト
	valList = []
	#対象領域のピクセルとその数を格納
	mask = cv2.imread('../image/mask.png',0)
	mask = cv2.merge((mask,mask,mask))
	active,num_pixcel = count_pixcel(mask,8)
	while (cap.isOpened()):
		ret,img = cap.read()
		frameNum += 1
		if ret == True:
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			flow = cv2.calcOpticalFlowFarneback(prevgray,gray,None,0.5,3,15,3,5,1.2,0)
			prevgray = gray
			
			#各フレームで平均値をリストとして格納
			mean = calc_mean(active,flow,num_pixcel)
			meanList.append(mean)
			#各フレームで分散をリストとして格納
			val = calc_val(active,flow,num_pixcel,mean)
			valList.append(val)

			#オプティカルフローを画像上に描写
			#flow_img = draw_flow(img,gray,flow,8)
			flow_img = partical_draw_flow(img,gray,flow,active)

			#保存、表示
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
