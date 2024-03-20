# FORMATING:
# Yolo 			[x_center,y_center,width,height] (all values normalized)
# Pascal_VOC	[x_min,y_min,x_max,y_max]
# Coco			[x_min,y_min,width,height]

import cv2 as cv
import os
import time
from threading import Thread
from enum import Enum

class Format(Enum):
	YOLO=0
	PASCAL_VOC=1
	COCO=2

dataFormat=Format.YOLO
classes=[]
labels=[]
roi=[]
mp=(15,15)
className='n/a'
windowName='IMG'
imagesToLable='imgs/'
classList='class_list.txt'
xmlPath='xml/'
windowInfo='press: Ctrl+p to show buttons'
createButtons=True
selecting=False
selected=False
run=True
save=None
labelIndex=None
imageGen=os.walk(imagesToLable)
global imgW,imgH

def saveToFile(labels,roi,imgW,imgH,imgName):
	line=''
	extention='.txt'
	for l,r in zip(labels,roi):
		cP=[]
		pt1=r[0]
		pt2=r[1]
		if dataFormat==Format.YOLO:
			w=(abs(pt1[0]-pt2[0]))/imgW
			h=(abs(pt1[1]-pt2[1]))/imgH
			cP.append((pt1[0]+w/2)/imgW)
			cP.append((pt1[1]+h/2)/imgH)
			line=str(l)+' '+str(cP[0])+' '+str(cP[1])+' '+str(w)+' '+str(h)
		if dataFormat==Format.PASCAL_VOC:
			line=l+','+str(pt1[0])+','+str(pt1[1])+','+str(pt2[0])+','+str(pt2[1])
		if dataFormat==Format.COCO:
			w=abs(pt1[0]-pt2[0])
			h=abs(pt1[1]-pt2[1])
			line=l+','+str(pt1[0])+','+str(pt1[1])+','+str(w)+','+str(h)
		with open(xmlPath+imgName[:-4]+extention,'a',encoding='utf-8') as file:
			file.write(line+'\n')

def selectROI(e,x,y,flag,params):
	global roi,selecting,selected,windowInfo,x0,x1,x2,y0,y1,y2,mp
	if e==cv.EVENT_LBUTTONDBLCLK:
		selecting=True
		if x0==0 and y0==0:
			x0,y0=(x+1,y+1)
		x1,y1=(x,y)

	if e==cv.EVENT_MOUSEMOVE and selecting:
		x0,y0=(x,y)
		mp=(x,y)
	elif e==cv.EVENT_MOUSEMOVE:
		mp=(x,y)

	if e==cv.EVENT_LBUTTONUP and selecting:
		selecting=False
		selected=True
		windowInfo='Select class'
		x2,y2=(x,y)
		if x1>x2:
			xT=x2
			x2=x1
			x1=xT
		if y1>y2:
			yT=y2
			y2=y1
			y1=yT

def buttonPress(state,index):
	global labelIndex,windowInfo,classes
	labelIndex=index
	windowInfo='Select '+str(classes[index])

def drawImg(img):
	global createButtons
	if createButtons:
		for i,c in zip(range(len(classes)),classes):
			cv.createButton(str(c),buttonPress,int(i),cv.QT_PUSH_BUTTON,False)
		createButtons=False
	cv.imshow(windowName,img)
	cv.moveWindow(windowName,150,150)
	return (img.shape[1],img.shape[0])

with open(classList) as file:
	classes=[line.rstrip() for line in file]

for _,_,images in imageGen:
	if run:
		x0,x1,x2,y0,y1,y2=(0,0,0,0,0,0)
		for img in images:
			if run:
				imgName=img
				cv.namedWindow(windowName)
				cv.setMouseCallback(windowName,selectROI)
				img=cv.imread(imagesToLable+str(img))
				save=img.copy()

				while run:
					draw=save.copy()
					cv.putText(draw,windowInfo,mp,cv.FONT_HERSHEY_COMPLEX,.3,(255,255,255),1)
					if selecting:
						cv.rectangle(draw,(x1,y1),(x0,y0),(200,200,200),1)
						imgW,imgH=drawImg(draw)
					elif selected:
						cv.rectangle(draw,(x1,y1),(x2,y2),(200,0,0),1)
						drawImg(draw)
						cv.waitKey(1)
						if type(labelIndex)==int:
							cv.putText(save,classes[labelIndex],(x1,y1-5),cv.FONT_HERSHEY_COMPLEX,.3,(250,250,250),1)
							cv.rectangle(save,(x1,y1),(x2,y2),(0,225,0),2)
							imgW,imgH=drawImg(draw)
							labels.append(labelIndex)
							roi.append([(x1,y1),(x2,y2)])
							labelIndex=None
						else:
							x0,x1,x2,y0,y1,y2=(0,0,0,0,0,0)
						selected=False
					else:
						imgW,imgH=drawImg(draw)

					key=cv.waitKey(1)
					if key==ord('n'):
						saveToFile(labels,roi,imgW,imgH,imgName)
						labels=[]
						roi=[]
						break
					if key==ord('q'):
						run=False
						break
				selected=False
				cv.destroyWindow(windowName)
				createButtons=True
