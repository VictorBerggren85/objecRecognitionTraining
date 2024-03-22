import cv2 as cv
import os
import time
from picamera2 import Picamera2

imgNr=0
capWidth=415
capHeight=415
capFormat='XBGR8888'
run=True
path='imgs/'
fileExtention='.jpg'

cap=Picamera2()            
cap.configure(cap.create_preview_configuration(main={"format":capFormat,"size": (capWidth,capHeight)}))
cap.start()
os.chdir(path)

while run:
	frame=cap.capture_array()
	frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
	frameDisp=frame.copy()
	cv.putText(frameDisp,'Images saved: '+str(imgNr),(25,25),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
	cv.imshow('test',frameDisp)
		
	if cv.waitKey(1)==ord('p'):
		imgNr+=1
		filename=str(time.time())+'-'+str(imgNr)+fileExtention
		cv.imwrite(filename,frame)
		print('Save img')
	if cv.waitKey(2)==ord('q'):
		run=False
		print('Quit')

cv.destroyAllWindows()
