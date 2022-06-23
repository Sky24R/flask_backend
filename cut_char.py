import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os


#根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
	up_point = -1#上升点
	is_peak = False
	if histogram[0] > threshold:
		up_point = 0
		is_peak = True
	wave_peaks = []
	for i,x in enumerate(histogram):
		if is_peak and x < threshold:
			if i - up_point > 2:
				is_peak = False
				wave_peaks.append((up_point, i))
		elif not is_peak and x >= threshold:
			is_peak = True
			up_point = i
	if is_peak and up_point != -1 and i - up_point > 4:
		wave_peaks.append((up_point, i))
	return wave_peaks

#根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves ,m ,M ):
	part_cards = []
	for wave in waves:
		part_cards.append(img[min(m[wave[0]:wave[1]]):max(M[wave[0]:wave[1]]), wave[0]:wave[1]])
	return part_cards



img=cv2.imread('./img/5.jpg')  

if img.shape[0]>1000:
	img=cv2.resize(img,(0,0),fx=0.1,fy=0.1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   
# gray = cv2.bitwise_not(gray)

ret,thresh=cv2.threshold(gray,130,255,cv2.THRESH_BINARY)  
#ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)  

print(thresh.shape)
h,w=thresh.shape
m=[h-1]*w
M=[0]*w
for j in range(w):
	for i in range(h):
		if thresh[i][j]==255:
			m[j]=min(m[j],i)
			M[j]=max(M[j],i)
# print(m,M)

row_num, col_num= gray.shape[:2]
# 去掉车牌上下边缘1个像素，避免白边影响阈值判断
thresh = thresh[1:row_num-1]
y_histogram = np.sum(thresh, axis=0)
y_min = np.min(y_histogram)
y_average = np.sum(y_histogram)/y_histogram.shape[0]
y_threshold = (y_min + y_average)/5    # U和0要求阈值偏小，否则U和0会被分成两半
wave_peaks = find_waves(y_threshold, y_histogram)
#分割字符

part_cards = seperate_card(thresh, wave_peaks ,m ,M )
cv2.imshow('img',img) 
cv2.waitKey(0) 
# print(len(img),len(img[0]))

for i in range(len(part_cards)):
	im=part_cards[i]
	cv2.imshow('{}'.format(i),im)
	cv2.waitKey(0)
	# cv2.imwrite('./'+str(i)+'.jpg',im)

cv2.destroyAllWindows()  
