#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import time


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1),
                    Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    return connects


def regionGrow(img, thresh, p=1):
    height, weight = img.shape
    seedMark = np.ones((height, weight), dtype=np.uint8) * 255
    seedList = []
    # for seed in seeds:
    #    seedList.append(seed)
    label = 0
    connects = selectConnects(p)
    currentPoint = Point(25, 25)
    seedMark[currentPoint.x, currentPoint.y] = label
    for i in range(8):
        tmpX = currentPoint.x + connects[i].x
        tmpY = currentPoint.y + connects[i].y
        if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
            continue
        grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
        if grayDiff < thresh and seedMark[tmpX, tmpY] == 255:
            seedMark[tmpX, tmpY] = label
            seedList.append(Point(tmpX, tmpY))
    return seedMark


def detection(img, click):
    # seeds = [Point(i[0],i[1]) for i in clicks]
    lights = '绿'
    # img = cv2.resize(img,(200,int(200/img.shape[1]*img.shape[0])))
    img = img[click[0] - 25:click[0] + 25, click[1] - 25:click[1] + 25]
    img = cv2.GaussianBlur(img, (5, 5), 0, 0)
    tmp = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    '''
    clicks=[]
    cv2.imshow('img',img)
    cv2.setMouseCallback('img', on_mouse )
    cv2.waitKey()
    '''

    # seeds = [Point(10,10),Point(82,150),Point(20,300)]
    # seeds = [Point(i[0],i[1]) for i in clicks]
    # seeds = [Point(5,5)]
    binaryImg = regionGrow(img, 10)
    # binaryImg = cv2.bitwise_and(binaryImg,img)
    # binaryImg = cv2.cvtColor(binaryImg , cv2.COLOR_)
    contours, hierarchy = cv2.findContours(binaryImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hsv = 0
    count = 0
    for contour in contours:
        (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
        circle_center = (int(circle_x), int(circle_y))
        radius = int(radius)
        if radius <= 0 or radius > 50:
            #    print(radius)
            continue
        # print(radius)
        # print(circle_center)
        tmp1 = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
        # print(tmp1.shape)
        hsv += tmp1[int(circle_y)][int(circle_x)][0]
        # count=count+1
        #print('HSV颜色空间为{}'.format(tmp1[int(circle_y)][int(circle_x)]))
        cv2.circle(tmp, circle_center, radius, (0, 0, 255), 2)

        cv2.imshow('regiongrow', binaryImg)
        # 显示指示灯位置
        '''
        cv2.imshow('tmp',tmp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        hsv += tmp1[int(circle_y)][int(circle_x)][0]
        count = count + 1
    hsv = hsv // count
    # print(hsv)
    # cv2.imshow('regiongrow', binaryImg)
    if 0 <= hsv <= 50 or 156 <= hsv <= 180:
        lights = '红'
    else:
        lights = '绿'
    return lights

# In[ ]:




