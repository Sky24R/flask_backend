import imutils
import numpy as np
from numpy.lib.histograms import histogram
import tensorflow as tf
import os
import random
import time

import torch
import torchvision.transforms as transforms
from PIL import Image #用于读取数据
from model_lenet import LeNet
from imutils import contours
import cv2
import numpy as np
import matplotlib.pyplot as plt

tar_temp=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.','A','C','E','F','H','L','P']

#输入二值图像，输出二值图像
#根据黑白占比判断是黑底白字/白底黑字
def bit_not(binary):
    height,width = binary.shape
    m = [height-1]*width
    M = [0]*width
    black,white = 0,0
    for j in range(width):
        for i in range(height):
            if binary[i][j] == 255:
                white += 1
                m[j] = min(m[j] ,i)#########？
                M[j] = max(M[j] ,i)#######？
            if binary[i][j] == 0:
                black += 1
    if black < white:
        binary = cv2.bitwise_not(binary)
    return binary

#输入RGB图，输出二值图像、竖直、水平投影

#画出水平、竖直直方图投影
def draw_hist(img):
    img = cv2.resize(img,(200,int(200/img.shape[1]*img.shape[0])))
    img = cv2.GaussianBlur(img,(5,5),0,0)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   

    ret,thresh=cv2.threshold(gray,130,255,cv2.THRESH_BINARY)  
    ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
    thresh = bit_not(thresh)


    thresh1=thresh.copy()
    thresh2=thresh.copy()
    h,w=thresh.shape

    row = [0]*w
    #记录每一列的波峰
    for j in range(w): #遍历一列 
        for i in range(h):  #遍历一行
            if  thresh1[i,j]==0:  #如果改点为黑点
                row[j]+=1  		#该列的计数器加一计数
                thresh1[i,j]=255  #记录完后将其变为白色 
            
    for j  in range(w):  #遍历每一列
        for i in range(row[j]):  #从该列应该变黑的最顶部的点开始向最底部涂黑
            thresh1[i,j]=0   #涂黑

    # plt.imshow(thresh1,cmap=plt.gray())
    # plt.show()

    col = [0]*h 
    for j in range(h):  
        for i in range(w):  
            if  thresh2[j,i]==0: 
                col[j]+=1 
                thresh2[j,i]=255
            
    for j  in range(h):  
        for i in range(w-col[j],w):   
            thresh2[j,i]=0    

    cv2.imshow('img',img)
    cv2.imshow('threshold',thresh)
    cv2.imshow('row',thresh1)  
    cv2.imshow('col',thresh2)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

#根据阈值、直方图，找出波峰
def find_waves(threshold,histogram):
    up_point = -1
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

#输入水平波峰,截取水平方向的图片
#返回值为一对对角点（左上、右下）
def cut(img ):
    img = cv2.resize(img,(200,int(200/img.shape[1]*img.shape[0])))
    img = cv2.GaussianBlur(img,(5,5),0,0)
    gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    # _,binary = cv2.threshold(gray ,130 ,255 ,cv2.THRESH_BINARY)
    _,binary = cv2.threshold(gray ,0 ,255 ,cv2.THRESH_BINARY + cv2.THRESH_OTSU) #cv2.THRESH_OTSU使用最小二乘法处理像素点
    binary = bit_not(binary)#变为黑底白字
    height,width = binary.shape

    # binary = binary[1:height-1]
    # print(" binary ",binary)
    y_histogram = np.sum(binary ,axis=0 )#np.sum(a, axis=0) ------->列求和
    # print(" y_histogram ", y_histogram)
    y_min = np.min(y_histogram)
    # print(" y_min ", y_min)#y_min  0
    y_average = np.sum(y_histogram)/y_histogram.shape[0]
    # print(" y_average ", y_average)#y_average  3076.575
    y_threshold = (y_min + y_average)/10  ##############需要修改
    print(" y_threshold ", y_threshold)#y_threshold  615.3149999999999

    wave_peaks = find_waves(y_threshold ,y_histogram)
    print(" wave_peaks ", wave_peaks)#wave_peaks  [(16, 35), (39, 59), (62, 108), (111, 131), (135, 157)]
    points=[]
    for wave in wave_peaks:
        x1,x2 = wave

        im = binary[:,x1:x2]
        x_histogram  = np.sum(im, axis=1)
        x_min = np.min(x_histogram)
        x_average = np.sum(x_histogram)/x_histogram.shape[0]
        x_threshold = (x_min + x_average)/2
        # print(x_histogram)
        for ind in range(1,len(x_histogram)):
            if x_histogram[ind] - x_histogram[ind-1] > x_threshold/2:
                m = ind
                break
        for ind in range(len(x_histogram)-2,-1,-1):
            if x_histogram[ind] - x_histogram[ind+1] > x_threshold/2:
                M = ind
                break
        y1,y2 = m,M
        if m > M:
            continue
        im = im[y1:y2,:]
        point = [(x1,y1),(x2,y2)]


        cv2.rectangle(img ,point[0] ,point[1] ,(0,255,0) ,2)
        # cv2.imshow(str(x1),im)
        # cv2.waitKey()


        if x1 < 3 or x2 > width-3:
            continue
        points.append(point)


    # cv2.imshow('img',img)
    # cv2.waitKey()
    cv2.destroyAllWindows()

    return points

def dilate_erode(binary):
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 4))
    kernel_dilate2 = cv2.getStructuringElement(cv2.MORPH_RECT,(4, 1))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 1))
    binary = cv2.dilate(binary ,kernel_dilate ,iterations = 1)
    return binary

#轻微仿射变换，向左倾斜
def Affine(img):
    height,width = img.shape[:2]
    p2 = np.float32([ [width-1,height-1], [width-6,0], [0,height-1] ])
    p1 = np.float32([ [width-1,height-1], [width-1,0], [0,height-1] ])
    M = cv2.getAffineTransform( p1, p2)
    dst = cv2.warpAffine( img, M, (width,height))
    return dst


def pred_nums(img):
    nums = '无'
    img = Affine(img) #仿射变换
    img = cv2.resize(img,(200,int(200/img.shape[1]*img.shape[0])))
    img = cv2.GaussianBlur(img,(5,5),0,0)
    gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)#转成灰度图像
    # _,binary = cv2.threshold(gray ,130 ,255 ,cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义矩形结构元素
    # dilated = cv2.dilate(gray, kernel)  # 膨胀图像
    # cv2.imshow('dilated', dilated)

    _,binary = cv2.threshold(gray ,0 ,255 ,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imshow('binary',binary)


    binary = bit_not(binary)
    points = cut(img)

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'A', 'C', 'E', 'F', 'H', 'L', 'P']
    model = LeNet()
    model.load_state_dict(torch.load('./model/net_020.pth'))
    print("加载模型成功")
    #img = Image.open('001.jpg').convert('RGB')  # 返回PIL类型数据

    #cv2_img = cv2.cvtColor(numpy.asarray(Img_img), cv2.COLOR_RGB2BGR)
    #pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    num_tup=[]
    average_len=20
    for point in points:
        x1,y1 = point[0]
        x2,y2 = point[1]

        imarray=[]
        if  (x2-x1)>2*average_len:
            print('x1-x2 ',x2-x1)
            im = binary[y1 - 2:y2 + 2, x1 - 2:(x1+(x2-x1)//2) ]
            imarray.append([im,y1 - 2,y2 + 2,x1 - 2,(x1+(x2-x1)//2)])
            im = binary[y1 - 2:y2 + 2, (x1+(x2-x1)//2):x2 + 2]
            imarray.append([im,y1 - 2,y2 + 2, (x1+(x2-x1)//2),x2 + 2])
        else:
            im = binary[y1-2:y2+2,x1-2:x2+2]
            imarray.append([im,y1-2,y2+2,x1-2,x2+2])


        for i in range(len(imarray)):
            im=imarray[i][0]
            # im = binary[y1-2:y2+2,x1-2:x2+2]
            if y1-2<0 or y2+2>binary.shape[0]-1 or x1-2<0 or x2+2>binary.shape[1]:
                im = binary[y1: y2, x1:x2]
            im = dilate_erode(im)
            roi = cv2.resize(im ,(28,28) ,interpolation=cv2.INTER_AREA)
            #roi = np.array([roi.reshape(28*28)/255])
            # cv2.imshow('roi', roi)
            # cv2.waitKey()
            out = roi.astype(np.uint8)  # python类型转换
            pil_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            imgpil = pil_img.convert('L')

            transform = transforms.Compose([transforms.ToTensor()])
            imgpil = transform(imgpil).unsqueeze(0)
            # img_ = img.to(device)
            outputs = model(imgpil)
            _, predicted = torch.max(outputs, 1)
            # print(predicted)
            print('识别结果为 :', classes[predicted[0]])
            cv2.rectangle(img, (imarray[i][3],imarray[i][1]), (imarray[i][4],imarray[i][2]), (0,0,255),2)  # filled
            cv2.putText(img, classes[predicted[0]], (imarray[i][3]+2,imarray[i][1]+2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0),2)

            # if ind==10 and (y2-y1)/(x2-x1)>1.5:ind=1
            # if ind==1 and (y2-y1)/(x2-x1)<1.5:ind=10
            num_tup.append(classes[predicted[0]])
        # if average_len<(x2 - x1):
        #     average_len = x2 - x1
        #     print(" average_len ", average_len)

    print(num_tup)
    if len(num_tup)>0:
        nums = ''.join([i for i in num_tup])
    else:
        nums='无'

    # print('实际结果为：',curr.split('_')[1][:-4])
    # print('预测结果为：',nums)
        

    # cv2.imshow('img' ,img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return nums


####以上是基于直方图投影的方法，确定数码管和小数点的位置
####以下是基于膨胀腐蚀的方法，确定数码管和小数点的位置
# 定义一个阈值函数，将数码管部分取出来，根据实际情况进行相应修改，找到最优参数
def thresholding_inv(image):
    # 定义膨胀核心，根据实际情况进行修改
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))  # 1代表横向膨胀，6代表纵向膨胀
    kernel_dilate2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    ## 腐蚀参数我已经注释掉，根据实际情况选择是否使用
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    ## 根据RGB图得到灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 灰度图二值化
    ret, bin = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # ret,bin = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # bin = cv2.medianBlur(bin, 3)
    ## 对灰度图进行腐蚀，主要是为了分离相近的小数点，如果足够清晰可以不使用腐蚀，我已注释掉
    bin = cv2.erode(bin, kernel_erode)
    ## 对灰度图进行膨胀

    bin = cv2.erode(bin, kernel_erode)
    bin = cv2.dilate(bin, kernel_dilate, iterations=1)
    # bin=cv2.dilate(bin,kernel_dilate2,iterations = 1)
    bin = cv2.erode(bin, kernel_erode, iterations=1)
    return bin

def cutbyexpand(im):

    ## 二值化处理
    im_th = thresholding_inv(im)
    # 显示图片
    cv2.imshow('im_th', im_th)
    cv2.waitKey()  # 显示1000ms
    Ksize = 3
    minVal = 20
    maxVal = 40
    L2g = True
    canny = cv2.Canny(im_th, minVal, maxVal, apertureSize=Ksize, L2gradient=False)
    cv2.imshow("canny ",canny)
    # Find contours in the image  寻找边界集合
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'A', 'C', 'E', 'F', 'H', 'L', 'P']
    model = LeNet()
    model.load_state_dict(torch.load('./model/net_020.pth'))
    print("加载模型成功")
    mm = {}
    average_len=0
    # for循环对每一个contour 进行预测和求解，并储存
    for rect in rects:
        # Draw the rectangles 得到数字区域 roi
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        cv2.imshow("img", im)
        cv2.waitKey()  # 显示1000ms
        # Make the rectangular region around the digit
        leng1 = int(rect[3])
        leng2 = int(rect[2])
        pt1 = int(rect[1])
        pt2 = int(rect[0])

        # 得到数字区域
        roi = im_th[pt1:pt1 + leng1, pt2:pt2 + leng2]
        # 尺寸缩放为模型尺寸
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        # 处理成一个向量，为了和模型输入一直
        out = roi.astype(np.uint8)  # python类型转换
        pil_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        img = pil_img.convert('L')

        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img).unsqueeze(0)
        # img_ = img.to(device)
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        # print(predicted)
        #print('识别结果为 :', classes[predicted[0]])

        mm[pt2] = classes[predicted[0]]

    # 最后的处理
    # 根据像素坐标，从左到右排序，得到数字的顺序
    print(mm)
    num_tup = sorted(mm.items(), key=lambda x: x[0])

    # 将数字列表连接为字符串
    num = (''.join([str(i[1]) for i in num_tup]))


    '''try:
        numn=float(num)
        print('图中数字为%s,数值大小为%s' %(num,numn))
    except:
        print('不好意思，目前不支持多个小数点的数值识别')
        print('图中数字为%s'% num)'''
    # # 显示图像
    # cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
    # cv2.imshow("Resulting Image with Rectangular ROIs", im)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return num







# 图像二值化处理
def imgThreshold(img):
    rosource, binary = cv2.threshold(img, 121, 255, cv2.THRESH_BINARY)
    return binary


# 1.先水平分割，再垂直分割
# 对图片进行垂直分割
def verticalCut(img, img_num):
    (x, y) = img.shape  # 返回的分别是矩阵的行数和列数，x是行数，y是列数
    pointCount = np.zeros(y, dtype=np.float32)  # 每列黑色的个数
    x_axes = np.arange(0, y)
    # i是列数，j是行数
    tempimg = img.copy()
    for i in range(0, y):
        for j in range(0, x):
            # if j<15:
            if (tempimg[j, i] == 0):
                pointCount[i] = pointCount[i] + 1
    figure = plt.figure(str(img_num))
    # for num in range(pointCount.size):
    #     pointCount[num]=pointCount[num]
    #     if(pointCount[num]<0):
    #         pointCount[num]=0
    plt.plot(x_axes, pointCount)
    start = []
    end=[]
    # 对照片进行分割

    for index in range(1, y - 1):
        # 上个为0当前不为0，即为开始
        if ((pointCount[index - 1] == 0) & (pointCount[index] != 0)):
            start.append(index)
        # 上个不为0当前为0，即为结束
        elif ((pointCount[index] != 0) & (pointCount[index + 1] == 0)):
            end.append(index)
    if end==[]:
        for i in range(len(start)-1):
            end.append(start[i + 1]-10)
        end.append(y)

    imgArr = []
    print(start)
    print(end)
    for idx in range(0, len(start)):
        tempimg = img[:, start[idx]:end[idx]]
        # cv2.imshow(str(img_num) + "_" + str(idx), tempimg)
        # cv2.imwrite(img_num + '_' + str(idx) + '.jpg', tempimg)
        imgArr.append(tempimg)
    return imgArr
    # cv2.waitKey()
    # plt.show()


# 对图片进行水平分割,返回的事照片数组
def horizontalCut(img):
    (x, y) = img.shape  # 返回的分别是矩阵的行数和列数，x是行数，y是列数
    pointCount = np.zeros(y, dtype=np.uint8)  # 每行黑色的个数
    x_axes = np.arange(0, y)
    for i in range(0, x):
        for j in range(0, y):
            if (img[i, j] == 0):
                pointCount[i] = pointCount[i] + 1
    plt.plot(x_axes, pointCount)
    start = []
    end = []
    # 对照片进行分割
    print(pointCount)
    for index in range(1, y):
        # 上个为0当前不为0，即为开始
        if ((pointCount[index] != 0) & (pointCount[index - 1] == 0)):
            start.append(index)
        # 上个不为0当前为0，即为结束
        elif ((pointCount[index] == 0) & (pointCount[index - 1] != 0)):
            end.append(index)
    print(start)
    print(end)
    img1 = img[start[0]:end[0], :]

    cv2.imshow(img1)
    cv2.waitKey()
    plt.show()
    return img1


# 输入的分别是原图模板和标签
def matchTemplate(src, matchSrc, label):
    binaryc = imgThreshold(src)
    result = cv2.matchTemplate(binaryc, matchSrc, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    tw, th = matchSrc.shape[:2]
    tl = (max_loc[0] + th + 2, max_loc[1] + tw + 2)
    cv2.rectangle(src, max_loc, tl, [0, 0, 0])
    cv2.putText(src, label, max_loc, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6,
                color=(240, 230, 0))
    cv2.imshow('001', src)

def test(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = imgThreshold(img)
    imgs=verticalCut(binary,'1')

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'A', 'C', 'E', 'F', 'H', 'L', 'P']
    model = LeNet()
    model.load_state_dict(torch.load('./model/net_020.pth'))
    print("加载模型成功")

    num_tup=[]
    for im in imgs:
        roi = cv2.resize(im ,(28,28) ,interpolation=cv2.INTER_AREA)
        cv2.imshow('roi',roi)
        cv2.waitKey()
        out = roi.astype(np.uint8)  # python类型转换
        pil_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        img = pil_img.convert('L')

        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img).unsqueeze(0)
        # img_ = img.to(device)
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        # print(predicted)
        #print('识别结果为 :', classes[predicted[0]])

        # if ind==10 and (y2-y1)/(x2-x1)>1.5:ind=1
        # if ind==1 and (y2-y1)/(x2-x1)<1.5:ind=10

        num_tup.append(classes[predicted[0]])

    nums = ''.join([i for i in num_tup])
    return nums
if __name__ == '__main__':
    #start = time.time()
    img = cv2.imread('./real/30.jpg')
    nums = pred_nums(img)
    print('预测结果为：',nums)
    # print ("加载模型并识别数字耗时：{}s".format(time.time() - start))
    # # 加载模型并识别数字耗时：0.2026069164276123s,约200ms

    # nums = cutbyexpand(img)
    # print('预测结果为：',nums)
    #nums = test(img)
    #print('预测结果为：',nums)



    