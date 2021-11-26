import cv2
import threading
import os
import time

global timer

# 视频文件输入初始化
filename = "E:/USTC/2021/工程实践2021/面向远程仪表的自动读数与识别系统设计/检测数据/meter1.mp4"
videoSourceIndex1 = filename
videoSourceIndex2 = 1
cap = cv2.VideoCapture(videoSourceIndex1)
cap_pc = cv2.VideoCapture(videoSourceIndex2)

# 视频文件输出参数设置
'''out_fps = 12.0  # 输出文件的帧率 
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2') 
out1 = cv2.VideoWriter('v1.mp4', fourcc, out_fps, (500, 400)) 
out2 = cv2.VideoWriter('v2.mp4', fourcc, out_fps, (500, 400)) '''

'''读取本地视频并展示
while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break'''

'''拍照并保存
def shot_img():
    tmp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    day, times = tmp.split(' ')
    times = times.replace(':', '-')
    print(day, times)

    success, frame = cap.read()
    success, frame_pc = cap_pc.read()
    path = './img/' + day
    if not os.path.exists(path):
        os.mkdir(path)

    cv2.imwrite(path + '/' + times + '_' + str(num) + '.jpg', frame)

    print(num)

    num += 1
    timer = threading.Timer(shot_img)
    timer.start()'''

# 初始化当前帧的前帧
lastFrame = None

# no = 205
# 遍历视频的每一帧
# while no != 660:
global num
num = 0

while True:
    # 读取下一帧
    # (ret, frame) = camera.read()
    # 'E:/USTC/2021/gcsjData/430/' + str(no) + '.jpg'

    ret, frame = cap.read()
    frame = cv2.resize(frame, (500, 400))
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    tmp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    day, times = tmp.split(' ')
    times = times.replace(':', '-')

    path = 'E:/USTC/img/' + day
    if not os.path.exists(path):
        os.makedirs(path)
    # print(day, times)

    # success, frame = cap.read()
    # success, frame_pc = cap_pc.read()
    # frame = cv2.imread(path + '/' + times + '_' + str(num) + '.jpg')
    #    no += 1

    # 如果不能抓取到一帧，说明我们到了视频的结尾
    # if not ret:
    #    break
    # 调整该帧的大小
    # frame = cv2.resize(frame, (500, 400), interpolation=cv2.INTER_CUBIC)

    # 如果第一帧是None，对其进行初始化
    if lastFrame is None:
        lastFrame = frame
        continue

    # 计算当前帧和前帧的不同
    frameDelta = cv2.absdiff(lastFrame, frame)

    # 当前帧设置为下一帧的前帧
    lastFrame = frame.copy()

    # 结果转为灰度图
    thresh = cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY)

    # 图像二值化
    thresh = cv2.threshold(thresh, 20, 255, cv2.THRESH_BINARY)[1]

    # 去除图像噪声,先腐蚀再膨胀(形态学开运算)
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # 阀值图像上的轮廓位置
    # cv2.RETR_EXTERNAL表示轮廓的检索模式为：只监测外轮廓
    # cv2.CHAIN_APPROX_SIMPLE表示轮廓的近似办法为：压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标
    # cv2.findContours函数返回两个参数：轮廓本身，还有每条轮廓对应的属性
    # cnts表示轮廓的坐标,hierarchy表示各轮廓之间的关系
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历轮廓
    for c in cnts:
        # 忽略小轮廓，排除误差

        if cv2.contourArea(c) < 100:
            continue

        # 计算轮廓的边界框，在当前帧中画出该框
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(day, times)
        print(num)
        cv2.imwrite(path + '/' + times + '_' + str(num) + '.jpg', frame)
        print('success')
        num += 1

    # 显示当前帧
    # cv2.imshow("frame", frame)
    # cv2.imshow("frameDelta", frameDelta)
    # cv2.imshow("thresh", thresh)

    # 保存视频
    # out1.write(frame)
    # out2.write(frameDelta)

    # 如果q键被按下，跳出循环
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

    # 清理资源并关闭打开的窗口
# out1.release()
# out2.release()
# camera.release()
cv2.destroyAllWindows()