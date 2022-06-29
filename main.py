# -*- coding: utf-8 -*-
from flask_cors import CORS
import base64
from flask import Flask, jsonify, request
app = Flask(__name__)
from regiongrow_api import detection
import time
import os
CORS(app, supports_credentials=True)#初始化的时候加载配置，这样就可以支持跨域访问
import subprocess
import requests

filename = "D:/DOC/enprictice/resources/data/meter1.mp4"
global CAP
global FIRST
FIRST =True
global selX
global selY
selX = 234
selY = 254
global CAP_PC
global FRAME
global FRAME_PC
global CLED
CLED ='绿色'
global LAST_FRAME
LAST_FRAME = None
global LAST_FRAME_PC
LAST_FRAME_PC =None
global P
global flag
import subprocess as sp
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image #用于读取数据
from model_lenet import LeNet
from led_nums import pred_nums,cutbyexpand


rtmpUrl = "rtmp://47.97.217.228:1935/live/302"
# 获取摄像头参数
# fps = 20
# width = 500
# height = 400


def detcstmodel():
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'A', 'C', 'E', 'F', 'H', 'L', 'P']
    model = LeNet()
    model.load_state_dict(torch.load('./model/net_020.pth'))
    img = Image.open('001.jpg').convert('RGB')  # 返回PIL类型数据
    img = img.convert('L')
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).unsqueeze(0)
    # img_ = img.to(device)
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    # print(predicted)
    print('识别结果为 :', classes[predicted[0]])


@app.route('/getMsg', methods=['GET', 'POST'])
def gen():
    global FIRST
    global CAP
    global CAP_PC
    global CLIGHT
    global selX
    global selY
    global LAST_FRAME
    global LAST_FRAME_PC
    global CLED
    CLED = '绿色'
    print('开始')
    global P
    global flag


    if request.method == 'POST':
        flag = request.form.get('preflag')
        return jsonify(flag)

    print(flag)
    if FIRST: #第一次请求
        print('开始')
        print('初始化')
        CAP = cv2.VideoCapture(filename)
        CAP_PC = cv2.VideoCapture(0)
        FIRST = False
        #向服务器传数据
        # fps = int(CAP.get(cv2.CAP_PROP_FPS))
        # width = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #
        # command = ['ffmpeg',
        #            '-y',
        #            '-f', 'rawvideo',
        #            '-vcodec', 'rawvideo',
        #            '-pix_fmt', 'bgr24',
        #            '-s', "{}x{}".format(width, height),
        #            '-r', str(fps),
        #            '-i', '-',
        #            '-c:v', 'libx264',
        #            '-pix_fmt', 'yuv420p',
        #            '-preset', 'ultrafast',
        #            '-f', 'flv',
        #            rtmpUrl]
        #P = subprocess.Popen(command, stdin=subprocess.PIPE)

    while True:


        success, FRAME = CAP.read()

        if not success:
            break
        success_pc, FRAME_PC = CAP_PC.read()

        if not success_pc:
            break

        print('读帧')

        #P.stdin.write(FRAME.tobytes())#向服务器推流
        if flag:
            print('检测')
            arr = [selX,selY]
            CLED = detection(FRAME, arr)#指示灯处理
            FRAME_PC = cv2.resize(FRAME_PC, (500, 400), interpolation=cv2.INTER_CUBIC)
            # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
            FRAME = cv2.resize(FRAME, (500, 400), interpolation=cv2.INTER_CUBIC)
            tmp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            day, times = tmp.split(' ')
            day = day.replace(':', '-')
            times = times.replace(':', '-')
            path = './img/' + day
            if not os.path.exists(path):
                os.makedirs(path)

            if LAST_FRAME is None:
                LAST_FRAME = FRAME.copy()
                continue

            # 计算当前帧和前帧的不同
            frameDelta = cv2.absdiff(LAST_FRAME, FRAME)

            # 当前帧设置为下一帧的前帧
            LAST_FRAME = FRAME.copy()
            LAST_FRAME_PC = FRAME_PC.copy()

            # 结果转为灰度图
            thresh = cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY)

            # 图像二值化
            thresh = cv2.threshold(thresh, 20, 255, cv2.THRESH_BINARY)[1]

            # 去除图像噪声,先腐蚀再膨胀(形态学开运算)
            thresh = cv2.erode(thresh, None, iterations=1)
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 遍历轮廓
            num=0
            for c in cnts:
                # 忽略小轮廓，排除误差

                if cv2.contourArea(c) < 100:
                    continue

                # 计算轮廓的边界框，在当前帧中画出该框
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(FRAME, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite(path + '/' + day + '-' + times + '_' + str(num) + '.jpg', FRAME)
                cv2.imwrite(path + '/' + day + '-' + times + '_' + str(num) + '_PC' + '.jpg', FRAME_PC)
                print('success')
                num += 1

        ret, jpeg = cv2.imencode('.jpg', FRAME)
        img_stream = base64.b64encode(jpeg).decode()

        ret, jpeg_pc = cv2.imencode('.jpg', FRAME_PC)
        img_stream_pc = base64.b64encode(jpeg_pc).decode()

        return jsonify({'img':img_stream, 'img_pc': img_stream_pc,'cled':CLED})


    ret, jpeg_g = cv2.imencode('.jpg', LAST_FRAME)
    img_stream= base64.b64encode(jpeg_g).decode()

    ret, jpeg_g = cv2.imencode('.jpg', LAST_FRAME_PC)
    img_stream_pc = base64.b64encode(jpeg_g).decode()
    return jsonify({'img':img_stream, 'img_pc': img_stream_pc,'cled':CLED})




@app.route('/close', methods=['GET','POST'])
def close():
    global CAP
    global FIRST
    global CAP_PC
    data = ''
    if request.method == 'POST':
        data = request.form.get('close')
        print(data)
        isClose = data
        if isClose:
            FIRST = True
            CAP.release()
            CAP_PC.release()
            print('摄像头已释放')

    return jsonify(data)


@app.route('/select', methods=['GET','POST'])
def select():

    if request.method == 'POST':
        selX = request.form.get('selX')
        print(selX)
        selY = request.form.get('selY')
        print(selY)


    return jsonify('ok')

@app.route('/detectChar', methods=['GET','POST'])
def detectChar():

    if request.method == 'POST':
        charimg = request.form.get('charimg')
        print("charimg")

        img_data = base64.b64decode(charimg)
        #print("img_data",img_data)
        with open('001.jpg', 'wb') as f:
            f.write(img_data)

    return jsonify('ok')

@app.route('/getresult', methods=['GET','POST'])
def getresult():
    img = cv2.imread('001.jpg')
    nums = pred_nums(img)
    #nums = cutbyexpand(img)
    print('预测结果为：', nums)
    return jsonify({'res':nums})

@app.route('/ocr' , methods=('GET', 'POST'))
def ocr():
    import json
    if request.method == 'POST':
        # print("request ",request)
        # img_up = request.form.get('file')
        # print("img_up ",img_up)
        #
        # compress = request.form.get('file')
        # print("compress ", compress)
        #
        # img = img_up.read()
        # import base64
        # img=base64.b64encode(img)
        charimg = request.form.get('charimg')
        print("charimg")
        url = 'http://192.168.1.132:8089/api/tr-run/'
        res = requests.post(url=url, data={'img': charimg})

        return res.text


# 启动运行
if __name__ == '__main__':
    app.run()   # 这样子会直接运行在本地服务器，也即是 localhost:5000
