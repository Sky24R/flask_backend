import  cv2
class camera(object):
    def __init__(self,filename):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(filename)  # 换成自己的视频文件
    def delete(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()  # 读视频
        return ret,frame
