import subprocess
import cv2
rtmp_url = "rtmp://47.97.217.228:1935/live/302"
filename = "D:/DOC/enprictice/resources/data/meter1.mp4"
# In my mac webcamera is 0, also you can set a video file name instead, for example "/home/user/demo.mp4"
path = 0
cap = cv2.VideoCapture(filename)

# gather video info to ffmpeg
fps = 20
width = 500
height = 400

# command and params for ffmpeg
command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(width, height),
           '-r', str(fps),
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           rtmp_url]

# using subprocess and pipe to fetch frame data
p = subprocess.Popen(command, stdin=subprocess.PIPE)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("frame read failed")
        break

    # YOUR CODE FOR PROCESSING FRAME HERE

    # write to pipe
    p.stdin.write(frame.tobytes())