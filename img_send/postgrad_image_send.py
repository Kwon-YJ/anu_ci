import socket
import random
import cv2
import numpy as np
import time
import os
import sys
import subprocess
import time
import datetime

sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

sock.connect(('220.69.240.236',9000))
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
prev_time = 0
FPS = 0
Total_Count = 0
Loss_Count = 0

# folder_path = '/home/pi/work/val2017'
folder_path = './val2017'
file_list = os.listdir(folder_path)


print(f'start time : {datetime.datetime.now()}')


while True:
    for i in range(len(file_list)):
        frame = cv2.imread(folder_path+'/'+file_list[i],cv2.IMREAD_COLOR)
        result, frame = cv2.imencode('.jpg', frame, encode_param)
        data = np.array(frame)
        stringData = data.tostring()
        sock.sendall((str(len(stringData))).encode().ljust(16) + stringData)

