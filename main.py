# -*- coding: utf-8 -*-
"""
Created on Thu May 13 22:09:58 2021

@author: vf19961226
"""
"""
Reference
[1] https://ithelp.ithome.com.tw/articles/10227131
[2] https://sites.google.com/site/ezpythoncolorcourse/globalvariablelocalvariable
[3] https://www.itread01.com/p/1426490.html
[?] https://github.com/NVIDIA-AI-IOT/yolov4_deepstream/tree/master/tensorrt_yolov4
[?] https://automaticaddison.com/how-to-set-up-the-nvidia-jetson-nano-developer-kit/
[?] https://automaticaddison.com/how-to-install-opencv-4-5-on-nvidia-jetson-nano/
[?]* http://server.zhiding.cn/server/2021/0426/3133640.shtml

[4] https://www.rs-online.com/designspark/nvidia-jetson-nanotensor-rtyolov4-cn
[5] https://github.com/jkjung-avt/tensorrt_demos
"""

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import time

import pycuda.autoinit  # This is needed for initializing CUDA driver


import Image_Processing as ip
import My_yolo

sub_topic = []
sub_return = ['Nothing','Nothing']

def initial (broker_IP, port):
    client = mqtt.Client()
    client.connect(broker_IP, port)
    
    return client

def publish (client, topic, messages):
    client.publish(topic, messages, qos = 1)

def sub_messages (client, userdata, message):
    topic = message.topic
    msg = message.payload.decode('utf-8')
    global sub_return
    sub_return = [topic, msg]
    print(sub_return)

 
def subscribe (client, topic): 
    global sub_topic
    sub_topic = topic
    for i in sub_topic:
        client.subscribe(i)

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    global sub_topic

    for i in sub_topic:
        client.subscribe(i)
        
def img_correction (img, npz_path):
    mtx, dist = ip.npz_read(npz_path)
    img = ip.img_correction(img, mtx, dist)
    return img

def positioning (img_gray, box):
    left, top, width, height = box
    left_top = [left , top]
    left_bottom = [left, top + height]
    right_bottom = [left + width, top + height]
    right_top = [left + width, top]
    vertices = np.array([ left_top, left_bottom, right_bottom, right_top], np.int32)
    
    blur_gray = cv2.GaussianBlur(img_gray,(21, 21), 0)
    canny = cv2.Canny(blur_gray,150,50)
    
    roi_image = ip.region_of_interest(canny, vertices) 

    contours, hierarchy = cv2.findContours(roi_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #尋找工件的輪廓
    
    total_cont = np.array(contours[0])
    for k in range(1,len(contours)):
        total_cont = np.vstack([total_cont, contours[k]])
    
    '''工件的外接矩形'''
    rect = cv2.minAreaRect(total_cont) #Rotated Rectangle
    w = rect[1][0] * pixel_per_metricX #工件的寬(? <--回傳雷雕機
    h = rect[1][1] * pixel_per_metricY #工件的高(? <--回傳雷雕機

    M = cv2.moments(total_cont) #尋找外接矩形的中點
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    Cpoint = [478,128.5] #Only for test
    cXpoint1 = np.array([cX, 0])
    cXpoint2 = np.array([Cpoint[0], 0])
    #longX = ip.get_pixel_long(really, cXpoint1, cXpoint2) #工件相對於定位工件的X軸座標  <--回傳雷雕機
    longX = abs(cX - Cpoint[0]) * pixel_per_metricX 
    cYpoint1 = np.array([cY, 0])
    cYpoint2 = np.array([Cpoint[1], 0])
    #longY = ip.get_pixel_long(really, cYpoint1, cYpoint2) #工件相對於定位工件的Y軸座標  <--回傳雷雕機
    longY = abs(cY - Cpoint[1]) * pixel_per_metricY
    
    '''計算工件旋轉角度'''
     #霍夫直線檢測參數設定
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 5 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    
    lines = cv2.HoughLinesP(roi_image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    location = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            location.append(y1+y2)
        
    low = location.index(min(location))
    for x1, y1, x2, y2 in lines[low]:
        slope = (y2 - y1)/(x2 - x1)
        arc = ip.get_theta(slope) + theta #旋轉角度(世界座標)   <--回傳雷雕機
    
    return longX, longY, w, h, arc

'''Mqtt設定'''
client = initial("35.238.26.242", 1883) #需修改成Broker的IP位置
#client = initial("140.116.86.58", 1883) #需修改成Broker的IP位置
subscribe(client, ["Command"]) #訂閱主題
client.on_connect = on_connect #連上Broker時要做的動作
client.on_message = sub_messages #接到訂閱消息回傳時的動作
client.loop_start() #MQTT啟動

'''計算必要參數'''
n = 0 #左相機編號
img = cv2.imread('./figure/c_L_img0.jpg') #匯入背景照片(左相機)
path = './data/camera_parameter_' + str(n) + '.npz'
img = img_correction(img, path) #影像校正

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #像素長度換算
blur_gray = cv2.GaussianBlur(gray,(3, 3), 0)
canny = cv2.Canny(blur_gray,150,50)

right_bottom = [img.shape[1] -150, img.shape[0]/2 -20]
left_top = [img.shape[1]/2 +75 , img.shape[0]*0 +110]
right_top = [img.shape[1] -150, img.shape[0]*0 +110]
mid_left_bottom = [img.shape[1]/2 +75, img.shape[0]*0 +135]
mid = [img.shape[1]/2 +145, img.shape[0]*0 +135]
mid_right_bottom = [img.shape[1]/2 +145, img.shape[0]/2 -20]
vertices = np.array([ mid_right_bottom, right_bottom, right_top, left_top, mid_left_bottom, mid], np.int32)
roi_image = ip.region_of_interest(canny, vertices) #L形遮罩，方便辨識定位工件上的圓

circles = ip.hough(roi_image) #偵測定位工件上的圓

really = 50 #世界座標中兩圓點間的距離為50mm(定位工件)
    
slope = [] #計算每兩點間的斜率
abs_slope = []
for j in range(len(circles[0])-1):
    x = circles[0][j][0] - circles[0][j+1][0]
    y = circles[0][j][1] - circles[0][j+1][1]
    slope.append(x/y)
    abs_slope.append(abs(x/y))
else:
    l = len(circles[0])-1
    x = circles[0][l][0] - circles[0][0][0]
    y = circles[0][l][1] - circles[0][0][1]
    slope.append(x/y)
    abs_slope.append(abs(x/y))

min_slope = abs_slope.index(min(abs_slope)) #斜率絕對值最小為水平(X)
max_slope = abs_slope.index(max(abs_slope)) #斜率絕對值最大維垂直(Y)
theta = ip.get_theta(slope[min_slope]) #取得像機座標與實際座標的旋轉角度
if min_slope == l:
    Xpoint1 = circles[0][l]
    Xpoint2 = circles[0][0]
else:
    Xpoint1 = circles[0][min_slope]
    Xpoint2 = circles[0][min_slope+1]
pixel_per_metricX = ip.get_pixel_long(really, Xpoint1, Xpoint2) #計算水平(X)的每像素公制距離
print(Xpoint1)
print(Xpoint2)

if max_slope == l:
    Ypoint1 = circles[0][l]
    Ypoint2 = circles[0][0]
else:
    Ypoint1 = circles[0][max_slope]
    Ypoint2 = circles[0][max_slope+1]
pixel_per_metricY = ip.get_pixel_long(really, Ypoint1, Ypoint2) #計算垂直(Y)的每像素公制距離
print(Ypoint1)
print(Ypoint2)
'''
if Xpoint1[0] == Ypoint1[0] and Xpoint1[1] == Ypoint1[1]: #尋找定位工件的中點(超過三個圓的時候可能會報錯)
    Cpoint = Xpoint1
elif Xpoint1[0] == Ypoint2[0] and Xpoint1[1] == Ypoint2[1]:
    Cpoint = Xpoint1
elif Xpoint2[0] == Ypoint1[0] and Xpoint2[1] == Ypoint1[1]:
    Cpoint = Xpoint2
elif Xpoint2[0] == Ypoint1[0] and Xpoint2[1] == Ypoint1[1]:
    Cpoint = Xpoint2
'''

while 1:
    '''相機影像截圖'''
    
    L_cap = cv2.VideoCapture(n)
    while(1):
        t1 = time.time()
        L_ret, L_frame = L_cap.read()
        if L_ret == True:
            cv2.imshow("L_capture", L_frame)
            if sub_return[1] == "OK": 
                #cv2.imwrite("L_img.jpg", L_frame) 
                break
            elif sub_return[1] == "END":
                break
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.waitKey(1)
    L_cap.release()
    cv2.destroyAllWindows()
    
    if sub_return[1] == "END":
        break
    t2 = time.time()
    '''左相機影像校正'''
    #L_mtx, L_dist = ip.npz_read('./data/camera_parameter_' + str(n) + '.npz')
    #img = cv2.imread("L_img.jpg")
    #L_frame = cv2.imread("./figure/obj_L_img/obj_L_img29.jpg") #Only for test
    #L_img = ip.img_correction(L_frame, L_mtx, L_dist)
    path = './data/camera_parameter_' + str(n) + '.npz'
    L_img = img_correction(L_frame, path)
    L_gray = cv2.cvtColor(L_img, cv2.COLOR_BGR2GRAY)
   
    '''Yolov4工件辨識'''
    
    classes, confidences, boxes = My_yolo.trt_identify(L_img)
    t3 = time.time()
    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
        thisClass = classId
        longX, longY, w, h, arc = positioning (L_gray, box)
        
        '''MQTT傳送結果至雷雕機'''
        output = str(thisClass) + "," + str(longX) + "," + str(longY) + "," + str(w) + "," + str(h) +  "," + str(arc) #發布指令[工件種類, 以定位工件中點為原點的工件中心點X座標, 以定位工件中點為原點的工件中心點Y座標, 工件外接矩形寬, 工件外接矩形高, 工件距離攝影機深度, 弓箭旋轉角度(弧度)]
        publish(client, "Feedback", output) #發布指令
        sub_return = ['Nothing','Nothing']
        t4 = time.time()
        print("影像擷取執行時間為：%f 秒" % (t2 - t1))
        print('YOLOv4執行時間為： %f 秒' % (t3 - t2))
        print('工件定位執行時間為： %f 秒' % (t4 - t3))
        print('全部執行時間為： %f 秒' % (t4 - t1))
        print(output)
client.loop_stop() #MQTT停止
