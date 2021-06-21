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
from itertools import combinations 
import math

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
    
    blur_gray = cv2.GaussianBlur(img_gray,(29, 29), 0)
    canny = cv2.Canny(blur_gray,50,60)
    
    roi_image = ip.region_of_interest(canny, vertices) 

    contours, hierarchy = cv2.findContours(roi_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #尋找工件的輪廓
    
    total_cont = np.array(contours[0])
    for k in range(1,len(contours)):
        total_cont = np.vstack([total_cont, contours[k]])
    
    '''工件的外接矩形'''
    rect = cv2.minAreaRect(total_cont) #Rotated Rectangle
    w = rect[1][0] * pixel_per_metricX #工件的寬(? <--回傳雷雕機
    h = rect[1][1] * pixel_per_metricY #工件的高(? <--回傳雷雕機
    centroid, dimensions, angle = cv2.minAreaRect(cv2.boxPoints(rect))

    M = cv2.moments(total_cont) #尋找外接矩形的中點
    #cX = int(M["m10"] / M["m00"])
    #cY = int(M["m01"] / M["m00"])
    cX = int(centroid[0])
    cY = int(centroid[1])
    
    #Cpoint = [478,128.5] #Only for test
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
        slope = (y2 - y1)/((x2 - x1) + 0.0000000001)
        arc = ip.get_theta(slope) + theta #旋轉角度(世界座標)   <--回傳雷雕機
    
    return longX, longY, w, h, arc, cX, cY

'''Mqtt設定'''
client = initial("35.238.26.242", 1883) #需修改成Broker的IP位置
#client = initial("140.116.86.58", 1883) #需修改成Broker的IP位置
subscribe(client, ["Command"]) #訂閱主題
client.on_connect = on_connect #連上Broker時要做的動作
client.on_message = sub_messages #接到訂閱消息回傳時的動作
client.loop_start() #MQTT啟動

'''計算必要參數'''
n = 0 #左相機編號
img = cv2.imread('./figure/background.png') #匯入背景照片(左相機)
path = './data/camera_parameter_' + str(n) + '.npz'
img = img_correction(img, path) #影像校正

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #像素長度換算
blur_gray = cv2.GaussianBlur(gray,(3, 3), 0)
canny = cv2.Canny(blur_gray,150,50)

right_bottom = [img.shape[1] -145, img.shape[0]/2]
left_top = [img.shape[1]/2 +80 , img.shape[0]*0 +130]
right_top = [img.shape[1] -145, img.shape[0]*0 +130]
mid_left_bottom = [img.shape[1]/2 +80, img.shape[0]*0 +155]
mid = [img.shape[1]/2 +150, img.shape[0]*0 +155]
mid_right_bottom = [img.shape[1]/2 +150, img.shape[0]/2]
vertices = np.array([ mid_right_bottom, right_bottom, right_top, left_top, mid_left_bottom, mid], np.int32)
roi_image = ip.region_of_interest(canny, vertices) #L形遮罩，方便辨識定位工件上的圓

circles = ip.hough(roi_image) #偵測定位工件上的圓

really = 50 #世界座標中兩圓點間的距離為50mm(定位工件)
    
index = list(range(len(circles[0])))
index = list(combinations(index, 3))
for i in index:
    abs_slope = []
    ptp_long = [] #計算每兩點間的距離
    slope = [] #計算每兩點間的斜率
    ax = int(circles[0][i[0]][0]) - int(circles[0][i[1]][0])
    ay = int(circles[0][i[0]][1]) - int(circles[0][i[1]][1])
    bx = int(circles[0][i[2]][0]) - int(circles[0][i[1]][0])
    by = int(circles[0][i[2]][1]) - int(circles[0][i[1]][1])
    cx = int(circles[0][i[0]][0]) - int(circles[0][i[2]][0])
    cy = int(circles[0][i[0]][1]) - int(circles[0][i[2]][1])
    ptp_long.append(math.hypot(ax,ay))
    ptp_long.append(math.hypot(bx,by))
    ptp_long.append(math.hypot(cx,cy))
    if ay == 0:
        ay += 0.000000000001
    elif by == 0:
        by += 0.000000000001
    elif cy == 0:
        cy += 0.000000000001
    slope.append(ax/ay)
    slope.append(bx/by)
    slope.append(cx/cy)
    abs_slope.append(abs(ax/ay))
    abs_slope.append(abs(bx/by))
    abs_slope.append(abs(cx/cy))
    min_slope = abs_slope.index(min(abs_slope)) #斜率絕對值最小為水平(X)
    max_slope = abs_slope.index(max(abs_slope)) #斜率絕對值最大維垂直(Y)
    X_long = ptp_long[min_slope]
    Y_long = ptp_long[max_slope]
    
    for j in range(len(ptp_long)): #尋找斜邊
        ptp_long2 = ptp_long.copy()
        ptp_long2.pop(j)
        hypotenuse = math.hypot(ptp_long2[0], ptp_long2[1])
        if ptp_long[j]/hypotenuse >= 0.8 and ptp_long[j]/hypotenuse <= 1.2:
            if X_long/Y_long >= 0.8 and X_long/Y_long <= 1.2: #兩邊距離需在一定區間內
                break
    
    if ptp_long[j]/hypotenuse >= 0.8 and ptp_long[j]/hypotenuse <= 1.2:
        if X_long/Y_long >= 0.8 and X_long/Y_long <= 1.2: #兩邊距離需在一定區間內
            break
else:
    print("ERROR")
    

if min_slope == 0 and max_slope == 1:
    Xpoint = circles[0][i[0]]
    Ypoint = circles[0][i[2]]
    Cpoint = circles[0][i[1]]
elif min_slope == 0 and max_slope == 2:
    Xpoint = circles[0][i[1]]
    Ypoint = circles[0][i[2]]
    Cpoint = circles[0][i[0]]
elif min_slope == 1 and max_slope == 2:
    Xpoint = circles[0][i[1]]
    Ypoint = circles[0][i[0]]
    Cpoint = circles[0][i[2]]
elif min_slope == 1 and max_slope == 0:
    Xpoint = circles[0][i[2]]
    Ypoint = circles[0][i[0]]
    Cpoint = circles[0][i[1]]
elif min_slope == 2 and max_slope == 1:
    Xpoint = circles[0][i[0]]
    Ypoint = circles[0][i[1]]
    Cpoint = circles[0][i[2]]
elif min_slope == 2 and max_slope == 0:
    Xpoint = circles[0][i[2]]
    Ypoint = circles[0][i[1]]
    Cpoint = circles[0][i[0]]

theta = ip.get_theta(slope[min_slope]) #取得像機座標與實際座標的旋轉角度

pixel_per_metricX = ip.get_pixel_long(really, Xpoint, Cpoint) #計算水平(X)的每像素公制距離
pixel_per_metricY = ip.get_pixel_long(really, Ypoint, Cpoint) #計算垂直(Y)的每像素公制距離
print(Cpoint)
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
    #L_frame = cv2.imread("./figure/1.png") #Only for test
    #L_img = ip.img_correction(L_frame, L_mtx, L_dist)
    path = './data/camera_parameter_' + str(n) + '.npz'
    L_img = img_correction(L_frame, path)
    L_gray = cv2.cvtColor(L_img, cv2.COLOR_BGR2GRAY)
   
    '''Yolov4工件辨識'''
    
    classes, confidences, boxes = My_yolo.identify(L_img)
    #print(boxes)
    t3 = time.time()
    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
        thisClass = classId
        longX, longY, w, h, arc, cX, cY = positioning (L_gray, box)
    
        '''MQTT傳送結果至雷雕機'''
        output = str(thisClass) + "," + str(longX) + "," + str(longY) + "," + str(w) + "," + str(h) +  "," + str(arc) #發布指令[工件種類, 以定位工件中點為原點的工件中心點X座標, 以定位工件中點為原點的工件中心點Y座標, 工件外接矩形寬, 工件外接矩形高, 工件距離攝影機深度, 弓箭旋轉角度(弧度)]
        publish(client, "Feedback", output) #發布指令
        sub_return = ['Nothing','Nothing']
        t4 = time.time()
        #print("影像擷取執行時間為：%f 秒" % (t2 - t1))
        #print('YOLOv4執行時間為： %f 秒' % (t3 - t2))
        #print('工件定位執行時間為： %f 秒' % (t4 - t3))
        #print('全部執行時間為： %f 秒' % (t4 - t1))
        #print(output)
    #cv2.circle(L_frame,(cX,cY),2,(0,0,255),3)
    #L_frame = My_yolo.draw_box(L_frame, classes, confidences, boxes)
    #cv2.imshow("1", L_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
client.loop_stop() #MQTT停止
