# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:09:05 2021
@author: vf19961226
"""
import cv2
import numpy as np


import Image_Processing as ip

i = 0
L_cap = cv2.VideoCapture(0)

while(1):
    L_ret, L_frame = L_cap.read()
    if L_ret == True :
        cv2.imshow("L_capture", L_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #需改成自動控制儲存要辨識的照片
            cv2.imwrite("./figure/background.png", L_frame) #需裁減
            i = i + 1
        elif i == 1:
            break
L_cap.release()
cv2.destroyAllWindows()

###校正/畫圓###
n = 0
path = './data/camera_parameter_' + str(n) + '.npz'
mtx, dist = ip.npz_read(path) #影像校正
img = ip.img_correction(L_frame, mtx, dist)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
roi_image2 = ip.region_of_interest(img, vertices)

cv2.imshow('roi_image2', roi_image2)

circles = ip.hough(roi_image) #偵測定位工件上的圓
print(circles)
try:
    print(len(circles))
    
    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(L_frame,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(L_frame,(i[0],i[1]),2,(0,0,255),3)
except:
    print('Not any circle')

cv2.imshow('detected circles',L_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
