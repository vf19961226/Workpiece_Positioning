# -*- coding: utf-8 -*-
"""
Created on Mon May  3 22:19:48 2021

@author: vf19961226
"""
"""
Reference
[1] https://blog.csdn.net/hjxu2016/article/details/77833336
[2] https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/403752/
[3] https://gogoprivateryan.blogspot.com/2015/08/opencv-2-opencv-python.html
[4] https://zhuanlan.zhihu.com/p/38739563
[5] https://blog.csdn.net/u010128736/article/details/52875137  <--可參考這篇調整相機參數
[6] https://chtseng.wordpress.com/2016/12/05/opencv-contour%E8%BC%AA%E5%BB%93/  <--輪廓檢測
[7] https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html

"""


import cv2
import numpy as np
import math

def region_of_interest(img, vertices): #遮罩
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def img_correction (img, mtx, dist): #使用相機參數校正影像
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h)) # 自由比例参数
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return dst

def xml_read (filepath): #讀取XML檔中的相機參數
    try:
        import xml.etree.cElementTree as ET
        print("Using cElementTree")
    except ImportError:
        import xml.etree.ElementTree as ET
        print("Using ElementTree")
    
    # 從檔案載入 XML 資料
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    mtx = root[9][3].text #相機內部參數
    mtx = mtx.split()
    mtx = np.asarray(mtx)
    mtx = mtx.reshape((3,3))
    mtx = mtx.astype(float)
    print(mtx)
    
    dist = root[10][3].text
    dist = dist.split()
    dist = np.asarray(dist)
    dist = dist.reshape((1,5))
    dist = dist.astype(float)
    print(dist)

    return mtx, dist

def npz_read (filepath):
    npz = np.load(filepath)
    mtx = npz['mtx']
    dist = npz['dist']
    
    return mtx, dist

def image_processing (img): #影像取灰階、高斯模糊（濾波）、肯尼邊緣檢測
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(15, 15), 0)
    canny = cv2.Canny(blur_gray,150,50)
    return canny

def hough (img): #使用霍夫源檢測尋找圓心位置
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1 = 10, param2 = 5, minRadius = 5, maxRadius = 7) #參數已經調整成找到定位塊上三個圓型定位點
    #circles = np.uint16(np.around(circles))
    return circles

def get_pixel_long (obj_long, point1, point2):
    long = point2 - point1
    pixle_long = math.hypot(long[0], long[1])
    pixel_per_metric = obj_long/pixle_long
    return pixel_per_metric

def get_theta (slope):
    return math.atan(slope)
