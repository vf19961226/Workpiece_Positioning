# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:47:03 2021

@author: vf199
"""
"""
Reference
[1] https://opencv-python-tutorials.readthedocs.io/zh/latest/7.%20%E7%9B%B8%E6%9C%BA%E6%A0%A1%E5%87%86%E5%92%8C3D%E9%87%8D%E5%BB%BA/7.1.%20%E7%9B%B8%E6%9C%BA%E6%A0%A1%E5%87%86/  <--取得相機參數並校正
[2] https://ithelp.ithome.com.tw/articles/10196167  <--Numpy檔案的儲存與讀取
"""

import numpy as np
import cv2 as cv
import glob

#擷取各個角度棋盤格的照片
'''相機影像截圖'''
i = 0
photo_num = 20 #要照幾張照片來辨識
n = 0 #使用第幾台相機
cap = cv.VideoCapture(n)
while(1):
    ret, frame = cap.read()
    if ret == True :
        cv.imshow("capture", frame)
        if cv.waitKey(1) & 0xFF == ord('q'): 
            cv.imwrite("./figure/" + str(n) + "/img" + str(i) + ".png", frame)
            i = i + 1
            if  i == photo_num:
                break

cap.release()
cv.destroyAllWindows()

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./figure/' + str(n) +'/*.png')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6,8), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (6,8), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 去畸变
img2 = cv.imread('./figure/' + str(n) +'/img0.png')
h,  w = img2.shape[:2]
newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h)) # 自由比例参数
# undistort
dst = cv.undistort(img2, mtx, dist, None, newcameramtx)

# crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult' + str(n) + '.png',dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

'''將結果寫入Numpy檔'''
np.savez('./data/camera_parameter' + str(n) + '.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs) 
