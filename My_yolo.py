# -*- coding: utf-8 -*-
"""
Created on Tue May 11 23:59:49 2021

@author: vf19961226
"""


import cv2
from TrtYOLO import TrtYOLO

net = cv2.dnn_DetectionModel('./data/My_yolov4.cfg','./data/My_yolov4_best.weights')
net.setInputSize(704, 704)
net.setInputScale(1.0/255) # input data normalize
net.setInputSwapRB(True)

with open('./data/obj.names') as f:
  names = f.read().rstrip('\n').split('\n')
  
trt_yolo = TrtYOLO('./data/My_yolov4.trt', len(names))

def identify (img):
    classes, confidences, boxes = net.detect(img, confThreshold=0.1, nmsThreshold = 0.4)
    return classes, confidences, boxes

def draw_box (img, classes, confidences, boxes):
    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
        label = '%.2f' %confidence
        label = '%s: %s' %(names[classId], label)
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        left, top, width, height = box

        cv2.rectangle(img, box, color=(0, 255, 0), thickness=3) # bounding box rectangle
        cv2.rectangle(img, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine),(255,255,255), cv2.FILLED) # text background
        cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
      
    return img

def trt_identify(img):
    boxes, confs, clss = trt_yolo.detect(img, 0.7) #0.7為信心閥值
    return clss, confs, boxes