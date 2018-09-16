import cv2
import numpy as np

def cut_face(path):

    face_cascade = cv2.CascadeClassifier(r'G:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

    img = cv2.imread(path)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]

        # 直接返回第一个找到的人脸
        return roi_color
        # cv2.imshow('img',img)
        # cv2.imshow('cropped',roi_color)
        # cv2.waitKey(0)