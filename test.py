import os
from keras.models import load_model
import cv2
import  numpy as np
from cut_face import cut_face

path1 = "H:\\face_detection\\face_detection\\liuxiang.jpg"
path2 = "H:\\face_detection\\face_detection\\wangziru.jpg"

def get_im_cv2(paths):
    imgs = []
    for path in paths:
        # print(path)
        # print(cv2.imread(path))
        img = cut_face(path)
        img = cv2.resize(img,(150,150))
        imgs.append(img/255.0)
        cv2.imshow('img',img)
        cv2.waitKey(0)
    return np.array(imgs).reshape(len(paths),150,150,3)

if os.path.exists("H:/face_detection/LeNet.h5"):
    model = load_model("H:/face_detection/LeNet.h5")

img1 = get_im_cv2([path1])
img2 = get_im_cv2([path2])

# img1,img2 = np.array([img1]),np.array([img2])
result = model.predict([img1,img2])
print(result)