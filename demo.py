'''
先拿几章图片做实验
'''
from dataset import DataSet
from keras.preprocessing.image import ImageDataGenerator
# from model import model
from LeNet import LeNet
import numpy as np
import os
from keras.models import load_model
import cv2
from random import randint

def get_im_cv2(paths):
    '''
    从根据图片地址列表读取图片
    '''
    imgs = []
    for path in paths:
        imgs.append(cv2.imread(path)/255.0)
    return np.array(imgs).reshape(len(paths),150,150,3)

def get_train_batch(x,y,batch_size=10):
    '''
    重构generator
    '''
    while 1:
        idx = np.random.randint(0, len(y), batch_size)
        x1 = get_im_cv2(x[0][idx])
        x2 = get_im_cv2(x[1][idx])
        y_train = y[idx]
        yield [x1,x2],y_train


# 加载图片地址
data = DataSet()
images = data.data[:10000]
labels = data.labels[:10000]

same_face = []
diff_face = []

for i in range(9999):
    # 如果两张图片标签相同，则将两个图片作为相同组样本
    if labels[i] == labels[i+1]:
        same_face.append([images[i],images[i+1]])
    # 如果两张图片标签不同，则作为差异组样本
    else:
        diff_face.append([images[i],images[i+1]])

# 转化为numpy.ndarray，便于传入keras构造的神经网络进行计算
x_train1 = np.array([f[0] for f in same_face+diff_face])
x_train2 = np.array([f[1] for f in same_face+diff_face])
y_train = np.array([1 for i in same_face] + [0 for j in diff_face])

# 如果有模型文件存在，则导入之前的模型参数继续计算
if os.path.exists("H:/face_detection/LeNet.h5"):
    LeNet = load_model("H:/face_detection/LeNet.h5")
    print("model loaded")

# 使用重构后的generator进行训练
LeNet.fit_generator(generator = get_train_batch([x_train1,x_train2],y_train,batch_size=100),
          steps_per_epoch=100,epochs=5)

# 保存模型
LeNet.save("H:/face_detection/face_detection/LeNet.h5")
