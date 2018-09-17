'''
先拿几章图片做实验
'''
from dataset import DataSet
# from model import model
from LeNet import LeNet
import numpy as np

print("Reading data")
data = DataSet()
images = data.data[:10000]
labels = data.labels[:10000]

del data

same_face = []
diff_face = []

for i in range(9999):
    # combine = np.concatenate((images[i], images[i + 1]), axis=2)

    if labels[i] == labels[i+1]:
        same_face.append([images[i]/255,images[i+1]/255])
    else:
        diff_face.append([images[i]/255,images[i+1]/255])

x_train1 = np.array([f[0] for f in same_face+diff_face])
x_train2 = np.array([f[1] for f in same_face+diff_face])
y_train = np.array([1 for i in same_face] + [0 for j in diff_face])


LeNet.fit([x_train1,x_train2],y_train,
          batch_size = 5,
          epochs=20)
