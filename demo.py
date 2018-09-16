'''
先拿几章图片做实验
'''
from dataset import DataSet
from model import model
import numpy as np

print("Reading data")
data = DataSet()
images = data.data[:10000]
labels = data.labels[:10000]

same_face = []
diff_face = []

for i in range(9999):
    combine = np.concatenate((images[i], images[i + 1]), axis=2)
    if labels[i] == labels[i+1]:
        same_face.append(combine)
    else:
        diff_face.append(combine)

x_train = np.array(same_face + diff_face)
y_train = np.array([1 for i in same_face] + [0 for j in diff_face])


model.fit(x_train,y_train,
          batch_size = 10,
          epochs=5)
