'''
建立两个conv network，然后将两个network的结果输入dicision network中
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Flatten

model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.MSE,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
