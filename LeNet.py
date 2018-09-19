from keras.models import Model
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Input,BatchNormalization
from keras.layers import concatenate
import keras

# 搭建LeNet，用于提取脸部特征
input1 = Input(shape=(150,150,3))
conv1 = Conv2D(
        32,(5,5),strides=(1,1),input_shape=(150,150,3),padding='valid',
        activation='relu',kernel_initializer='uniform'
    )(input1)
# 因为relu会影响数据分布，所以常在relu运算后添加batchnormalization层
# batch1 = BatchNormalization()(input1)
maxpool1 = MaxPool2D(pool_size=(2,2))(conv1)
conv2 = Conv2D(
        64,(5,5),strides=(1,1),padding='valid',activation='relu',
        kernel_initializer='uniform'
    )(maxpool1)
maxpool2 = MaxPool2D(pool_size=(2,2))(conv2)
fla = Flatten()(maxpool2)
dense1 = Dense(
        100,activation='relu'
    )(fla)
dense2 = Dense(
        20,activation='relu'
    )(dense1)


# 另一个LeNet
_input1 = Input(shape=(150,150,3))
# _batch1 = BatchNormalization()(_input1)
_conv1 = Conv2D(
        32,(5,5),strides=(1,1),input_shape=(150,150,3),padding='valid',
        activation='relu',kernel_initializer='uniform'
    )(_input1)
_maxpool1 = MaxPool2D(pool_size=(2,2))(_conv1)
_conv2 = Conv2D(
        64,(5,5),strides=(1,1),padding='valid',activation='relu',
        kernel_initializer='uniform'
    )(_maxpool1)
_maxpool2 = MaxPool2D(pool_size=(2,2))(_conv2)
_fla = Flatten()(_maxpool2)
_dense1 = Dense(
        100,activation='relu'
    )(_fla)
_dense2 = Dense(
        20,activation='relu'
    )(_dense1)



# 将两个LeNet的输出合并，并在后面添加两个全连接层，用于计算相似度
merged = concatenate([dense2,_dense2])
out1 = Dense(10,activation='relu')(merged)
out2 = Dense(2,activation='softmax')(out1)

LeNet = Model(inputs=[input1,_input1],outputs=[out2])
LeNet.compile(
    optimizer= keras.optimizers.Adam(lr=0.001),
    loss='sparse_categorical_crossentropy',
)
