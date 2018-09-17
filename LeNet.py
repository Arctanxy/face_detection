from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Input
from keras.layers import merge,concatenate
from keras.utils.vis_utils import plot_model

#
# model = Sequential()
# model.add(
#     Conv2D(
#         32,(5,5),strides=(1,1),input_shape=(250,250,6),padding='valid',
#         activation='relu',kernel_initializer='uniform'
#     )
# )
#
# model.add(
#     MaxPool2D(pool_size=(2,2))
# )
#
# model.add(
#     Conv2D(
#         64,(5,5),strides=(1,1),padding='valid',activation='relu',
#         kernel_initializer='uniform'
#     )
# )
#
# model.add(
#     MaxPool2D(pool_size=(2,2))
# )
#
# model.add(
#     Flatten()
# )
#
# model.add(
#     Dense(
#         100,activation='relu'
#     )
# )
#
# model.add(
#     Dense(
#         10,activation='sigmoid'
#     )
# )

input1 = Input(shape=(250,250,3))
conv1 = Conv2D(
        32,(5,5),strides=(1,1),input_shape=(250,250,6),padding='valid',
        activation='relu',kernel_initializer='uniform'
    )(input1)
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
        10,activation='relu'
    )(dense1)



_input1 = Input(shape=(250,250,3))
_conv1 = Conv2D(
        32,(5,5),strides=(1,1),input_shape=(250,250,6),padding='valid',
        activation='relu',kernel_initializer='uniform'
    )(input1)
_maxpool1 = MaxPool2D(pool_size=(2,2))(conv1)
_conv2 = Conv2D(
        64,(5,5),strides=(1,1),padding='valid',activation='relu',
        kernel_initializer='uniform'
    )(maxpool1)
_maxpool2 = MaxPool2D(pool_size=(2,2))(conv2)
_fla = Flatten()(maxpool2)
_dense1 = Dense(
        100,activation='relu'
    )(fla)
_dense2 = Dense(
        10,activation='relu'
    )(dense1)




merged = concatenate([dense2,_dense2])
out = Dense(1,activation='softmax')(merged)

LeNet = Model(inputs=[input1,_input1],outputs=[out])
LeNet.compile(
    optimizer='adam',
    loss='mean_squared_error',
)

# plot_model(LeNet)