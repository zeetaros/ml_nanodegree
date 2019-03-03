from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', 
    activation='relu', input_shape=(128, 128, 3)))
model.summary()

"""
    The number of parameters is 32*(3*3)*3+32 = 896
    Breakdown: 32 is the number of filters
               Each filter has a window size (square) of 3 by 3
               The input layer (/previous layer) have a depth of 3
               There is one bias term per layer, hence there are 32 biases
    Formula: filters * (kernel_size)^2 * input_shape[2] + filters = parameters in the convolutional layer

    N.B. the depth of the convolutional layer will always equal the number of filters

    If padding='same', then the spatial Dim of the convolutional layer are:
        @height = ceil(input_shape[0] / strides)
        @width = ceil(input_shape[1] / strides)

    If padding='valid', which means no padding, then the spatial Dim of the convolutional layer are:
        @height = ceil((input_shape[0] - kernel_size + 1) / strides)
        @width = ceil((input_shape[1] - kernel_size + 1) / strides)

"""

