from keras.models import Sequential
from keras.layers import MaxPooling2D

model = Sequential()
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid',  painput_shape=(100, 100, 15)))
model.summary()

"""
    pool_size: number specifying the height and width of the pooling window
"""