from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
# since using convolutional layer as the first layer, input_shape needs to be specified
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
# the final layer in the network should be a Dense layer with a softmax activation function
# to turn the result into a probability
# the number of nodes in the final layer should equal the total number of classes in the dataset
model.add(Dense(10, activation='softmax'))

"""
The network begins with a sequence of three convolutional layers, followed by max pooling layers.
These first six layers are designed to take the input array of image pixels and convert it to an 
array where all of the spatial information has been squeezed out, and only information encoding 
the content of the image remains. The array is then flattened to a vector in the seventh layer 
of the CNN. It is followed by two dense layers designed to further elucidate the content of the 
image. The final layer has one entry for each object class in the dataset, and has a softmax 
activation function, so that it returns probabilities.

"""

