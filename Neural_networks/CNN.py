from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution2D(32,3,3, input_shape(64,64,3), activation = 'relu')) #makes 32 feature maps
classifier.add(MaxPooling(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128,activation = 'relu'))
classifier.add(Dense(output_dim = 1,activation = 'sigmoid'))


classifier.compile(optimizer = "adam", loss = 'binary_crossentropy',metric = ['accuracy'])