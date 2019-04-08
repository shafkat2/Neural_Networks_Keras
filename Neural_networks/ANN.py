
import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim = 6, init = "uniform", activation = 'relu',input_dim = 11  ))
classifier.add(Dense(output_dim = 6, init = "uniform", activation = 'relu'  ))
classifier.add(Dense(output_dim = 1, init = "uniform", activation = 'sigmoid'  ))

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',merics = ['accuracy'])

classifier.fit(x,y,batch_size = 10, nb_epoch = 100) #x,y represent dependent and independent varuiable