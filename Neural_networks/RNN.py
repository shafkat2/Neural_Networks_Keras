from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input,RepeatVector,TimeDistributed
from tensorflow.keras.layers import LSTM,Dropout,Bidirectional,GRU


  regressor = Sequential()
  regressor.add(LSTM(units = 3, return_sequences = True, batch_input_shape = (batch_size,X_tobe_trained.shape[1] ,X_tobe_trained.shape[2])))
  regressor.add(Bidirectional(LSTM(60,return_sequences=True,recurrent_dropout= 0.1)))
  regressor.add(Bidirectional(LSTM(60,return_sequences=True,recurrent_dropout= 0.1)))
  regressor.add(Bidirectional(LSTM(60)))
  
  regressor.add(Dense(units = 1,activation='tanh'))