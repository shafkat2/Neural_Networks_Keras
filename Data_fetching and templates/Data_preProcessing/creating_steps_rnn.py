##data preprocessing part 2

def data_preprocessing_train(train,test,timestamp,feature,parallel_process = 0):
  
  X_train =[]
  y_fet1 = []
  y_test = []
  
  for i in range(timestamp,test.shape[0]-parallel_process):
     y_fet1.append(train.iloc[i-timestamp:i,0:2].values)  
     y_test.append(test.iloc[i-timestamp,0])    
  y_fet1 ,y_test = np.array(y_fet1),np.array(y_test)
  X_train = np.reshape(y_fet1, (y_fet1.shape[0], y_fet1.shape[1], feature))
  print(X_train.shape)
  print(y_test.shape)
  return  X_train,y_test


def data_preprocessing_predict(data1,timestamp,feature):
  y_fet1 = []
  inputs = []
  for i in range(timestamp,data1.shape[0]+1,timestamp):
     y_fet1.append(data1.iloc[i-timestamp:i,0:1].values)
  y_fet1   = np.array(y_fet1)
  print(y_fet1.shape)
  inputs = np.reshape(y_fet1, (y_fet1.shape[0], y_fet1.shape[1], feature))
  print(inputs.shape)
  return  inputs

#data_preprocessing part 2
def pre_data_for_predict(to_predict,timestamp,features):
  
  predict = data_preprocessing_predict(minMaxScaler(to_predict),timestamp,features)
  return  predict


def making_train_test_for_ml(Train_set,Correct_train_set,features,Test_non_scaled_first_column,Test_non_scaled_predict,Test_scaled_for_prediction,timestamp,parallel_process1,parallel_process2):
  Train,Label = data_preprocessing_train(Train_set,Correct_train_set,timestamp,features,parallel_process1)
  Test_non_scaled = Test_non_scaled_first_column
  Test_non_scaled_predict = Test_non_scaled_predict
  Test_scaled = data_preprocessing_predict(minMaxScaler(Test_scaled_for_prediction),timestamp,features,parallel_process2)
  
  return  Train,Label,Test_non_scaled,Test_non_scaled_predict,Test_scaled

# prediction
def predict(model,non_scaled_pre,scaled_pre,batch_size):
  
  min_max_scaler.fit(non_scaled_pre)
  predict = model.predict(scaled_pre,batch_size)
  predict1 = predict.flatten()
  predict1 = np.array(predict1).reshape(-1, 1)
  predict1 = min_max_scaler.inverse_transform(predict1)
  plt.figure(1)
  plt.subplot(211)
  plt.plot(non_scaled_pre, label='Noisy')
  plt.subplot(212)
  plt.figure(figsize=(18,8))
  plt.plot(predict1, label='Denoised')
  plt.xlabel("datapoints")
  plt.ylabel("Fuel Information")
  plt.legend()
  predict1 = pd.DataFrame(predict1)
#   plt.savefig('/content/drive/My Drive/Machine_Learning/after_machine_learing.png')
  return predict1
  
def recurrent_pred(model,S,New_feature,timestamp,columns,batch_size):

  prediction = pre_data_for_predict(New_feature,timestamp,columns)
  Test_scaled  = S
  neural = predict(model,Test_scaled,prediction,batch_size)
  
  return neural

#evalutation
def evaluate(model,non_scaled_test,scaled_test,tobe_predicted,batch_size):
   
  min_max_scaler.fit(non_scaled_test)

  predict = model.predict(scaled_test,batch_size)
  predict1 = min_max_scaler.inverse_transform(predict)
  rmse = math.sqrt(mean_squared_error(tobe_predicted[0:], predict1[:-64]))
  return rmse  