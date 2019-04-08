
from keras.layers import Conv1D, Input, Add, Activation, Dropout
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.initializers import TruncatedNormal
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras import optimizers




def DC_CNN_Block(nb_filter, filter_length, dilation, l2_layer_reg):
    def f(input_):
        
        residual =    input_
        
        layer_out =   Conv1D(filters=nb_filter, kernel_size=filter_length, 
                      dilation_rate=dilation, 
                      activation='linear', padding='causal', use_bias=False,
                      kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                      seed=42), kernel_regularizer=l2(l2_layer_reg))(input_)
                    
        layer_out =   Activation('selu')(layer_out)
        
        skip_out =    Conv1D(1,1, activation='linear', use_bias=False, 
                      kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                      seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)
        
        network_in =  Conv1D(1,1, activation='linear', use_bias=False, 
                      kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                      seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)
                      
        network_out = Add()([residual, network_in])
        
        return network_out, skip_out
    
    return f  
  

def make_model():
  input = Input(shape=(None,1))
    
  l1a, l1b = DC_CNN_Block(60,4,1,0.001)(input)    
  l2a, l2b = DC_CNN_Block(60,4,2,0.001)(l1a) 
  l3a, l3b = DC_CNN_Block(60,4,4,0.001)(l2a)
  l4a, l4b = DC_CNN_Block(60,4,8,0.001)(l3a)
  l5a, l5b = DC_CNN_Block(60,4,16,0.001)(l4a)
  l6a, l6b = DC_CNN_Block(60,4,32,0.001)(l5a)
  l6b = Dropout(0.8)(l6b) #dropout used to limit influence of earlier data
  l7a, l7b = DC_CNN_Block(60,2,64,0.001)(l6a)
  l7b = Dropout(0.8)(l7b) #dropout used to limit influence of earlier data

  l8 =   Add()([l1b, l2b, l3b, l4b, l5b, l6b, l7b])
    
  l9 =   Activation('relu')(l8)
           
  l21 =  Conv1D(1,1, activation='linear', use_bias=False, 
           kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
           kernel_regularizer=l2(0.001))(l9)

  model = Model(inputs=input, outputs=l21)
    
  adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, 
                           decay=0.0, amsgrad=False)
  
  model.compile(optimizer =adam, loss = "mae")
  return model