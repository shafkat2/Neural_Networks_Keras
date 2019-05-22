#importting libraies

import numpy as np
import tensorflow as tf

class ConVAE(object):
    #initializing all the parameter and the varialbe of the conVAE class 
    def __init__(self,z_size =32,batch_size = 1,learning_rate = 0.0001, kl_tolerance = 3, is_training =false, reuse = false , gpu_model = false ):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolearnce = kl_tolearnce
        self.is_training = is_training
        self.reuse = reuse
        with tf.variable_scope('conv_vae', reuse = self.reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using cpu.')
                    self._build_graph()
            else:
                tf.logging.info('Model using gpu.')
                self._build_graph()
        self._init_session()

    # making a method that creates the vae mode architecture iteself
    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.x = tf.placeholder(tf.float32,shape = [None,64,64,3])
            #building the encoder part
            h = tf.layers.conv2d(self.x,32,4,strides =2,activation = tf.nn.relu, name = "enc_conv1")
            h = tf.layers.conv2d(h,64,4,strides =2,activation = tf.nn.relu, name = "enc_conv2")
            h = tf.layers.conv2d(self.x,128,4,strides =2,activation = tf.nn.relu, name = "enc_conv3")
            h = tf.layers.conv2d(self.x,256,4,strides =2,activation = tf.nn.relu, name = "enc_conv4")
            h = tf.reshape(h,[-1,2*2*256])





