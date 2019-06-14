#importting libraies

import numpy as np
import tensorflow as tf

class ConVAE(object):
    #initializing all the parameter and the varialbe of the conVAE class 
    def __init__(self,z_size =32,batch_size = 1,learning_rate = 0.0001, kl_tolerance = 3, is_training =False, reuse = False , gpu_mode = False ):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
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
            #building the "V" part of the VAE
            self.mu = tf.layers.dense(h, self.z_size, name="enc_fc_mu")
            self.logvar = tf.layers.dense(h, self.z_size, name="enc_fc_logvar")
            self.sigma  = tf.exp(self.logvar/2.0)
            self.epsilon = tf.random_normal([self.batch_size,self.z_size])
            self.z = self.mu + self.sigma* self.epsilon
            #building the decoder part of the  VAE

            h = tf.layers.dense(self.z,1024,name = "dec_fc")
            h = tf.reshape(h,[-1,1,1,1024])
            h = tf.layers.conv2d_transpose(h,128,5,strides =2,activation = tf.nn.relu, name= "dec_decon1")
            h = tf.layers.conv2d_transpose(h,64,5,strides =2,activation = tf.nn.relu, name= "dec_decon2")
            h = tf.layers.conv2d_transpose(h,32,6,strides =2,activation = tf.nn.relu, name= "dec_decon3")
            self.y = tf.layers.conv2d_transpose(h,3,4,strides =2,activation = tf.sigmoid, name= "dec_decon4")
            # training operations
            if self.is_training:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.r_loss = tf.reduce_sum(tf.square(self.x - self.y), reduction_indices = [1,2,3])
                self.r_loss = tf.reduce_mean(self.r_loss)
                self.kl_loss = - 0.5 * tf.reduce_sum((1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)), reduction_indices = 1)
                self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size)
                self.kl_loss = tf.reduce_mean(self.kl_loss)
                self.loss = self.r_loss + self.kl_loss
                self.lr = tf.Variable(self.learning_rate, trainable=False)
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                grads = self.optimizer.compute_gradients(self.loss)
                self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step, name='train_step')
            self.init = tf.global_variables_initializer()




