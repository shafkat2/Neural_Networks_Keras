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
                    self._buil_graph()
            else:
                tf.logging.info('Model using gpu.')
                self._build_graph()
        self._init_session()

