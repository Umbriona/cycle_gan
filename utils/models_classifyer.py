import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Activation, Embedding, Input, LeakyReLU, LayerNormalization, BatchNormalization, Softmax, Dropout, Concatenate
from utils.layers_new import SelfAttention, SelfAttentionSN, ResMod, GumbelSoftmax, ResModPreActSN
from utils import preprocessing as pre

class Classifier_reg1(Model):
    def __init__(self, config, name):
        super(Classifier_reg1, self).__init__()
        self.use_temporal_enc = config['use_global_pos'] 
        self.n_layers = config['n_layers']
        self.use_atte = config['use_attention']
        self.atte_loc = config['attention_loc']
        self.batch_size = config['batch_size']
        self.max_length = config['max_length']
        self.model_type = config['model_type']
        
        assert self.model_type in ['ResPreAct', "Res"], "model type {} not implemented needs to be [ResPreAct, Res]".format(self.model_type)  
        
        if self.use_temporal_enc:
            self.get_time()
            self.emb  = Embedding(config['max_length'], config['temporal_encode'])
            
        self.inp = Input((config['max_length'], config['vocab_size']), name = 'input_1')
        
        
        self.conv1 = Conv1D(config['filters'][0], 9, padding= 'same', activation = 'relu', name = 'conv1d')
        self.conv2 = Conv1D(config['filters'][config['n_layers']-1]*2, 9, padding= 'same', activation = 'relu', name = 'conv1d_1')
        
        self.res = []
        if self.model_type == "Res":
            self.res = [ResMod_old(config['filters'][i],
                           config['kernels'][i],
                           strides=config['strides'][i],
                           dilation = config['dilations'][i],
                           constrains=None,
                          # l1=config['l1'],
                          # l2=config['l2'],
                           ) for i in range(self.n_layers)]
        elif self.model_type == "ResPreAct":
        #    self.res = [ResModPreActSN(config['filters'][i],
        #                   config['kernels'][i],
        #                   strides=config['strides'][i],
        #                   dilation = config['dilations'][i],
        #                   constrains=None,
        #                   l1=config['l1'],
        #                   l2=config['l2'],
        #                   rate = config['rate']) for i in range(self.n_layers)]
            self.res.append(ResModPreActSN(config['filters'][0],
                                        config['kernels'][0],
                                        strides=config['strides'][0],
                                        dilation = config['dilations'][0],
                                        l1=config['l1'],l2=config['l2'],
                                        rate =config['rate']))
            self.res.append(ResModPreActSN(config['filters'][1],
                                        config['kernels'][1],
                                        strides=config['strides'][1],
                                        dilation = config['dilations'][1],
                                        l1=config['l1'],l2=config['l2'],
                                        rate =config['rate']))
            self.res.append(ResModPreActSN(config['filters'][2],
                                        config['kernels'][2],
                                        strides=config['strides'][2],
                                        dilation = config['dilations'][2],
                                        l1=config['l1'],l2=config['l2'],
                                        rate = config['rate']))
            self.res.append(ResModPreActSN(config['filters'][3],
                                        config['kernels'][3],
                                        strides=config['strides'][3],
                                        dilation = config['dilations'][3],
                                        l1=config['l1'],l2=config['l2'],
                                        rate = config['rate']))
            self.res.append(ResModPreActSN(config['filters'][4],
                                        config['kernels'][4],
                                        strides=config['strides'][4],
                                        dilation = config['dilations'][4],
                                        l1=config['l1'],l2=config['l2'],
                                        rate = config['rate']))
            self.res.append(ResModPreActSN(config['filters'][5],
                                        config['kernels'][5],
                                        strides=config['strides'][5],
                                        dilation = config['dilations'][5],
                                        l1=config['l1'],l2=config['l2'],
                                        rate = config['rate']))
            self.res.append(SelfAttentionSN(config['filters'][self.atte_loc]))
            self.res.append(ResModPreActSN(config['filters'][6],
                                        config['kernels'][6],
                                        strides=config['strides'][6],
                                        dilation = config['dilations'][6],
                                        l1=config['l1'],l2=config['l2'],
                                        rate = config['rate']))
            self.res.append(ResModPreActSN(config['filters'][7],
                                        config['kernels'][7],
                                        strides=config['strides'][7],
                                        dilation = config['dilations'][7],
                                        l1=config['l1'],l2=config['l2'],
                                        rate = config['rate']))
            self.res.append(ResModPreActSN(config['filters'][8],
                                        config['kernels'][8],
                                        strides=config['strides'][8],
                                        dilation = config['dilations'][8],
                                        l1=config['l1'],l2=config['l2'],
                                        rate = config['rate']))
            
        self.conv2 = Conv1D(config['filters'][config['n_layers']-1]*2, 9, padding= 'same', activation = 'relu', name = 'conv1d_1')
        self.flatten = tf.keras.layers.GlobalAveragePooling1D(name = 'global_average_pooling1d') #Flatten()
        self.dense = Dense(config['filters'][config['n_layers']-1], activation = 'relu', name = 'dense_4')
        self.dout = Dropout(0.3, name = 'dropout_10')
        self.out1 = Dense(1, activation = None, name = 'dense_5') 
        self.outa = Activation('linear', dtype='float32', name = 'activation')


        self.out = self.call(self.inp)                         
        # Reinitial
        #super(Classifier_reg1, self).__init__(
        #inputs=self.inp,
        #outputs=self.out, name=name)
        
    def get_time(self):
        temp = [tf.range(self.max_length, delta=1, dtype=tf.int32) for i in range(self.batch_size)] 
        self.temp = tf.stack(temp)

    #def build(self):
    #    # Initialize the graph
    #    self._is_graph_network = False
    #    self._init_graph_network(
    #        inputs=self.inp,
    #        outputs=self.out
    #    )
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'use_temporal_enc': self.use_temporal_enc,
            'n_layers': self.n_layers,
            'use_atte': self.use_atte,
            'atte_loc': self.atte_loc,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def call(self, x, training = True):        
        if self.use_temporal_enc:
            #masked_temp = tf.math.multiply(self.temp, x[1])
            temp_vec = self.emb(self.temp)
            x = tf.concat([temp_vec, x], axis = -1)
        x = self.conv1(x)

        for i in range(self.n_layers+1):
            if i == self.atte_loc+1 and self.use_atte:
                x, self.a_w = self.res[i](x)
            else:
                x= self.res[i](x)
            #if i == self.atte_loc+1 and self.use_atte:
            #    x, self.a_w = self.atte(x)
                
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dout(x, training = training)
        x = self.out1(x)
        return self.outa(x)
    
class Classifier_reg(Model):
    def __init__(self, config, name):
        super(Classifier_reg, self).__init__()
        self.use_temporal_enc = config['use_global_pos'] 
        self.n_layers = config['n_layers']
        self.use_atte = config['use_attention']
        self.atte_loc = config['attention_loc']
        self.batch_size = config['batch_size']
        self.max_length = config['max_length']
        self.model_type = config['model_type']
        
        assert self.model_type in ['ResPreAct', "Res"], "model type {} not implemented needs to be [ResPreAct, Res]".format(self.model_type)  
        
        if self.use_temporal_enc:
            self.get_time()
            self.emb  = Embedding(config['max_length'], config['temporal_encode'])
            
        self.inp = Input((config['max_length'], config['vocab_size']))
        
        
        self.conv1 = Conv1D(config['filters'][0], 9, padding= 'same', activation = 'relu')
        self.conv2 = Conv1D(config['filters'][config['n_layers']-1]*2, 9, padding= 'same', activation = 'relu')
        
        if self.model_type == "Res":
            self.res = [ResMod_old(config['filters'][i],
                           config['kernels'][i],
                           strides=config['strides'][i],
                           dilation = config['dilations'][i],
                           constrains=None,
                          # l1=config['l1'],
                          # l2=config['l2'],
                           ) for i in range(self.n_layers)]
        elif self.model_type == "ResPreAct":
            self.res = [ResModPreActSN(config['filters'][i],
                           config['kernels'][i],
                           strides=config['strides'][i],
                           dilation = config['dilations'][i],
                           constrains=None,
                           l1=config['l1'],
                           l2=config['l2'],
                           rate = config['rate']) for i in range(self.n_layers)]
        
        self.atte = SelfAttentionSN(config['filters'][self.atte_loc])    
        self.flatten = tf.keras.layers.GlobalAveragePooling1D() #Flatten()
        self.dense = Dense(config['filters'][config['n_layers']-1], activation = 'relu')
        self.out1 = Dense(1, activation = None) 
        self.outa = Activation('linear', dtype='float32')
        self.dout = Dropout(0.3)

        self.out = self.call(self.inp)                         
        # Reinitial
        super(Classifier_reg, self).__init__(
        inputs=self.inp,
        outputs=self.out, name=name)
        
    def get_time(self):
        temp = [tf.range(self.max_length, delta=1, dtype=tf.int32) for i in range(self.batch_size)] 
        self.temp = tf.stack(temp)

    def build(self):
    #    # Initialize the graph
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.inp,
            outputs=self.out
        )
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'use_temporal_enc': self.use_temporal_enc,
            'n_layers': self.n_layers,
            'use_atte': self.use_atte,
            'atte_loc': self.atte_loc,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def call(self, x, training = True):        
        if self.use_temporal_enc:
            #masked_temp = tf.math.multiply(self.temp, x[1])
            temp_vec = self.emb(self.temp)
            x = tf.concat([temp_vec, x], axis = -1)
        x = self.conv1(x)

        for i in range(self.n_layers):
            x= self.res[i](x)
            if i == self.atte_loc and self.use_atte:
                x, self.a_w = self.atte(x)
                
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dout(x, training = training)
        x = self.out1(x)
        return self.outa(x)
    
class Classifier_class(Model):
    def __init__(self, config):
        super(Classifier_class, self).__init__()
        
        self.use_temporal_enc = config['use_global_pos'] 
        self.n_layers = config['n_layers']
        self.use_atte = config['use_attention']
        self.atte_loc = config['attention_loc']
        self.batch_size = config['batch_size']
        self.max_length = config['max_length']
        self.model_type = config['model_type']
        
        assert self.model_type in ['ResPreAct', "Res"], "model type {} not implemented needs to be [ResPreAct, Res]".format(self.model_type)  
        
        if self.use_temporal_enc:
            self.get_time()
            self.emb  = Embedding(config['max_length'], config['temporal_encode'])
            
        self.inp = Input((config['max_length'], config['vocab_size']))
        self.conv1 = Conv1D(config['filters'][0], 6, padding= 'same', activation = 'relu')
        self.conv2 = Conv1D(config['filters'][config['n_layers']-1]*2, 6, padding= 'same', activation = 'relu')
        
        if self.model_type == "Res":
            self.res = [ResMod_old(config['filters'][i],
                           config['kernels'][i],
                           strides=config['strides'][i],
                           dilation = config['dilations'][i],
                           constrains=None,
                          # l1=config['l1'],
                          # l2=config['l2'],
                           ) for i in range(self.n_layers)]
        elif self.model_type == "ResPreAct":
            self.res = [ResModPreActSN(config['filters'][i],
                           config['kernels'][i],
                           strides=config['strides'][i],
                           dilation = config['dilations'][i],
                           constrains=None,
                           l1=config['l1'],
                           l2=config['l2'],
                           rate = config['rate']) for i in range(self.n_layers)]
        
        self.atte = SelfAttentionSN(config['filters'][self.atte_loc])    
        self.flatten = tf.keras.layers.GlobalAveragePooling1D() #Flatten()
        self.out1 = Dense(1, activation = "sigmoid") #, kernel_constraint=self.constraint)
        self.outa = Activation('sigmoid', dtype='float32', name='predictions')
        self.dout = Dropout(0.3)

        self.out = self.call(self.inp)                         
        # Reinitial
        super(Classifier_class, self).__init__(
        inputs=self.inp,
        outputs=self.out)
        
    def get_time(self):
        temp = [tf.range(self.max_length, delta=1, dtype=tf.int32) for i in range(self.batch_size)] 
        self.temp = tf.stack(temp)

    def build(self):
        # Initialize the graph
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.inp,
            outputs=self.out
        )
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'use_temporal_enc': self.use_temporal_enc,
            'n_layers': self.n_layers,
            'use_atte': self.use_atte,
            'atte_loc': self.atte_loc,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def call(self, x, training = True):        
        if self.use_temporal_enc:
            #masked_temp = tf.math.multiply(self.temp, x[1])
            temp_vec = self.emb(self.temp)
            x = tf.concat([temp_vec, x], axis = -1)
        x = self.conv1(x)

        for i in range(self.n_layers):
            x= self.res[i](x)
            if i == self.atte_loc and self.use_atte:
                x, self.a_w = self.atte(x, training=training)
                
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dout(x, training = training)
        x = self.out1(x)
        return self.outa(x)

