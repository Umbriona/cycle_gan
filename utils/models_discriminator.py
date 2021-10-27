import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv1D, Dense, Flatten, Activation, Embedding, Input, LeakyReLU, LayerNormalization, BatchNormalization, Softmax, Dropout, Concatenate
from tensorflow.keras.regularizers import L1L2
from utils.layers_new import  SelfAttentionSN, SelfAttention, GumbelSoftmax, SpectralNormalization
from utils.layers_residual import ResModPreAct, ResModPreActSN
from utils import preprocessing as pre

class Linear(Layer):
    def __init__(self, obj, name):
        super(Linear, self).__init__(name=name)
        self.obj = obj
    def call(x, training=True):
        x = self.obj(x)
        return x
def first_layers_sn(name, l1, l2, patch, activation):
    projection = SpectralNormalization(Dense(64,
                                        activation = None,
                                        use_bias = False,
                                        kernel_regularizer = L1L2(l1=l1, l2=l2),
                                        name = name+"_proj"),
                                  name = name+"_proj_sn")
        
    input_conv = SpectralNormalization(Conv1D(64,
                                         9,
                                         padding = "same",
                                         activation = LeakyReLU(0.2),
                                         kernel_regularizer = L1L2(l1=l1, l2=l2),
                                         name = name+"_input_conv"),
                                  name = name+"_input_conv_sn")
    if patch:
            out = SpectralNormalization(Conv1D(1,
                                      9,
                                      padding= "same",
                                      activation = activation,
                                      kernel_regularizer = L1L2(l1=l1, l2=l2),
                                      name =name + "_conv_out"))
    else:
        out = SpectralNormalization(Dense(1,
                       activation= activation,
                       use_bias = False,
                        kernel_regularizer = L1L2(l1=l1, l2=l2),
                        name = name + "_dense_out"), name = name + "_dense_out_norm")
    return projection, input_conv, out

def first_layers(name, l1, l2, patch, activation):
    projection = Dense(64,
                                        activation = None,
                                        use_bias = False,
                                        kernel_regularizer = L1L2(l1=l1, l2=l2),
                                        name = name+"_proj")
        
    input_conv = Conv1D(64,
                                         9,
                                         padding = "same",
                                         activation = LeakyReLU(0.2),
                                         kernel_regularizer = L1L2(l1=l1, l2=l2),
                                         name = name+"_input_conv")
    if patch:
        out = Conv1D(1,
                                  9,
                                  padding= "same",
                                  activation = activation,
                                  kernel_regularizer = L1L2(l1=l1, l2=l2),
                                  name = name + "_conv_out")
    else:
        out = Dense(1,
                       activation= activation,
                       use_bias = False,
                        kernel_regularizer = L1L2(l1=l1, l2=l2),
                        name = name + "_dense_out")
    return projection, input_conv, out

class Discriminator(Model):
    def __init__(self, config, vocab, activation = 'sigmoid' , name = "dis"):
        super(Discriminator, self).__init__(name = name)
        self.conv=[]
        self.type = config["type"]
        self.n_layers = config['n_layers']
        self.patch = config["patch"]
        self.l1 = config['l1']
        self.l2 = config['l2']
        self.activation = activation
        
        if config['use_spectral_norm'] == True:
            self.projection, self.input_conv, self.out = first_layers_sn(self.name, self.l1, self.l2, self.patch, self.activation)
            res = ResModPreActSN
            att = SelfAttentionSN
        else:
            self.projection, self.input_conv, self.out = first_layers(self.name, self.l1, self.l2, self.patch, self.activation)
            res = ResModPreAct
            att = SelfAttention

        self.conv = [res(config['filters'][i],
                           config['kernels'][i],
                           strides=config['strides'][i],
                           dilation = config['dilations'][i],
                           l1=config['l1'],
                           l2=config['l2'],
                           rate = config['rate'],
                           norm=config['norm'],
                           name = self.name + "res_{}".format(i)) for i in range(self.n_layers)]  

            
        self.use_atte = config['use_attention']
        self.atte_loc = config['attention_loc'] 
        self.atte = att(config['filters'][self.atte_loc], name = self.name + "_attention")
        
        if not self.patch:
            self.flat = Flatten(name = self.name + "_flat")


    @classmethod
    def from_config(cls, config):
        return cls(**config)  
    
    def call(self, x, training= True):
        x = self.projection(x)
        x = self.input_conv(x)
        for i in range(self.n_layers):
            x = self.conv[i](x)
            if self.atte_loc == i and self.use_atte:
                x, self.a_w = self.atte(x, training=training)
        if not self.patch:
            x = self.flat(x)
        x = self.out(x)
        return x
    

def discriminator(config):
    
    
    return model