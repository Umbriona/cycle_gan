import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Activation, Embedding, Input, LeakyReLU, LayerNormalization, BatchNormalization, Softmax, Dropout, Concatenate
from utils.layers_new import  SelfAttentionSN, GumbelSoftmax, SpectralNormalization
from utils.layers_residual import ResModPreAct
from utils import preprocessing as pre

class Discriminator(Model):
    def __init__(self, config, vocab, activation = 'sigmoid' , name = "dis"):
        super(Discriminator, self).__init__(name = name)
        self.conv=[]
        self.type = config["type"]
        self.n_layers = config['n_layers']
        self.patch = config["patch"]
        
        if self.type == "res":
            self.conv = [ResModPreAct(config['filters'][i],
                           config['kernels'][i],
                           strides=config['strides'][i],
                           dilation = config['dilations'][i],
                           l1=config['l1'],
                           l2=config['l2'],
                           rate = config['rate'],
                           spectral_norm=True,
                            name = self.name + "res_{}".format(i)) for i in range(self.n_layers)]  
        else:
            self.conv = [SpectralNormalization(Conv1D(config['filters'][i],
                        config['kernels'][i],
                        strides=config['strides'][i],
                        dilation_rate = config['dilations'][i],
                        activation = LeakyReLU(0.2),
                        name = self.name + "res_{}".format(i)), name = self.name + "_conv_{}_norm".format(i)) for i in range(self.n_layers)] 
            
        self.use_atte = config['use_attention']
        self.atte_loc = config['attention_loc'] 
        self.atte = SelfAttentionSN(config['filters'][self.atte_loc], name = self.name + "_attention")
        
        if self.patch:
            self.out = Conv1D(1, 9, padding= "same", activation = activation, name = self.name + "_conv_out")
        else:
            self.flat = Flatten(name = self.name + "_flat")
            self.out = SpectralNormalization(Dense(1,
                           activation= activation,
                           use_bias = False,
                            name = self.name + "_dense_out"), name = self.name + "_dense_out_norm")


    @classmethod
    def from_config(cls, config):
        return cls(**config)  
    
    def call(self, x, training= True):
        for i in range(self.n_layers):
            x = self.conv[i](x)
            if self.atte_loc == i and self.use_atte:
                x, self.a_w = self.atte(x, training=training)
        if not self.patch:
            x = self.flat(x)
        x = self.out(x)
        return x
    
