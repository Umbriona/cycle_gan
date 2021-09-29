import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D,Conv1DTranspose, Dense, Flatten, Activation, Embedding, Input, LeakyReLU, LayerNormalization, BatchNormalization, Softmax, Dropout, Concatenate
from utils.layers_new import SelfAttention, SelfAttentionSN, GumbelSoftmax
from utils.layers_residual import ResModPreAct
from utils import preprocessing as pre

ATTENTION_FEATURES = 512

class Generator_res(Model):
    def __init__(self, config, vocab, name):
        super(Generator_res, self).__init__(name = name)

        self.n_layers = config['n_layers']
        self.down_sample = config['down_sample']
        
        assert len(config['filters']) >= self.n_layers, "not enough filters specified"
        assert len(config['kernels']) >= self.n_layers, "not enough kernels specified"
        assert len(config['dilations']) >= self.n_layers, "not enough dilations specified"
        
        # Set up down sample
        self.conv_down = [Conv1D(config['filters'][0], 9,
                                 strides=2,
                                 padding="same",
                                 activation = "relu",
                                 name = self.name + "_down_{}".format(i)) for i in range(self.down_sample)]
        
        # Set up res
        self.res = [ResModPreAct(config['filters'][i],
                           config['kernels'][i],
                           strides=1,
                           dilation = config['dilations'][i],
                           l1=config['l1'],
                           l2=config['l2'],
                           rate = config['rate'],
                           spectral_norm=True,
                            name = self.name + "_res_{}".format(i)) for i in range(self.n_layers)]
        
        # Set up Attention
        self.atte_loc = config['attention_loc']
        self.use_atte = config['use_attention']
        self.atte = SelfAttentionSN(config['filters'][self.atte_loc], name = self.name + "_attention")

        # Set output distribution (Activation)
        self.use_gumbel = config['use_gumbel']
        if self.use_gumbel:
            self.gms = GumbelSoftmax(temperature = 0.5, name = self.name + "_gumbel")
        else:
            self.gms = Softmax(name=self.name + "_softmax")
        
        # Set up up sample and output
        self.conv_up = [Conv1DTranspose(config['filters'][0], 9,
                                        strides=2,
                                        padding="same",
                                        activation = "relu",
                                        name = self.name + "up_{}".format(i)) for i in range(self.down_sample)]
        self.outconv = Conv1D(vocab, 9, padding = 'same', activation = self.gms, name = self.name + "_output")
        

        
    def call(self, x, training = True):
        # Down sample
        for i in range(self.down_sample):
            x = self.conv_down[i](x)
        # Ress modules
        for i in range(self.n_layers):
            x = self.res[i](x, training = training)
            # Attention
            if self.atte_loc == i and self.use_atte:
                x, self.a_w = self.atte(x)
        # Up sample
        for i in range(self.down_sample):
            x = self.conv_up[i](x)
        # Output dist
        x = self.outconv(x)
        return x
    
