import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer, Conv1D, BatchNormalization, LayerNormalization, Add, Concatenate, LeakyReLU, Softmax, Dropout
from tensorflow.keras.regularizers import L1L2
from typeguard import typechecked

from utils.layers_new import SpectralNormalization

def linear(x):
    return x

class ResMod_old(Layer):
    """
    Residual layer as originaly constructed
    No Normalization is applied to the output     
    
    """
    def __init__(self, filters, size, strides=1, dilation=1, use_bias = False, spectral_norm = False):
        """
        Args: 
        
            filter: filters in residual block
            size: kernel size in residual bloc
            dilation: dilation rate in block
            stride: wether or not to use strides at the end of the block if 1< 
            use_bias: weather to use bias in the convolutions
            spectral_norm: weather or not to use spectral normalization
        
        """

        super(ResMod_old, self).__init__()
        if spectral_norm:
            self.norm = SpectralNormalization 
        else:
            self.norm = linear
            
        self.use_bias = use_bias
        
        self.conv1 = self.norm(Conv1D(filters,
                                size,
                                dilation_rate = dilation,
                                padding = 'same',
                                use_bias = self.use_bias))
        
        self.conv2 = self.norm(Conv1D(filters,
                                size,
                                dilation_rate = dilation,
                                padding = 'same',
                                use_bias = self.use_bias))
        
        self.conv3 = self.norm(Conv1D(filters,
                                size,
                                dilation_rate = dilation,
                                padding = 'same',
                                use_bias = self.use_bias))
        self.strides = False
        
        if strides > 1:
            self.strides = True
            self.conv4 = self.norm(Conv1D(filters,
                            size,
                            dilation_rate = 1,
                            strides = strides,
                            padding ='same',
                            use_bias = self.use_bias))
            
            
        self.conv  = self.norm(Conv1D(filters, 1,
                                padding = 'same',
                                use_bias = False))   
        self.add = Add()
        self.act = LeakyReLU(0.2)
        
    def call(self, x):
        x_in = self.conv(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.add([x, x_in])
        if self.strides:
            x = self.act(self.conv4(x)) 
        x = self.act(x)
        return x
    
class ResModPreAct(Layer):
    """
    Residual layer with full preactivation schema
    No Normalization is applied to the output     
    
    """
    def __init__(self, filters, size, strides=1, dilation=1, constrains = None, l1=0.0, l2=0.0, rate = 0.2, use_bias=False, spectral_norm=False, name = "res"):
        super(ResModPreAct, self).__init__(name = name)
        self.filters = filters
        self.kernel  = size
        self.strides = strides
        self.dilation= dilation
        self.constrains = constrains
        self.l1 = l1
        self.l2 = l2
        self.rate = rate
        self.use_bias = use_bias
        if spectral_norm:
            self.norm = SpectralNormalization 
        else:
            self.norm = linear
 
        self.conv1 = self.norm(Conv1D(self.filters, 
                                    self.kernel,
                                    dilation_rate = self.dilation,
                                    padding = 'same',
                                    use_bias = self.use_bias,
                                    kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                    name = self.name + "_conv_1"), name = self.name + "_conv_1_norm")
        
        self.conv2 = self.norm(Conv1D(self.filters,
                                    self.kernel,
                                    dilation_rate = self.dilation,
                                    padding = 'same',
                                    use_bias = self.use_bias,
                                    kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                    name = self.name + "_conv_2"), name = self.name + "_conv_2_norm")
                    
            
        self.conv  = self.norm(Conv1D(self.filters, 1,
                                    padding = 'same',
                                    use_bias = False,
                                    kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                    name = self.name + "_conv_3"), name = self.name + "_conv_3_norm")
        if self.strides > 1:
            self.conv3 = self.norm(Conv1D(self.filters,
                                        self.kernel,
                                        dilation_rate = 1,
                                        strides = self.strides,
                                        padding = 'same',
                                        use_bias = self.use_bias,
                                        kernel_regularizer = L1L2(l1=self.l1, l2=self.l2),
                                        name = self.name + "_conv_4"), name = self.name + "_conv_4_norm")
    
        self.add  = Add()
        self.dout = Dropout(self.rate, name = self.name + "_dropout")
        self.act  = LeakyReLU(0.2, name = self.name + "_activation")



        
    def call(self, x, training=True):
        x_in = self.conv(x)

        x = self.conv1(self.act(x))
        x = self.conv2(self.act(x))
        
        x = self.dout(x, training = training)
        x = self.add([x, x_in])
        
        if self.strides > 1:
            x = self.act(self.conv3(x)) 
        
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel': self.kernel,
            'strides': self.strides,
            'dilation': self.dilation,
            'l1': self.l1,
            'l2': self.l2,
            'rate': self.rate,
            'norm': self.norm,
            'use_bias': self.use_bias
            
        })
        return config