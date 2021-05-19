import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer, Conv1D, BatchNormalization, LayerNormalization, Add, Concatenate, LeakyReLU, Softmax, Dropout
from tensorflow.keras.regularizers import L1L2



POWER_ITERATIONS = 5


    
class GumbelSoftmax(Layer):
    def __init__(self,temperature = 0.5, *args, **kwargs):
        super(GumbelSoftmax,self).__init__()
        
        # Temperature
        self.tau = tf.Variable(temperature, dtype=tf.float32, trainable=False)
        self.smx = Softmax()
    
    def call(self, logits):
        U = tf.random.uniform(tf.shape(logits), minval=0, maxval=1, dtype=tf.dtypes.float32)
        g = -tf.math.log(-tf.math.log(U+1e-20)+1e-20)
        prob = self.smx(logits)
        log_prob = tf.math.log(prob)
        nom = tf.keras.activations.softmax((g + logits)/self.tau, axis=-1)
        return nom

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'tau': self.tau
        })
        return config
    
class Conv1DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides=1, activation = None, name=None,use_bias=False, *args, **kwargs):
        self._filters = filters
        self._kernel_size =  kernel_size
        self._strides =  strides
        self.activation = activation
        self._args, self._kwargs = args, kwargs
        self.use_bias = use_bias
        self.conv = Conv1D(self._filters,
                           kernel_size=self._kernel_size,
                           strides=1,
                           activation = self.activation,
                           padding='same',
                           use_bias=self.use_bias)
        
        super(Conv1DTranspose, self).__init__(name=name)
        
    def call(self, x):
        shape = tf.shape(x)
        b_size = shape[0]
        s_size = shape[1]
        f_size = shape[2]
        n = tf.zeros_like(x)
        x = tf.stack([x,n],axis = 1)
        x = tf.transpose(x, perm=[0,2,1,3])
        x = tf.reshape(x, shape=(b_size,s_size*2,f_size,))
        x = self.conv(x)
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            '_filters': self._filters,
            '_kernel_size': self._kernel_size,
            '_strides': self._strides,
            'activation': self.activations,
            '_args': self._args,
            '_kwargs': self._kwargs,
            'use_bias': use_bias
            
        })
        return config

class SelfAttentionSN(Layer):
    def __init__(self,  filters ):
        super(SelfAttentionSN, self).__init__()
        
        self.kernel_querry = tfa.layers.SpectralNormalization(tf.keras.layers.Dense(filters))
        self.kernel_key    = tfa.layers.SpectralNormalization(tf.keras.layers.Dense(filters))
        self.kernel_value  = tfa.layers.SpectralNormalization(tf.keras.layers.Dense(filters))
        self.out           = tfa.layers.SpectralNormalization(tf.keras.layers.Dense(filters))
        self.num_heads = 8
        self.filters = filters
        self.depth = filters // self.num_heads
        self.gamma = self.add_weight(name='gamma', initializer=tf.keras.initializers.Constant(value=1), trainable=True)
        self.dout = Dropout(0.3)
        self.norm = LayerNormalization(axis = -1, epsilon = 1e-6)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])        
    
    def call(self, x, mask=None, training = True):
        batch_size = tf.shape(x)[0]
        
        querry = self.kernel_querry(x)
        key    = self.kernel_key(x)
        value  = self.kernel_value(x)
        
        querry = self.split_heads( querry, batch_size)
        key    = self.split_heads( key, batch_size)
        value  = self.split_heads( value, batch_size)
         
        
        attention_logits  = tf.matmul(querry, key, transpose_b = True)
        attention_weights = tf.math.softmax(attention_logits, axis=-1)
        
        attention_feature_map = tf.matmul(attention_weights, value)
        if mask is not None:
            attention_feature_map = tf.math.multiply(attention_feature_map, mask)
            
        attention_feature_map = tf.transpose(attention_feature_map, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attention_feature_map, (batch_size, -1, self.filters))
        concat_attention = self.dout(concat_attention, training = training)
        out = x + self.out(concat_attention)*self.gamma
        out = self.norm(out)
        return out, attention_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters     
        })
        return config
    
class SelfAttention(Layer):
    def __init__(self,  filters ):
        super(SelfAttention, self).__init__()
        
        self.kernel_querry = tf.keras.layers.Dense(filters)
        self.kernel_key    = tf.keras.layers.Dense(filters)
        self.kernel_value  = tf.keras.layers.Dense(filters)
        self.out           = tf.keras.layers.Dense(filters)
        self.num_heads = 8
        self.filters = filters
        self.depth = filters // self.num_heads
        self.gamma = self.add_weight(name='gamma', initializer=tf.keras.initializers.Constant(value=1), trainable=True)
        self.dout = Dropout(0.3)
        self.norm = LayerNormalization(axis = -1, epsilon = 1e-6)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])        
    
    def call(self, x, mask=None, training = True):
        batch_size = tf.shape(x)[0]
        
        querry = self.kernel_querry(x)
        key    = self.kernel_key(x)
        value  = self.kernel_value(x)
        
        querry = self.split_heads( querry, batch_size)
        key    = self.split_heads( key, batch_size)
        value  = self.split_heads( value, batch_size)
         
        
        attention_logits  = tf.matmul(querry, key, transpose_b = True)
        attention_weights = tf.math.softmax(attention_logits, axis=-1)
        
        attention_feature_map = tf.matmul(attention_weights, value)
        if mask is not None:
            attention_feature_map = tf.math.multiply(attention_feature_map, mask)
            
        attention_feature_map = tf.transpose(attention_feature_map, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attention_feature_map, (batch_size, -1, self.filters))
        concat_attention = self.dout(concat_attention, training = training)
        out = x + self.out(concat_attention)*self.gamma
        out = self.norm(out)
        return out, attention_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters     
        })
        return config
    
class ResMod_old(Layer):
    def __init__(self, filters, size, strides=1, dilation=1, constrains = None):
        super(ResMod_old, self).__init__()
        
        self.conv1 = tfa.layers.SpectralNormalization(Conv1D(filters,
                                                             size,
                                                             dilation_rate = dilation,
                                                             padding = 'same',
                                                             use_bias = False))
        
        self.conv2 = tfa.layers.SpectralNormalization(Conv1D(filters,
                                                             size,
                                                             dilation_rate = dilation,
                                                             padding = 'same',
                                                             use_bias = False))
        
        self.conv3 = tfa.layers.SpectralNormalization(Conv1D(filters,
                                                             size,
                                                             dilation_rate = dilation,
                                                             padding = 'same',
                                                             use_bias = False))
        self.strides = False
        
        if strides > 1:
            self.strides = True
            self.conv4 = tfa.layers.SpectralNormalization(Conv1D(filters,
                                                                 size,
                                                                 dilation_rate = 1,
                                                                 strides = strides,
                                                                 padding ='same',
                                                                 use_bias = True))
            
            
        self.conv  = tfa.layers.SpectralNormalization(Conv1D(filters, 1,
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

class ResModSN(Layer):
    def __init__(self, filters, size, strides=1, dilation=1, constrains = None, l1=0.0, l2=0.0, rate = 0.2):
        super(ResModSN, self).__init__()
        self.filters = filters
        self.kernel  = size
        self.strides = strides
        self.dilation= dilation
        self.constrains = constrains
        self.l1 = l1
        self.l2 = l2
        self.rate = rate
        self.conv1 = tfa.layers.SpectralNormalization(Conv1D(self.filters, 
                                                             self.kernel,
                                                             dilation_rate = self.dilation,
                                                             padding = 'same',
                                                             use_bias = True,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        
        self.conv2 = tfa.layers.SpectralNormalization(Conv1D(self.filters,
                                                             self.kernel,
                                                             dilation_rate = self.dilation,
                                                             padding = 'same',
                                                             use_bias = True,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        
        self.conv3 = tfa.layers.SpectralNormalization(Conv1D(self.filters,
                                                             self.kernel,
                                                             dilation_rate = self.dilation,
                                                             padding = 'same',
                                                             use_bias = True,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))

        
        if self.strides > 1:
            self.conv4 = tfa.layers.SpectralNormalization(Conv1D(self.filters,
                                                                 self.kernel,
                                                                 dilation_rate = 1,
                                                                 strides = self.strides,
                                                                 padding = 'same',
                                                                 use_bias = True,
                                                                 kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
            
            
        self.conv  = tfa.layers.SpectralNormalization(Conv1D(self.filters, 1,
                                                             padding = 'same',
                                                             use_bias = False,
                                                             kernel_constraint = self.constrains,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))   
        self.add  = Add()
        self.dout = Dropout(self.rate)
        self.act  = LeakyReLU(0.1)
        self.norm1 = LayerNormalization(axis = -1)


        
    def call(self, x, training=True):
        x_in = self.conv(x)
        x = self.conv1(self.act(x))
        x = self.conv2(self.act(x))
        x = self.conv3(self.act(x))
        x = self.add([x, x_in])
        x = self.dout(x, training = training)
        if self.strides > 1:
            x = self.act(self.conv4(x)) 
        return self.norm1(x)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel': self.kernel,
            'strides': self.strides,
            'dilation': self.dilation,
            'constrains': self.constrains,
            'l1': self.l1,
            'l2': self.l2,
            'rate': self.rate
            
        })
        return config
    
class ResMod(Layer):
    def __init__(self, filters, size, strides=1, dilation=1, l1=0.01, l2=0.01, rate = 0.2):
        super(ResMod, self).__init__()
        self.filters = filters
        self.kernel  = size
        self.strides = strides
        self.dilation= dilation
        self.l1 = l1
        self.l2 = l2
        self.rate = rate
        
        self.conv1 = tfa.layers.SpectralNormalization(Conv1D(self.filters, 
                            self.kernel,
                            dilation_rate = self.dilation,
                            padding = 'same',
                            use_bias = True,
                            kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        self.conv2 = tfa.layers.SpectralNormalization(Conv1D(self.filters,
                            self.kernel,
                            dilation_rate = self.dilation,
                            padding = 'same',
                            use_bias = True,
                            kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
                    
            
        self.conv  = tfa.layers.SpectralNormalization(Conv1D(self.filters, 1,
                            padding = 'same',
                            use_bias = False,
                            kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        if self.strides > 1:
            self.conv3 = tfa.layers.SpectralNormalization(Conv1D(self.filters,
                                self.kernel,
                                dilation_rate = 1,
                                strides = self.strides,
                                padding = 'same',
                                use_bias = True,
                                kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        self.add  = Add()
        self.dout = Dropout(self.rate)
        self.act  = LeakyReLU(0.2)
        self.norm1 = BatchNormalization()
        self.norm2 = BatchNormalization()


        
    def call(self, x, training=True):
        x_in = self.conv(x)

        x = self.conv1(self.act(self.norm1(x, training = training)))
        x = self.conv2(self.act(self.norm2(x, training = training)))
        
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
            'constrains': self.constrains,
            'l1': self.l1,
            'l2': self.l2,
            'rate': self.rate
            
        })
        return config
        
    
class ResModPreActSN(Layer):
    def __init__(self, filters, size, strides=1, dilation=1, constrains = None, l1=0.0, l2=0.0, rate = 0.2):
        super(ResModPreActSN, self).__init__()
        self.filters = filters
        self.kernel  = size
        self.strides = strides
        self.dilation= dilation
        self.constrains = constrains
        self.l1 = l1
        self.l2 = l2
        self.rate = rate
        self.conv1 = tfa.layers.SpectralNormalization(Conv1D(self.filters, 
                                                             self.kernel,
                                                             dilation_rate = self.dilation,
                                                             padding = 'same',
                                                             use_bias = True,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        
        self.conv2 = tfa.layers.SpectralNormalization(Conv1D(self.filters,
                                                             self.kernel,
                                                             dilation_rate = self.dilation,
                                                             padding = 'same',
                                                             use_bias = True,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
                    
            
        self.conv  = tfa.layers.SpectralNormalization(Conv1D(self.filters, 1,
                                                             padding = 'same',
                                                             use_bias = False,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        if self.strides > 1:
            self.conv3 = tfa.layers.SpectralNormalization(Conv1D(self.filters,
                                                                 self.kernel,
                                                                 dilation_rate = 1,
                                                                 strides = self.strides,
                                                                 padding = 'same',
                                                                 use_bias = True,
                                                                 kernel_regularizer = L1L2(l1=self.l1, l2=self.l2)))
        self.add  = Add()
        self.dout = Dropout(self.rate)
        self.act  = LeakyReLU(0.2)



        
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
            'rate': self.rate
            
        })
        return config
    
class ResModPreAct(Layer):
    def __init__(self, filters, size, strides=1, dilation=1, constrains = None, l1=0.0, l2=0.0, rate = 0.2):
        super(ResModPreAct, self).__init__()
        self.filters = filters
        self.kernel  = size
        self.strides = strides
        self.dilation= dilation
        self.constrains = constrains
        self.l1 = l1
        self.l2 = l2
        self.rate = rate
        self.conv1 = Conv1D(self.filters, 
                                                             self.kernel,
                                                             dilation_rate = self.dilation,
                                                             padding = 'same',
                                                             use_bias = True,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2))
        
        self.conv2 = Conv1D(self.filters,
                                                             self.kernel,
                                                             dilation_rate = self.dilation,
                                                             padding = 'same',
                                                             use_bias = True,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2))
                    
            
        self.conv  = Conv1D(self.filters, 1,
                                                             padding = 'same',
                                                             use_bias = False,
                                                             kernel_regularizer = L1L2(l1=self.l1, l2=self.l2))
        if self.strides > 1:
            self.conv3 = Conv1D(self.filters,
                                                                 self.kernel,
                                                                 dilation_rate = 1,
                                                                 strides = self.strides,
                                                                 padding = 'same',
                                                                 use_bias = True,
                                                                 kernel_regularizer = L1L2(l1=self.l1, l2=self.l2))
        self.add  = Add()
        self.dout = Dropout(self.rate)
        self.act  = LeakyReLU(0.2)



        
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
            'rate': self.rate
            
        })
        return config
        