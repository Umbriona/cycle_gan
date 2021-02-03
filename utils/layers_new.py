import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer, Conv1D, LayerNormalization, Add, Concatenate, LeakyReLU, Softmax
import tensorflow_probability as tfp


POWER_ITERATIONS = 5

class Spectral_Norm(Constraint):
    '''
    Uses power iteration method to calculate a fast approximation 
    of the spectral norm (Golub & Van der Vorst)
    The weights are then scaled by the inverse of the spectral norm
    '''
    def __init__(self, power_iters=POWER_ITERATIONS):
        self.n_iters = power_iters

    def __call__(self, w):
        flattened_w = tf.reshape(w, [w.shape[0], -1])
        u = tf.random.normal([flattened_w.shape[0]])
        v = tf.random.normal([flattened_w.shape[1]])
        for i in range(self.n_iters):
            v = tf.linalg.matvec(tf.transpose(flattened_w), u)
            v = tf.keras.backend.l2_normalize(v)
            u = tf.linalg.matvec(flattened_w, v)
            u = tf.keras.backend.l2_normalize(u)
            sigma = tf.tensordot(u, tf.linalg.matvec(flattened_w, v), axes=1)
        return w / sigma

    def get_config(self):
        return {'n_iters': self.n_iters}
    
class GumbelSoftmax(Layer):
    def __init__(self,temperature = 0.5, *args, **kwargs):
        super(GumbelSoftmax,self).__init__()
        tfd = tfp.distributions

        # Define a single scalar Gumbel distribution.
        #self.dist = tfd.Gumbel(loc=0., scale=1.)
        
 
        # Temperature
        self.tau = temperature
    
    def call(self, logits):
        g = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), minval=0, maxval=1, dtype=tf.dtypes.float32)))
        nom = tf.keras.activations.softmax(tf.math.exp((g + tf.math.log(logits))/self.tau), axis=-1)
        return nom

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

class SelfAttention(Layer):
    def __init__(self, filters):
        super(SelfAttention, self).__init__()
        self.kernel_querry = Conv1D(max(1,filters//8), 1, padding= 'same', use_bias = False)
        self.kernel_key    = Conv1D(max(1,filters//8), 1, padding= 'same', use_bias = False)
        self.kernel_value  = Conv1D(max(1,filters//8), 1, padding= 'same', use_bias = False)
        self.out           = Conv1D(filters,    1, padding = 'same', use_bias = False)
        self.gamma = self.add_weight(name='gamma', initializer=tf.keras.initializers.Constant(value=1), trainable=True)
            
            
    def call(self, x, mask=None):
        querry = self.kernel_querry(x)
        key = self.kernel_key(x)
        value = self.kernel_value(x)
        attention_weights = tf.math.softmax(tf.matmul(querry, key, transpose_b = True), axis=1)
        attention_feature_map = tf.matmul(value, attention_weights, transpose_a = True)
        if mask is not None:
            attention_feature_map = tf.math.multiply(attention_feature_map, mask)
        attention_feature_map = tf.transpose(attention_feature_map, [0,2,1])
        
        out = x + self.out(attention_feature_map)*self.gamma
        
        return out, attention_weights

class ResMod(Layer):
    def __init__(self, filters, size, strides=1, dilation=1, constrains = None):
        super(ResMod, self).__init__()
        
        self.conv1 = Conv1D(filters, size,
                            dilation_rate = dilation,
                            padding = 'same',
                            use_bias = False,
                            kernel_constraint = constrains )
        self.conv2 = Conv1D(filters, size, dilation_rate = dilation, padding = 'same',
                            use_bias = False,
                            kernel_constraint = constrains )
        self.conv3 = Conv1D(filters, size, dilation_rate = dilation, padding = 'same',
                            use_bias = False,
                            kernel_constraint = constrains )
        self.strides = False
        
        if strides > 1:
            self.strides = True
            self.conv4 = Conv1D(filters, size, dilation_rate = 1, strides = strides, padding = 'same',
                            use_bias = False,
                            kernel_constraint = constrains )
            
            
        self.conv  = Conv1D(filters, 1, padding = 'same',
                            use_bias = False,
                            kernel_constraint = constrains )   
        self.add = Add()
        

        self.norm = tfa.layers.InstanceNormalization(
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
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
        