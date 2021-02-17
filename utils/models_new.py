import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LeakyReLU, Embedding, Input, LeakyReLU, LayerNormalization, BatchNormalization, Softmax
from utils.layers_new import SelfAttention, ResMod, Spectral_Norm, GumbelSoftmax 



class Generator_res(Model):
    def __init__(self, filters, size, dilation, vocab, use_gumbel, temperature = 0.5):
        super(Generator_res, self).__init__()
        self.res1 = ResMod(filters[0], size[0], dilation = dilation[0])
        self.res2 = ResMod(filters[1], size[1], dilation = dilation[1])
        self.res3 = ResMod(filters[2], size[2], dilation = dilation[2])
        self.res4 = ResMod(filters[3], size[3], dilation = dilation[3])
        self.res5 = ResMod(filters[4], size[4], dilation = dilation[4])
        self.res6 = ResMod(filters[5], size[5], dilation = dilation[5])
        
        self.atte = SelfAttention(256)
        if use_gumbel:
            self.gms = GumbelSoftmax(temperature = 0.5)
        else:
            self.gms = Softmax()
        self.out = Conv1D(vocab, 3, padding = 'same', activation = self.gms)
    def call(self, x):
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x, a_w = self.atte(x)
        x = self.res6(x)
        x = self.out(x)
        return x, a_w
    
class Discriminator_res(Model):
    def __init__(self, filters, size, strides, dilation, vocab, activation = 'linear'):
        super(Discriminator_res, self).__init__()
        self.constraint = Spectral_Norm()
        self.res1 = ResMod(filters[0], size[0], strides=strides[0], dilation = dilation[0], constrains=self.constraint)
        self.res2 = ResMod(filters[1], size[1], strides=strides[1], dilation = dilation[1], constrains=self.constraint)
        self.res3 = ResMod(filters[2], size[2], strides=strides[2], dilation = dilation[2], constrains=self.constraint)
        self.res4 = ResMod(filters[3], size[3], strides=strides[3], dilation = dilation[3], constrains=self.constraint)
        self.res5 = ResMod(filters[4], size[4], strides=strides[4], dilation = dilation[4], constrains=self.constraint)
        self.res6 = ResMod(filters[5], size[5], strides=strides[5], dilation = dilation[5], constrains=self.constraint)
        
        

        self.atte = SelfAttention(vocab)
        self.flatten = Flatten()
        
        self.out = Dense(1, activation = activation, kernel_constraint=self.constraint)
        
    def call(self, x):
        x, a_w = self.atte(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.flatten(x)
        x = self.out(x)
        return x, a_w
    
class Discriminator(Model):
    def __init__(self, filters, size, strides, dilation, vocab, activation = 'sigmoid' ):
        super(Discriminator, self).__init__()
        self.constraint = Spectral_Norm()
        self.act = LeakyReLU(0.2)
        self.conv1 = Conv1D(filters[0], size[0], strides=strides[0], padding='same', kernel_constraint = self.constraint, use_bias = False)
        self.conv2 = Conv1D(filters[0], size[0], strides=strides[0], padding='same', kernel_constraint = self.constraint, use_bias = False)
        self.conv3 = Conv1D(filters[0], size[0], strides=strides[0], padding='same', kernel_constraint = self.constraint, use_bias = False)
        self.conv4 = Conv1D(filters[0], size[0], strides=strides[0], padding='same', kernel_constraint = self.constraint, use_bias = False)
        self.atte = SelfAttention(vocab)
        self.conv = Conv1D(1, 4, strides=1, activation= activation, padding='same', kernel_constraint = self.constraint, use_bias = False)
        
    def call(self, x):
        x, a_w = self.atte(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        return self.conv(x), a_w
    
class Classifier(Model):
    def __init__(self, filters, size, strides, dilation, vocab, input_shape = (512, 21), **kwargs):
        super(Classifier, self).__init__( **kwargs)
        
        self.input_layer = Input(input_shape)
        
        self.constraint = Spectral_Norm()
      #  self.emb  = Embedding(vocab, 10)
        self.noise = tf.keras.layers.GaussianNoise(1.0)
        self.sfm   = tf.keras.layers.Softmax(axis=-1)
        
        self.res = [ResMod(filters[i], size[i], strides=strides[i], dilation = dilation[i], constrains=None) for i in range(9)]
        
        
        self.atte = SelfAttention(vocab)
        self.flatten = Flatten()
        self.out1 = Dense(256, activation = 'relu', kernel_constraint=self.constraint)
        self.out2 = Dense(1, activation = None, kernel_constraint=self.constraint)
        
        # Get output layer with `call` method
        self.out = self.call(self.input_layer)

        # Reinitial
        super(Classifier, self).__init__(
            inputs=self.input_layer,
            outputs=self.out,
            **kwargs)
    def build(self):
        # Initialize the graph
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.out
        )
        
    def call(self, x, training = True):
    #    x = self.emb(x)
        x = self.sfm(self.noise(x*5, training))
        x, self.a_w = self.atte(x)
        for i in range(9):
            x= self.res[i](x)
        x = self.flatten(x)
        x = self.out1(x)
        x = self.out2(x)
        return x
    
class Classifier1(Model):
    def __init__(self, filters, size, strides, dilation, vocab, input_shape = (512, 21), **kwargs):
        super(Classifier1, self).__init__( **kwargs)
        
        self.input_layer = Input(input_shape)
        
        self.constraint = Spectral_Norm()
        self.noise = tf.keras.layers.GaussianNoise(1.0)
        self.sfm   = tf.keras.layers.Softmax(axis=-1)
        
        self.norm = [BatchNormalization(beta_initializer='zeros' ,gamma_initializer="ones") for i in range(10)]
        self.res  = [Conv1D(filters[i], size[i], strides=strides[i], kernel_constraint=self.constraint) for i in range(9)]
        self.atte = SelfAttention(vocab)
        self.flatten = Flatten()
        self.out1 = Dense(256, kernel_constraint=self.constraint)
        self.out2 = Dense(1, activation = 'linear', kernel_constraint=self.constraint)
        
        # Get output layer with `call` method
        self.out = self.call(self.input_layer)

        # Reinitial
        super(Classifier1, self).__init__(
            inputs=self.input_layer,
            outputs=self.out,
            **kwargs)
    def build(self):
        # Initialize the graph
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.out
        )
        
    def call(self, x, training = True):
        x = self.sfm(self.noise(x*5, training))
        x, self.a_w = self.atte(x)
        for i in range(9):
            x = tf.keras.activations.relu((self.res[i](x)))
            
        x = self.flatten(x)
        x = tf.keras.activations.relu((self.out1(x)))
        x = self.out2(x)
        return x
    