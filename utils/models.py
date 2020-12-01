import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers import LSTM, Dense, Flatten, Input, Dropout, BatchNormalization, Concatenate, Conv1D, MaxPool1D, Dot, Layer, Embedding, Add
from utils.layers import Conv1DTranspose, UNetModule, UNetModule_res, UNetModule_ins, Angularization, DownSampleMod_res
from tensorflow.keras.activations import relu, elu
VOCAB_SIZE = 21


class SelfAttention(Layer):
    def __init__(self, filters):
        super(SelfAttention, self).__init__()
        self.kernel_querry = Conv1D(filters, 1)
        self.kernel_key    = Conv1D(filters, 1)
        self.kernel_value  = Conv1D(filters, 1)
        
        self.gamma = self.add_weight(name='gamma', initializer=tf.keras.initializers.Constant(
    value=1
), trainable=True)
        
    def call(self, x, mask=None):
        
        querry = self.kernel_querry(x)
        key = self.kernel_key(x)
        value = self.kernel_value(x)
        attention_weights = tf.math.softmax(tf.matmul(querry, key, transpose_b = True))
        attention_feature_map = tf.matmul(value, attention_weights, transpose_a = True)
        if mask is not None:
            attention_feature_map = tf.math.multiply(attention_feature_map, mask)
        attention_feature_map = tf.transpose(attention_feature_map, [0,2,1])    
        out = x + attention_feature_map*self.gamma
        return out, attention_weights
    

class UpsampleMod_s_res(Layer):
    def __init__(self, num_filter, size_filter, sampling_stride, use_regular_uppsampling = False,
                size = 2, rate = 0.1, l1 = 0.01, l2 = 0.01, use_max_pool = False):
        super(UpsampleMod_s_res, self).__init__()
        self.use_max_pool = use_max_pool
        self.bn1   = BatchNormalization()
        self.bn2   = BatchNormalization()
        self.bn3   = BatchNormalization()
        self.bn4   = BatchNormalization()

        self.reg1 = L1L2(l1=l1, l2=l2)
        self.reg2 = L1L2(l1=l1, l2=l2)
        self.reg3 = L1L2(l1=l1, l2=l2)
        self.reg4 = L1L2(l1=l1, l2=l2)
        
        self.add = Add()
        self.concat = Concatenate(axis=2)

        self.conv1 = Conv1D(num_filter, size_filter, padding='same',
                            kernel_regularizer = self.reg1, use_bias=False)
        self.conv2 = Conv1D(num_filter, size_filter, padding='same',
                            kernel_regularizer = self.reg2, use_bias=False)
        self.conv3 = Conv1D(num_filter, size_filter, padding='same',
                            kernel_regularizer = self.reg3, use_bias=False)
        
        if not self.use_max_pool:
            self.u_sample = Conv1DTranspose(num_filter, size_filter, strides = sampling_stride,
                            kernel_regularizer = self.reg4, use_bias=False)
        else:
            self.u_sample = UpSampling1D(size = size)
        self.dOut = Dropout(rate)
    
    def call(self, x, training=None):
        x = self.u_sample(x)
        x = relu(self.bn1(x, training=training))
        x_c = self.conv1(x)
        x = relu(self.bn2(x, training=training))
        x = relu(self.bn3(self.conv2(x),training=training))
        x = self.conv3(x)
        x = relu(self.bn4(self.add([x_c, x]),training=training))
        return self.dOut(x, training=training)  


class U_net(Model):    
    def __init__(self, p, typ = 'original', emb = True):
        super(U_net,self).__init__()
        assert typ in ['original', 'res', 'ins'], r"typ needs to be one of 'original', 'res' or 'ins' "
        self.h_parameters = p
        reg = L1L2(l1 = p['l1'][-1], l2 = p['l2'][-1])
        if typ == 'original':
            self.u_net_module = UNetModule(self.h_parameters)
        elif typ == 'res':
            self.u_net_module = UNetModule_res(self.h_parameters)
        elif typ == 'ins':
            self.u_net_module = UNetModule_ins(self.h_parameters)
        if emb:
            self.embedding = Embedding(VOCAB_SIZE, self.h_parameters['emb_size'])
            self.bn_embedding = BatchNormalization()
        self.conv_out = Conv1D(VOCAB_SIZE, self.h_parameters['kernel_size'][-1]
                               ,activation=self.h_parameters['output_activation'], padding='same'
                               ,kernel_regularizer = reg, name='out1')
        self.emb = emb
        
    def call(self, x, training = True):
        if self.emb:
            x = self.embedding(x)
            x = self.bn_embedding(x, training = training)
            x = self.u_net_module(x, training = training)
            out = self.conv_out(x)
        else:
            x = self.u_net_module(x, training = training)
            out = self.conv_out(x)
        return out

class Discriminator(Model):
    def __init__(self, p):
        super(Discriminator,self).__init__()
        
        # Expanding Hyper parameters
        num_filter = p['num_filter']
        size_filter= p['kernel_size']
        sampling_stride = p['sampling_stride']
        l1 = p['l1']
        l2 = p['l2']
        emb_size = ['emb_size']
        
        #self.embeding = Embedding(VOCAB_SIZE, emb_size)
        
        self.mod1 = DownSampleMod_res(num_filter[0], size_filter[0], sampling_stride[0], use_max_pool=False, l1 = l1[0], l2 = l2[0], rate = 0.0, sample=True)
        self.mod2 = DownSampleMod_res(num_filter[1], size_filter[1], sampling_stride[1], use_max_pool=False, l1 = l1[1], l2 = l2[1], rate = 0.0, sample=True)
        self.mod3 = DownSampleMod_res(num_filter[2], size_filter[2], sampling_stride[2], use_max_pool=False, l1 = l1[2], l2 = l2[2], rate = 0.0, sample=True)
        self.mod4 = DownSampleMod_res(num_filter[0], size_filter[0], sampling_stride[0], use_max_pool=False, l1 = l1[0], l2 = l2[0], rate = 0.0, sample=True)
        
        self.attention = SelfAttention(num_filter[0])
        self.flatten = tf.keras.layers.Flatten()
        self.out = Dense(1, activation='sigmoid')
        
    def call(self, x, training=True):
        x = self.mod1(x, training = training)
        x, a_w = self.attention(x[0])
        x = self.mod2(x, training = training)
        x = self.mod3(x[0], training = training)
        x = self.mod4(x[0], training = training)
        x = self.flatten(x[0])
        out = self.out(x)
        return out, a_w
    
    
class Discriminator_conv(Model):
    def __init__(self, p):
        super(Discriminator_conv,self).__init__()
        
        # Expanding Hyper parameters
        num_filter = p['num_filter']
        size_filter= p['kernel_size']
        sampling_stride = p['sampling_stride']
        l1 = p['l1']
        l2 = p['l2']
        emb_size = ['emb_size']
        
        #self.embeding = Embedding(VOCAB_SIZE, emb_size)
        
        self.mod1 = DownSampleMod_res(num_filter[0], size_filter[0], sampling_stride[0], use_max_pool=False, l1 = l1[0], l2 = l2[0], rate = 0.0, sample=True)
        self.mod2 = DownSampleMod_res(num_filter[1], size_filter[1], sampling_stride[1], use_max_pool=False, l1 = l1[1], l2 = l2[1], rate = 0.0, sample=True)
        self.mod3 = DownSampleMod_res(num_filter[2], size_filter[2], sampling_stride[2], use_max_pool=False, l1 = l1[2], l2 = l2[2], rate = 0.0, sample=True)
        self.mod4 = DownSampleMod_res(num_filter[0], size_filter[0], sampling_stride[3], use_max_pool=False, l1 = l1[0], l2 = l2[0], rate = 0.0, sample=True)
        
        self.attention = SelfAttention(num_filter[0])
        self.flatten = tf.keras.layers.Flatten()
        self.out = Conv1D(1, 3, activation='sigmoid')
        
    def call(self, x, training=True):
        x = self.mod1(x, training = training)
        x, a_w = self.attention(x[0])
        x = self.mod2(x, training = training)
        x = self.mod3(x[0], training = training)
        x = self.mod4(x[0], training = training)
        out = self.out(x[0])
        return out , a_w
    
class Generator(Model):
    def __init__(self, p, emb = True):
        super(Generator,self).__init__()
        
        # Expanding Hyper parameters
        num_filter = p['num_filter']
        size_filter= p['kernel_size']
        sampling_stride = p['sampling_stride']
        l1 = p['l1']
        l2 = p['l2']
        emb_size = p['emb_size']
        self.emb = emb

        self.embedding = Embedding(VOCAB_SIZE, emb_size)
        
        self.mod1 = DownSampleMod_res(num_filter[0], size_filter[0], sampling_stride[0], use_max_pool=False, l1 = l1[0], l2 = l2[0], rate = 0.0, sample=True)
        self.mod2 = DownSampleMod_res(num_filter[1], size_filter[1], sampling_stride[1], use_max_pool=False, l1 = l1[1], l2 = l2[1], rate = 0.0, sample=True)
        self.mod3 = DownSampleMod_res(num_filter[2], size_filter[2], sampling_stride[2], use_max_pool=False, l1 = l1[2], l2 = l2[2], rate = 0.0, sample=True)
        
        self.mod4 = UpsampleMod_s_res(num_filter[2], size_filter[0], sampling_stride[0], use_max_pool=False, l1 = l1[0], l2 = l2[0], rate = 0.0)
        self.mod5 = UpsampleMod_s_res(num_filter[1], size_filter[1], sampling_stride[1], use_max_pool=False, l1 = l1[1], l2 = l2[1], rate = 0.0)
        self.mod6 = UpsampleMod_s_res(num_filter[0], size_filter[2], sampling_stride[2], use_max_pool=False, l1 = l1[2], l2 = l2[2], rate = 0.0)
        
        self.endconv = Conv1D(VOCAB_SIZE, size_filter[2], padding='same')

        
    def call(self, x):

        if self.emb:
            x = self.embedding(x)
        x = self.mod1(x, training = True)
        x = self.mod2(x[0], training = True)
        x = self.mod3(x[0], training = True)
        
        x = self.mod4(x[0], training = True)
        x = self.mod5(x, training = True)
        x = self.mod6(x, training = True)
        
        x = self.endconv(x)
        x = tf.keras.activations.softmax(x)

        return x


####################################### PixelCNN ###################################

class Causal_layer(Layer):
    def __init__(self,):
        super(Causal_layer, self).__init__()
        self.padding  = tf.constant([[0,0], [s_kernel, 0], [0, 0]])
    
    def call(self):
        return 0
class Gated_Res(Layer):
    def __init__(self, n_filter, s_kernel, dilation, rate = 0.05, first = False):
        super(Gated_Res, self).__init__()
        self.conv = Conv1D(n_filter, s_kernel, padding = 'valid', dilation_rate=dilation, activation='tanh')
        self.gate = Conv1D(n_filter, s_kernel, padding = 'valid', dilation_rate=dilation, activation='sigmoid')
        self.first = first
        if first:
            self.conv_in = Conv1D(n_filter, 1, padding='same')
            
        self.add = Add()
        self.bn = BatchNormalization()
        self.do = Dropout(rate)

        self.paddings = tf.constant([[0,0], [s_kernel-1, 0], [0, 0]])

        
    def call(self, x_in, training = True):
        x_pad  = tf.pad(x_in, self.paddings, "CONSTANT") 
        if self.first:
            x_in = self.conv_in(x_in)
        x_val = self.conv(x_pad)
        x_gat = self.gate(x_pad)
        x_out = tf.math.multiply(x_val, x_gat)
        x_out = self.add([x_in, x_out])
        x_out = self.bn(x_out, training)
        return self.do(x_out, training)
    
class AminoCNN_Mod(Layer):
    def __init__(self, n_filter, s_kernel, dilation, rate = 0.05):
        super(AminoCNN_Mod, self).__init__()
        self.layer1 = Gated_Res(n_filter, s_kernel, dilation, rate, first = True)
        self.layer2 = Gated_Res(n_filter, s_kernel, dilation, rate)
        self.layer3 = Gated_Res(n_filter, s_kernel, dilation, rate)
        
    def call(self, x, training = True):
        x = self.layer1(x, training)
        x = self.layer2(x, training)
        x = self.layer3(x, training)
        return x
    
    


class AminoCNN(Model):
    def __init__(self,vocab, emb_s, n_filter, s_kernel, dilation, rate = 0.05):
        super(AminoCNN,self).__init__()
        
        self.embedd = Embedding(vocab, emb_s)
        self.causal = 0
        self.module1 = AminoCNN_Mod(n_filter, s_kernel, dilation, rate)
        self.module2 = AminoCNN_Mod(n_filter, s_kernel, dilation, rate)
        self.module3 = AminoCNN_Mod(n_filter, s_kernel, dilation, rate)
        self.module4 = AminoCNN_Mod(n_filter, s_kernel, dilation, rate)
        self.module5 = AminoCNN_Mod(n_filter, s_kernel, dilation, rate)
        self.module6 = AminoCNN_Mod(n_filter, s_kernel, dilation, rate)

        self.conv = Conv1D(vocab, 3, padding='same', activation='softmax')
        self.concat = Concatenate(axis=-1)
    
    def call(self, x, training):
        x = self.embedd(x)
        x1 = self.module1(x, training)
        x2 = self.module2(x1, training)
        x3 = self.module3(x2, training)
        x4 = self.module4(x3, training)
        x5 = self.module5(self.concat([x4,x2]), training)
        x6 = self.module6(self.concat([x5,x1]), training)
        x_out = self.conv(x6)
        return x_out
        

    
####################################### Transformer Model #################################################3#
    
class Encoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, max_seq_len, rate = 0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = layers.TokenAndPositionEmbedding(max_seq_len, input_vocab_size, d_model)
        self.enc_layers = [layers.TransformerEncoderModule(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.max_seq_len = max_seq_len
        self.rate = rate

    def call(self, x, training, mask=None):

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x += self.pos_encoding(tf.range(seq_len))
        x = self.dropout(x, training = training)
        for i in range(self.num_layers):
            x, _ = self.enc_layers[i](x, training)

        return x 
        
class Decoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, max_seq_len, rate = 0.1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = layers.TokenAndPositionEmbedding(max_seq_len, input_vocab_size, d_model)
        self.dec_layers = [layers.TransformerDecoderModule(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.max_seq_len = max_seq_len
        self.rate = rate

    def call(self, x, enc_out, training, mask=None):

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x += self.pos_encoding(tf.range(seq_len))
        x = self.dropout(x, training = training)
        for i in range(self.num_layers):
            x, weight_1, weight_2 = self.dec_layers[i](x,enc_out, training)
        
            attention_weights['decoder_layer{}_block1'.format(i+1)] = weight_1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = weight_2
    
        return x, attention_weights
    
#class Transformer(Layers):

############################## H param optimization #######################################
def u_net_module_hp(hp):
    
############################## Special Fun ###################################################

   # embedding = tf.Variable(initial_value = np.load('../Embedding/embedd_w_5_ogt.npy').reshape(1,21,5))
    
############################## Transformation module #################################################
    
    p = {'n_fil_1':hp.Choice('number_filters_conv1', [8, 16, 32, 64]),
         's_fil_1':hp.Choice('size_filters_conv1',[3, 5, 7, 10]),
         'stride_1':hp.Choice('stride_length_sampling1', [2,4]),
         'dOut_1':hp.Choice('Dropout_module1',[0.1, 0.2, 0.3, 0.4]),
         'n_fil_2':hp.Choice('number_filters_conv2', [32, 64, 128]),
         's_fil_2':hp.Choice('size_filters_conv2',[3, 5, 7, 10]),
         'stride_2':hp.Choice('stride_length_sampling2', [2,4]),
         'dOut_2':hp.Choice('Dropout_module2',[0.1, 0.2, 0.3, 0.4]),
         'n_fil_3':hp.Choice('number_filters_conv3', [64, 128, 256]),
         's_fil_3':hp.Choice('size_filters_conv3',[3, 5, 7, 10]),
         'stride_3':hp.Choice('stride_length_sampling3', [2,4]),
         'dOut_3':hp.Choice('Dropout_module3',[0.1, 0.2, 0.3, 0.4]),
         'n_fil_4':hp.Choice('number_filters_conv4', [128, 256, 512]),
         's_fil_4':hp.Choice('size_filters_conv4',[3, 5, 7, 10]),
         'dOut_4':hp.Choice('Dropout_module4',[0.1, 0.2, 0.3, 0.4]),
         'dOut_5':hp.Choice('Dropout_module5',[0.05, 0.1, 0.2, 0.3]),
         's_fil_5':hp.Choice('size_filters_conv5',[3, 5, 7, 10])
        }
    
    # Layers of stage 0 contraction
    
    inp    = Input(shape=(1024,21))
                 
    #tct0_in =  Dot(axes=(2,1))([inp,embedding])
    tct0_bn1   = BatchNormalization()(inp)#tct0_in)
    #tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
    tct0_conv1 = Conv1D(int(p['n_fil_1']), int(p['s_fil_1']), activation='relu', padding='same', name='Convolution_Ct0_0')(tct0_bn1)
    tct0_bn2   = BatchNormalization()(tct0_conv1)
    tct0_conv2 = Conv1D(int(p['n_fil_1']), int(p['s_fil_1']), activation='relu', padding='same', name='Convolution_Ct0_1')(tct0_bn2)
    tct0_bn3   = BatchNormalization()(tct0_conv2)
    tct0_conv3 = Conv1D(int(p['n_fil_1']), int(p['s_fil_1']), activation='relu',strides = int(p['stride_1']), padding='same', name='Convolution_Ct0_2')(tct0_bn3)
    tct0_bn4   = BatchNormalization()(tct0_conv3)
    #tct0_max   = MaxPool1D(pool_size=2, strides=2)(tct0_bn2)
    tct0_dp    = Dropout(p['dOut_1'])(tct0_bn4)
    
    # Layers of stage 1 contraction
    
    tct1_conv1 = Conv1D(int(p['n_fil_2']), int(p['s_fil_2']), activation='relu', padding='same', name='Convolution_Ct1_0')(tct0_dp)
    tct1_bn1   = BatchNormalization()(tct1_conv1)
    tct1_conv2 = Conv1D(int(p['n_fil_2']), int(p['s_fil_2']), activation='relu', strides=1, padding='same', name='Convolution_Ct1_1')(tct1_bn1)
    tct1_bn2   = BatchNormalization()(tct1_conv2)
    tct1_conv3 = Conv1D(int(p['n_fil_2']), int(p['s_fil_2']), activation='relu', strides=int(p['stride_2']), padding='same', name='Convolution_Ct1_2')(tct1_bn2)
    tct1_bn3   = BatchNormalization()(tct1_conv3)
    #tct1_max   = MaxPool1D(pool_size=2, strides=2)(tct1_bn2)
    tct1_dp    = Dropout(p['dOut_2'])(tct1_bn3)
    
    # Layers of stage 2 contraction
    
    tct2_conv1 = Conv1D(int(p['n_fil_3']), int(p['s_fil_3']), activation='relu', padding='same', name='Convolution_Ct2_0')(tct1_dp)
    tct2_bn1   = BatchNormalization()(tct2_conv1)
    tct2_conv2 = Conv1D(int(p['n_fil_3']), int(p['s_fil_3']), activation='relu', strides=1, padding='same', name='Convolution_Ct2_1')(tct2_bn1)
    tct2_bn2   = BatchNormalization()(tct2_conv2)
    tct2_conv3 = Conv1D(int(p['n_fil_3']), int(p['s_fil_3']), activation='relu', strides=int(p['stride_3']), padding='same', name='Convolution_Ct2_2')(tct2_bn2)
    tct2_bn3   = BatchNormalization()(tct2_conv3)
    #tct2_max   = MaxPool1D(pool_size=2, strides=2)(tct2_bn2)
    tct2_dp    = Dropout(p['dOut_3'])(tct2_bn3)
    
    # Layers of stage 3 contraction
    
    tct3_conv1 = Conv1D(int(p['n_fil_4']), int(p['s_fil_4']), activation='relu', padding='same', name='Convolution_Ce3_0')(tct2_dp)
    tct3_bn1   = BatchNormalization()(tct3_conv1)
    tct3_conv2 = Conv1D(int(p['n_fil_4']), int(p['s_fil_4']), activation='relu', padding='same', name='Convolution_Ce3_1')(tct3_bn1)
    tct3_bn2   = BatchNormalization()(tct3_conv2)
    tct3_dp    = Dropout(p['dOut_4'])(tct3_bn2)
    
    # Layers of stage 1 expansion
    
    tet1_Tconv  = Conv1DTranspose(int(p['n_fil_3']), int(p['s_fil_3']), strides=int(p['stride_3']) ,activation='relu', padding='same', name='TransConv_Et1')(tct3_dp)
    tet1_Concat = Concatenate(axis=2)([tet1_Tconv, tct2_conv1])
    tet1_bn1    = BatchNormalization()(tet1_Concat)
    tet1_conv1  = Conv1D(int(p['n_fil_3']), int(p['s_fil_3']), activation='relu', padding='same', name='Convolution_Et1_0')(tet1_bn1)
    tet1_bn2    = BatchNormalization()(tet1_conv1)
    tet1_conv2  = Conv1D(int(p['n_fil_3']), int(p['s_fil_3']), activation='relu', padding='same', name='Convolution_Et1_1')(tet1_bn2)
    tet1_bn3    = BatchNormalization()(tet1_conv2)
    tet1_dp     = Dropout(p['dOut_3'])(tet1_bn3)
    
    #Layers of stage 2 expansion
               
    tet2_Tconv  = Conv1DTranspose(int(p['n_fil_2']), int(p['s_fil_2']), strides=int(p['stride_2']) ,activation='relu', padding='same', name='TransConv_Et2')(tet1_dp)
    tet2_Concat = Concatenate(axis=2)([tet2_Tconv, tct1_conv1])
    tet2_bn1    = BatchNormalization()(tet2_Concat)
    tet2_conv1  = Conv1D(int(p['n_fil_2']), int(p['s_fil_2']), activation='relu', padding='same', name='Convolution_Et2_0')(tet2_bn1)
    tet2_bn2    = BatchNormalization()(tet2_conv1)
    tet2_conv2  = Conv1D(int(p['n_fil_2']), int(p['s_fil_2']), activation='relu', padding='same', name='Convolution_Et2_1')(tet2_bn2)
    tet2_bn3    = BatchNormalization()(tet2_conv2)
    tet2_dp     = Dropout(p['dOut_2'])(tet2_bn3)
                       
    #Layers of stage 3 expansion
               
    tet3_Tconv = Conv1DTranspose(int(p['n_fil_1']), int(p['s_fil_1']), strides=int(p['stride_1']) ,activation='relu', padding='same', name='TransConv_Et3')(tet2_dp)
    tet3_Concat = Concatenate(axis=2)([tet3_Tconv, tct0_conv1])
    tet3_bn1 = BatchNormalization()(tet3_Concat)
    tet3_conv1 = Conv1D(int(p['n_fil_1']), int(p['s_fil_1']), activation='relu', padding='same', name='Convolution_Et3_1')(tet3_bn1)
    tet3_bn2 = BatchNormalization()(tet3_conv1)
    tet3_conv2 = Conv1D(int(p['n_fil_1']), int(p['s_fil_1']), activation='relu', padding='same', name='Convolution_Et3_2')(tet3_bn2)
    tet3_bn3 = BatchNormalization()(tet3_conv2)
    tet3_dp = Dropout(p['dOut_5'])(tet3_bn3)
    tet3_conv3 = Conv1D(3, int(p['s_fil_5']), activation='softmax', padding='same', name='Convolution_Et3_3')(tet3_dp)

    model = Model(inputs = inp, outputs = tet3_conv3)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True) #, sample_weight = )
    add = tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4]),name='Adam')
    model.compile(optimizer = add, loss = cce, metrics=['accuracy']) 
    
    return model
    