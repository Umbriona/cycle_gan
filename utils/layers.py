import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer, Conv2DTranspose, Lambda, Dense, LayerNormalization, Dropout, Concatenate, Conv1D, MaxPool1D, BatchNormalization, UpSampling1D, Add
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import L1L2
import tensorflow.keras.backend as K
from tensorflow.keras.activations import relu, elu

VOCAB_SIZE = 21

######################################### U-Net components #############################################

class Embedding_transfer(Layer):
    def __init__(self, embedding_size = 10):
        super(Embedding_transfer,self).__init__()
        self.embedding_size = embedding_size
        assert embedding_size in [ 5,10], "No precomputed weights exists"
        if embedding_size == 5:
            self.pre_comp_emb = '../Embedding/embedd_w_5.npy'
        elif embedding_size == 10:
            self.pre_comp_emb = '../Embedding/embedd_w_10.npy'             
        self.w = tf.Variable(initial_value = np.load(self.pre_comp_emb), trainable= True)
    def call(self,x): 
        x = tf.gather(self.w, x)
    
        return x

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
    
class Angularization(Layer):
    def __init__(self, size_alphabet):
        super(Angularization,self).__init__()
        self.size_alphabet = size_alphabet
        self.angle_matris = tf.Variable(tf.zeros([2,self.size_alphabet]))
        
    def call(self, x):
        #assert self.size_alphabet == tf.shape(x)
        s = tf.math.sin(self.angle_matris)
        c = tf.math.cos(self.angle_matris)
        x = tf.transpose(x, perm=[0,2,1])
        out = tf.math.atan2(tf.matmul(s,x), tf.matmul(c,x))
        out = tf.transpose(out,perm=[0,2,1])
        return out
    
class CoordinalizationCell(Layer):
    def __init__(self, batch_size):
        super(CoordinalizationCell, self).__init__()
        self.state_size = [3,3,3]
        self.batch_size = batch_size
        
        # Bond lengths
        self.bond_CN = 1.32
        self.bond_CaC = 1.52
        self.bond_NCa = 1.45
        self.bond_CO = 1.23
        
        #Bond angles
        self.angle_NCaC = 109.5/180*np.pi
        self.angle_CaCN = 116/180*np.pi
        self.angle_CaCO = 121/180*np.pi
        self.angle_CNCa = 122/180*np.pi
        
        #Bond angle sin
        self.sin_NCaC = tf.math.sin(self.angle_NCaC)
        self.sin_CaCO = tf.math.sin(self.angle_CaCO)
        self.sin_CaCN = tf.math.sin(self.angle_CaCN)
        self.sin_CNCa = tf.math.sin(self.angle_CNCa)
        
        #Bond angle cos
        self.cos_NCaC = tf.math.cos(self.angle_NCaC)
        self.cos_CaCO = tf.math.cos(self.angle_CaCO)
        self.cos_CaCN = tf.math.cos(self.angle_CaCN)
        self.cos_CNCa = tf.math.cos(self.angle_CNCa)
        
        self.D_0_C = tf.reshape(tf.repeat(self.bond_CaC * self.cos_NCaC, self.batch_size), shape=(-1,1))
        self.D_0_O = tf.reshape(tf.repeat(self.bond_CO * self.cos_CaCO, self.batch_size), shape=(-1,1))
        self.D_0_N = tf.reshape(tf.repeat(self.bond_CN * self.cos_CaCN, self.batch_size), shape=(-1,1))
        self.D_0_Ca = tf.reshape(tf.repeat(self.bond_NCa * self.cos_CNCa, self.batch_size), shape=(-1,1))
        
    def call(self, x, state):
        prev_pos_Ca = state[2]
        prev_pos_N = state[1]
        prev_pos_C = state[0]
        
        #print('Ca', prev_pos_Ca)
        #print('N', prev_pos_N)
       

        D_C = tf.keras.layers.concatenate([self.D_0_C,
               tf.reshape(self.bond_CaC*self.sin_NCaC*tf.math.cos(x[:,0]),shape=(-1,1)),
               tf.reshape(self.bond_CaC*self.sin_NCaC*tf.math.sin(x[:,0]),shape=(-1,1))])
        D_C = tf.reshape(D_C, shape=(-1,3, 1))

        D_O = tf.keras.layers.concatenate([self.D_0_O,
               tf.reshape(self.bond_CO * self.sin_CaCO*tf.math.cos(x[:,1] + np.pi),shape=(-1,1)),
               tf.reshape(self.bond_CO * self.sin_CaCO*tf.math.sin(x[:,1] + np.pi),shape=(-1,1))])
        D_O = tf.reshape(D_O, shape=(-1,3,1))
        
        D_N = tf.keras.layers.concatenate([self.D_0_N,
               tf.reshape(self.bond_CN * self.sin_CaCN * tf.math.cos(x[:,1]),shape=(-1,1)),
               tf.reshape(self.bond_CN * self.sin_CaCN * tf.math.sin(x[:,1]),shape=(-1,1))])
        D_N = tf.reshape(D_N, shape=(-1,3,1))
        
        D_Ca = tf.keras.layers.concatenate([self.D_0_Ca,
               tf.reshape(self.bond_NCa * self.sin_CNCa*tf.math.cos(tf.repeat(-np.pi, self.batch_size)),shape=(-1,1)),
               tf.reshape(self.bond_NCa * self.sin_CNCa*tf.math.sin(tf.repeat(-np.pi, self.batch_size)),shape=(-1,1))])
        D_Ca = tf.reshape(D_Ca, shape=(-1,3,1))
        
        # First atom coordinat computation
        bc_C = tf.math.divide(tf.math.subtract(prev_pos_Ca,prev_pos_N), self.bond_NCa)
        nk_C = tf.linalg.normalize(tf.linalg.cross(tf.math.subtract(prev_pos_N,prev_pos_C), bc_C))[0]
        cross_bc_nk_C = tf.linalg.cross(nk_C,bc_C)
        
        
        bc_C = tf.reshape(bc_C, shape=(-1, 1, 3))  
        nk_C = tf.reshape(nk_C, shape = (-1, 1, 3))
        cross_bc_nk_C = tf.reshape(cross_bc_nk_C, shape= (-1, 1, 3))
        
        M_C = tf.reshape(tf.keras.layers.concatenate([bc_C,cross_bc_nk_C, nk_C], axis=1), shape=(-1,3,3))

        tmp = tf.reshape(tf.matmul(M_C,D_C),shape=(-1,3))
        C_C = tf.math.add(tmp, prev_pos_Ca)
        #print('x',x[:,0])
        #print('D_C',D_C)
        
        # Second attom coordinat
        bc_O = tf.math.divide(tf.math.subtract(C_C,prev_pos_Ca),self.bond_CaC)        
        nk_O = tf.linalg.normalize(tf.linalg.cross(prev_pos_Ca-prev_pos_N, bc_O))[0]
        cross_bc_nk_O = tf.linalg.cross(nk_O,bc_O)
        
        bc_O = tf.reshape(bc_O, shape=(-1,1,3))
        nk_O = tf.reshape(nk_O, shape=(-1,1,3))
        cross_bc_nk_O = tf.reshape(cross_bc_nk_O, shape=(-1, 1, 3))
        
        M_O = tf.reshape(tf.keras.layers.concatenate([bc_O, cross_bc_nk_O, nk_O], axis=1), shape=(-1,3,3))
        tmp = tf.reshape(tf.matmul(M_O, D_O), shape=(-1, 3))
        C_O = tf.math.add(tmp, C_C)

        # Third atom
        bc_N = tf.math.divide(tf.math.subtract(C_C,prev_pos_Ca), self.bond_CaC)
        nk_N = tf.linalg.normalize(tf.linalg.cross(prev_pos_Ca-prev_pos_N, bc_N))[0]
        cross_bc_nk_N = tf.linalg.cross(nk_N,bc_N)
        
        bc_N = tf.reshape(bc_N, shape=(-1,1,3))
        nk_N = tf.reshape(nk_N, shape=(-1,1,3))
        cross_bc_nk_N = tf.reshape(cross_bc_nk_N, shape=(-1, 1, 3))
        
        M_N = tf.reshape(tf.keras.layers.concatenate([bc_N, cross_bc_nk_N, nk_N], axis=1), shape=(-1,3,3))
        tmp = tf.reshape(tf.matmul(M_N, D_N), shape=(-1, 3))
        C_N =  tf.math.add(tmp, C_C)

        # Fourth atom
        bc_Ca = tf.math.divide(tf.math.subtract(C_N,C_C),self.bond_CN)
        nk_Ca = tf.linalg.normalize(tf.linalg.cross(C_C-prev_pos_Ca, bc_Ca))[0]
        cross_bc_nk_Ca = tf.linalg.cross(nk_Ca,bc_Ca)
        
        bc_Ca = tf.reshape(bc_Ca, shape=(-1,1,3))
        nk_Ca = tf.reshape(nk_Ca, shape=(-1,1,3))
        cross_bc_nk_Ca = tf.reshape(cross_bc_nk_Ca, shape=(-1, 1, 3))
        
        M_Ca = tf.reshape(tf.keras.layers.concatenate([bc_Ca, cross_bc_nk_Ca, nk_Ca], axis=1), shape=(-1,3,3))
        tmp = tf.reshape(tf.matmul(M_Ca, D_Ca), shape=(-1,3))
        C_Ca =  tf.math.add(tmp, C_N)
        
        output = tf.keras.layers.concatenate([C_C, C_O, C_N, C_Ca])
        state = [C_C, C_N, C_Ca]
        
        return output, state
        
    
class DownSampleMod(Layer):
    def __init__(self, num_filter, size_filter, sampling_stride, use_max_pool=False,
                 pool_size = 2, rate = 0.1, l1 = 0.01, l2 = 0.01, sample=True):
        super(DownSampleMod, self).__init__()
        self.use_max_pool = use_max_pool
        self.bn1   = BatchNormalization()
        self.bn2   = BatchNormalization()
        self.bn3   = BatchNormalization()

        self.reg1 = L1L2(l1=l1, l2=l2)
        self.reg2 = L1L2(l1=l1, l2=l2)
        self.reg3 = L1L2(l1=l1, l2=l2)
        
        self.add = Add()

        self.conv1 = Conv1D(num_filter, size_filter, activation=None, padding='same',
                            kernel_regularizer = self.reg1, bias_regularizer = self.reg1)
        self.conv2 = Conv1D(num_filter, size_filter, activation=None, padding='same',
                            kernel_regularizer = self.reg2, use_bias=False)
        if not self.use_max_pool:
            self.d_sample = Conv1D(num_filter, size_filter, activation=None, padding='same', strides = sampling_stride,
                            kernel_regularizer = self.reg3, use_bias=False)
        else:
            self.d_sample = MaxPool1D(pool_size = pool_size, strides = sampling_stride)
        self.dOut = Dropout(rate)
        self.sample = sample
    
    def call(self, inp, training = None):
        x = relu(self.bn1(self.conv1(inp), training=training))
        out_enc = relu(self.bn2(self.conv2(x), training=training))   
        if self.sample:
            out = elu(self.bn3(self.d_sample(out_enc),training=training))               
        else:
            out = out_enc
        
        return self.dOut(out, training=training), self.dOut(out_enc, training=training)
        
        
class DownSampleMod_res(Layer):
    def __init__(self, num_filter, size_filter, sampling_stride, use_max_pool=False,
                 pool_size = 2, rate = 0.1, l1 = 0.01, l2 = 0.01, sample=True):
        super(DownSampleMod_res, self).__init__()
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

        self.conv1 = Conv1D(num_filter, size_filter, padding='same',
                            kernel_regularizer = self.reg1, use_bias=False)
        self.conv2 = Conv1D(num_filter, size_filter, padding='same',
                            kernel_regularizer = self.reg2, use_bias=False)
        self.conv3 = Conv1D(num_filter, size_filter, padding='same',
                            kernel_regularizer = self.reg3, use_bias=False)
        if not self.use_max_pool:
            self.d_sample = Conv1D(num_filter, size_filter, padding='same', strides = sampling_stride,
                            kernel_regularizer = self.reg4, use_bias=False)
        else:
            self.d_sample = MaxPool1D(pool_size = pool_size, strides = sampling_stride)
        self.dOut = Dropout(rate)
        self.sample = sample
    
    def call(self, inp, training = None):
        x_1 = self.conv1(inp)
        x = relu(self.bn1(x_1, training = training))
        x = relu(self.bn2(self.conv2(x), training=training))
        out_enc = self.conv3(x)
        x = relu(self.bn3(self.add([x_1,out_enc]), training=training))
        if self.sample:
            x = self.d_sample(x)
            x = self.bn4(x, training=training)
            out = relu(x)
        else:
            out = x        
        return self.dOut(out, training=training), self.dOut(out_enc, training=training)
    
class DownSampleMod_ins(Layer):
    def __init__(self, num_filter, size_filter, sampling_stride, use_max_pool=False,
                 pool_size = 2, rate = 0.1, l1 = 0.01, l2 = 0.01, sample=True):
        super(DownSampleMod_ins, self).__init__()
        self.use_max_pool = use_max_pool
        self.bn1   = BatchNormalization()
        self.bn2   = BatchNormalization()
        self.bn3   = BatchNormalization()
        self.bn4   = BatchNormalization()

        self.reg1 = L1L2(l1=l1, l2=l2)
        self.reg2 = L1L2(l1=l1, l2=l2)
        self.reg3 = L1L2(l1=l1, l2=l2)
        self.reg4 = L1L2(l1=l1, l2=l2)
        
        self.concat = Concatenate(axis=2)
        self.add = Add()
        self.conv1 = Conv1D(int(num_filter//1.5), size_filter, padding='same',
                            kernel_regularizer = self.reg1, use_bias=False)
        self.conv2 = Conv1D(num_filter, size_filter, padding='same',
                            kernel_regularizer = self.reg2, use_bias=False)
        self.conv3 = Conv1D(int(num_filter*1.5), size_filter, padding='same',
                            kernel_regularizer = self.reg3, use_bias=False)
        
        self.tot_filter = int(num_filter//1.5+num_filter+1.5*num_filter)
        self.convRes = Conv1D(self.tot_filter, 1, padding='same', use_bias=False)
        
        if not self.use_max_pool:
            self.d_sample = Conv1D(num_filter, size_filter, padding='same', strides = sampling_stride,
                            kernel_regularizer = self.reg4, use_bias=False)
        else:
            self.d_sample = MaxPool1D(pool_size = pool_size, strides = sampling_stride)
        self.dOut = Dropout(rate)
        self.sample = sample
    
    def call(self, inp, training=None):
        x_1 = relu(self.bn1(self.conv1(inp),training=training))
        x_2 = relu(self.bn2(self.conv2(x_1),training=training))
        x_3 = relu(self.bn3(self.conv3(x_2),training=training))
        x_res = self.convRes(inp)
        out_enc = self.add([x_res,self.concat([x_1,x_2,x_3])])
        if self.sample:
            out = relu(self.bn4(self.d_sample(out_enc), training=training))
        else:
            out = out_enc        
        return self.dOut(out, training=training), self.dOut(out_enc,training=training)
        

        
class UpsampleMod(Layer):
    def __init__(self, num_filter, size_filter, sampling_stride, use_regular_uppsampling = False,
                size = 2, rate = 0.1, l1 = 0.01, l2 = 0.01, use_max_pool = False):
        super(UpsampleMod, self).__init__()
        self.use_max_pool = use_max_pool
        self.bn1   = BatchNormalization()
        self.bn2   = BatchNormalization()
        self.bn3   = BatchNormalization()

        self.reg1 = L1L2(l1=l1, l2=l2)
        self.reg2 = L1L2(l1=l1, l2=l2)
        self.reg3 = L1L2(l1=l1, l2=l2)
        
        self.concat = Concatenate(axis=2)

        self.conv1 = Conv1D(num_filter, size_filter, padding='same',
                            kernel_regularizer = self.reg1, use_bias=False)
        self.conv2 = Conv1D(num_filter, size_filter, padding='same',
                            kernel_regularizer = self.reg2, use_bias=False)
        
        if not self.use_max_pool:
            self.u_sample = Conv1DTranspose(num_filter, size_filter, strides = sampling_stride,
                            kernel_regularizer = self.reg3, use_bias=False)
        else:
            self.u_sample = UpSampling1D(size = size)
        self.dOut = Dropout(rate)
    
    def call(self, x, enc, training = None):
        x = self.u_sample(x)
        x = relu(self.bn1(self.concat([x, enc]), training=training))
        x = relu(self.bn2(self.conv1(x), training=training))
        x = relu(self.bn3(self.conv2(x), training=training))
        return self.dOut(x, training=training)
    
class UpsampleMod_res(Layer):
    def __init__(self, num_filter, size_filter, sampling_stride, use_regular_uppsampling = False,
                size = 2, rate = 0.1, l1 = 0.01, l2 = 0.01, use_max_pool = False):
        super(UpsampleMod_res, self).__init__()
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
    
    def call(self, x, enc, training=None):
        x = self.u_sample(x)
        x = relu(self.bn1(self.concat([x, enc]), training=training))
        x_c = self.conv1(x)
        x = relu(self.bn2(x, training=training))
        x = relu(self.bn3(self.conv2(x),training=training))
        x = self.conv3(x)
        x = relu(self.bn4(self.add([x_c, x]),training=training))
        return self.dOut(x, training=training)
    
class UpsampleMod_ins(Layer):
    def __init__(self, num_filter, size_filter, sampling_stride, use_regular_uppsampling = False,
                size = 2, rate = 0.1, l1 = 0.01, l2 = 0.01, use_max_pool = False):
        super(UpsampleMod_ins, self).__init__()
        self.use_max_pool = use_max_pool
        self.bn1   = BatchNormalization()
        self.bn2   = BatchNormalization()
        self.bn3   = BatchNormalization()
        self.bn4   = BatchNormalization()

        self.reg1 = L1L2(l1=l1, l2=l2)
        self.reg2 = L1L2(l1=l1, l2=l2)
        self.reg3 = L1L2(l1=l1, l2=l2)
        self.reg4 = L1L2(l1=l1, l2=l2)
        
        self.concat1 = Concatenate(axis=2)
        self.concat2 = Concatenate(axis=2)
        self.add = Add()

        self.conv1 = Conv1D(int(num_filter//1.5), size_filter, padding='same',
                            kernel_regularizer = self.reg1, use_bias=False)
        self.conv2 = Conv1D(num_filter, size_filter, padding='same',
                            kernel_regularizer = self.reg2, use_bias=False)
        self.conv3 = Conv1D(int(num_filter*1.5), size_filter, padding='same',
                            kernel_regularizer = self.reg3, use_bias=False)
        
        self.tot_filter = int(num_filter//1.5+num_filter+1.5*num_filter)
        self.convRes = Conv1D(self.tot_filter, 1, padding='same', use_bias=False)
        
        if not self.use_max_pool:
            self.u_sample = Conv1DTranspose(num_filter, size_filter, strides = sampling_stride,
                            kernel_regularizer = self.reg4, use_bias=False)
        else:
            self.u_sample = UpSampling1D(size = size)
        self.dOut = Dropout(rate)
    
    def call(self, x, enc, training = None):
        x = relu(self.bn1(self.u_sample(x),training=training))
        x = self.concat1([x, enc])
        x_1 = relu(self.bn2(self.conv1(x),   training=training))
        x_2 = relu(self.bn3(self.conv2(x_1), training=training))
        x_3 = relu(self.bn4(self.conv3(x_2), training=training))
        x_Res = self.convRes(x)
        x = self.add([x_Res,self.concat2([x_1, x_2, x_3])])
        return self.dOut(x, training=training)
    
class UNetModule(Layer):
    def __init__(self, p):
        super(UNetModule, self).__init__()
        self.num_filter = p['num_filter']
        self.size_filter = p['kernel_size']
        self.sampling_stride = p['sampling_stride']
        self.use_max_pool = p['use_max_pool']
        self.pool_size = p['pool_size']
        self.rate = p['rate']
        self.l1 = p['l1']
        self.l2 = p['l2']
        self.len = len(p['sampling_stride'])
        
        
        self.comp_stage = [DownSampleMod(self.num_filter[i], self.size_filter[i], self.sampling_stride[i],
                                            use_max_pool=False, pool_size = self.pool_size[i], rate = self.rate[i],
                                            l1 = self.l1[i], l2 = self.l2[i],
                                            sample = True if i <self.len else False) for i in range(self.len)]
        
        self.exp_stage = [UpsampleMod( self.num_filter[i], self.size_filter[i], self.sampling_stride[i],
                                           use_regular_uppsampling = False, size = self.sampling_stride[i],
                                           rate = self.rate[i], l1 = self.l1[i], l2 = self.l2[i])  for i in range(self.len)]
          

    def call(self, x, training=None):
        tmp = []
        for i in range(self.len):
            x, out_enc = self.comp_stage[i](x, training=training)
            tmp.append(out_enc)

        for i in reversed(range(self.len)):
            x = self.exp_stage[i](x, tmp[i], training=training)
        return(x)
    
class UNetModule_res(Layer):
    def __init__(self, p):
        super(UNetModule_res, self).__init__()
        self.num_filter = p['num_filter']
        self.size_filter = p['kernel_size']
        self.sampling_stride = p['sampling_stride']
        self.use_max_pool = p['use_max_pool']
        self.pool_size = p['pool_size']
        self.rate = p['rate']
        self.l1 = p['l1']
        self.l2 = p['l2']
        self.len = len(p['sampling_stride'])
        
        
        self.comp_stage = [DownSampleMod_res(self.num_filter[i], self.size_filter[i], self.sampling_stride[i],
                                            use_max_pool=False, pool_size = self.pool_size[i], rate = self.rate[i],
                                            l1 = self.l1[i], l2 = self.l2[i],
                                            sample = True if i <self.len else False) for i in range(self.len)]
        
        self.exp_stage = [UpsampleMod_res( self.num_filter[i], self.size_filter[i], self.sampling_stride[i],
                                           use_regular_uppsampling = False, size = self.sampling_stride[i],
                                           rate = self.rate[i], l1 = self.l1[i], l2 = self.l2[i])  for i in range(self.len)]
          

    def call(self, x, training=None):
        tmp = []
        for i in range(self.len):
            x, out_enc = self.comp_stage[i](x, training=training)
            tmp.append(out_enc)
        for i in reversed(range(self.len)):
            x = self.exp_stage[i](x, tmp[i], training=training)
        return(x)
    
class UNetModule_ins(Layer):
    def __init__(self, p):
        super(UNetModule_ins, self).__init__()
        self.num_filter = p['num_filter']
        self.size_filter = p['kernel_size']
        self.sampling_stride = p['sampling_stride']
        self.use_max_pool = p['use_max_pool']
        self.pool_size = p['pool_size']
        self.rate = p['rate']
        self.l1 = p['l1']
        self.l2 = p['l2']
        self.len = len(p['sampling_stride'])
        
        
        self.comp_stage = [DownSampleMod_ins(self.num_filter[i], self.size_filter[i], self.sampling_stride[i],
                                            use_max_pool=False, pool_size = self.pool_size[i], rate = self.rate[i],
                                            l1 = self.l1[i], l2 = self.l2[i],
                                            sample = True if i <self.len else False) for i in range(self.len)]
        
        self.exp_stage = [UpsampleMod_ins( self.num_filter[i], self.size_filter[i], self.sampling_stride[i],
                                           use_regular_uppsampling = False, size = self.sampling_stride[i],
                                           rate = self.rate[i], l1 = self.l1[i], l2 = self.l2[i])  for i in range(self.len)]
          

    def call(self, x, training=None):
        tmp = []
        for i in range(self.len):
            x, out_enc = self.comp_stage[i](x, training=training)
            tmp.append(out_enc)
        for i in reversed(range(self.len)):
            x = self.exp_stage[i](x, tmp[i], training=training)
        return(x)
    
    
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

    
##################################### Transformer Components ############################################
    
class MultiHeadedAttention(Layer):
    def __init__(self, input_dim, num_heads = 1, name = None, *args, **kwargs):
        super(MultiHeadedAttention, self).__init__(name=name)
        self.input_dim = input_dim
        self.heads = num_heads
        if self.input_dim % self.n_heads != 0:
            raise ValueError("input_dim should be divisable num_heads")
        self.proj_dim = self.input_dim // self.num_heads
        self.query_proj = Dense(self.proj_dim)
        self.key_proj = Dense(self.proj_dim)
        self.value_proj = Dense(self.proj_dim)
        self.concat_proj = Dense(self.input_dim)
        
#    def build(self, input_shape):
        
#        self._model = 
    def attention(self, query, key, value):
        attend = tf.matmul(query, key, transpose_b=True)
        scaled_attend = attend / tf.math.sqrt(self.proj_dim)
        weights = tf.nn.softmax(scaled_attend)
        output = tf.matmul(weights,value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x,(batch_size,-1, self.num_heads, self.proj_dim))
        return tf.transpose(x, perm=[0,2,1,3])
    
    def call(self, x1, x2, x3):
        batch_size = tf.shape(x)[0]
        query =self.query_dens(x1)
        key = self.key_dens(x2)
        value = self.value_dense(x3)
        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)  # (batch_size, seq_len, embed_dim)
        return output, weights

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    
class TransformerEncoderModule(Layer):
    def __init__(self, input_dim, num_heads, ff_dim, d_rate=0.1):
        super(TransformerEncoderModule, self).__init__()
        self.attend = MultiHeadedAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def call(self, inputs, training=True):
        attn_output, weights_enc = self.attend(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
class TransformerDencoderModule(Layer):
    def __init__(self, input_dim, num_heads, ff_dim, d_rate=0.1):
        super(TransformerDecoderModule, self).__init__()
        self.attend_1 = MultiHeadedAttention(embed_dim, num_heads)
        self.attend_2 = MultiHeadedAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)
        
    def call(self, target, enc_out, training=True):
        attn_output, weights_dec_1 = self.attend(target, target, target)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(target + attn_output)
        
        attn_output, weights_dec_2 = self.attend(out1, enc_out, enc_out)
        attn_output = self.dropout2(attn_output, training=training)
        out2 = self.layernorm2(target + attn_output)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)
    