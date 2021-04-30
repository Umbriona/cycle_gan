import os
import numpy as np
import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LeakyReLU, Embedding, Input, LeakyReLU, LayerNormalization, BatchNormalization, Softmax
from utils.layers_new import SelfAttention, ResMod,ResMod_old, ResModPreAct, Spectral_Norm, GumbelSoftmax 

ATTENTION_FEATURES = 512

class Generator_res(Model):
    def __init__(self, config, vocab):
        super(Generator_res, self).__init__()

        self.n_layers = config['n_layers']
        assert len(config['filters']) >= self.n_layers, "not enough filters specified"
        assert len(config['kernels']) >= self.n_layers, "not enough kernels specified"
        assert len(config['dilations']) >= self.n_layers, "not enough dilations specified"
        
        self.res = [ResMod_old(config['filters'][i],
                           config['kernels'][i],
                           strides=config['strides'][i],
                           dilation = config['dilations'][i],
                           constrains=None) for i in range(self.n_layers)]
                           #l1=config['l1'],
                           #l2=config['l2'],
                           #rate = config['rate']) for i in range(self.n_layers)]
           
        self.atte_loc = config['attention_loc']
        self.use_atte = config['use_attention']
        self.atte = SelfAttention(config['filters'][self.atte_loc])
        self.use_gumbel = config['use_gumbel']
        if self.use_gumbel:
            self.gms = GumbelSoftmax(temperature = 0.5)
        else:
            self.gms = Softmax()
        self.inp = Conv1D(config['filters'][0], config['kernels'][0], padding = 'same', activation = 'relu')
        self.out = Conv1D(vocab, 3, padding = 'same', activation = self.gms)
    def call(self, x):
      #  print('xin', x)
        x = self.inp(x)
      #  print('xinres', x)
        for i in range(self.n_layers):
            x = self.res[i](x)
      #      print('x{}'.format(i), x)
            if self.atte_loc == i and self.use_atte:
                x, a_w = self.atte(x)
                #print('xa', x)
        x = self.out(x)
       # print('x4', x)
        return x, a_w
    
class Discriminator_res(Model):
    def __init__(self, config, vocab):
        super(Discriminator_res, self).__init__()

        self.constraint = Spectral_Norm()
        self.n_layers = config['layers']
        assert len(config['filters']) >= self.n_layers, "not enough filters specified"
        assert len(config['kernels']) >= self.n_layers, "not enough kernels specified"
        assert len(config['dilations']) >= self.n_layers, "not enough dilations specified"
        assert len(config['strides']) >= self.n_layers, "not enough strides specified"
        self.res = [ResMod(config['filters'][i],
                           config['kernels'][i],
                           strides=config['strides'][i],
                           dilation = config['dilations'][i],
                           constrains=None,
                           l1=config['l1'],
                           l2=config['l2'],
                           rate = config['rate']) for i in range(self.n_layers)]
        
        self.atte_loc = config['attention_loc']
        self.use_atte = config['use_attention']
        self.atte = SelfAttention(vocab)
        self.flatten = Flatten()        
        self.out = Dense(1, activation = activation, kernel_constraint=self.constraint)
        
    def call(self, x):
        for i in range(slef.n_layers):
            x = self.res[i](x)
            if self.atte_loc == i and self.use_atte:
                x, a_w = self.atte(x)
        x = self.out(x)
        x = self.flatten(x)
        x = self.out(x)
        return x, a_w
    
class Discriminator(Model):
    def __init__(self, config, vocab, activation = 'sigmoid' ):
        super(Discriminator, self).__init__()
        self.conv=[]
        self.constraint = Spectral_Norm()
        self.n_layers = config['n_layers']
        assert len(config['filters']) >= self.n_layers, "not enough filters specified"
        assert len(config['kernels']) >= self.n_layers, "not enough kernels specified"
        assert len(config['dilations']) >= self.n_layers, "not enough dilations specified"
        assert len(config['strides']) >= self.n_layers, "not enough strides specified"
        self.act = LeakyReLU(0.2)
        for i in range(self.n_layers):
            self.conv.append(Conv1D(config['filters'][i],
                                    config['kernels'][i], 
                                    strides=config['strides'][i],
                                    padding='same',
                                    kernel_constraint = self.constraint,
                                    use_bias = False))
        self.use_atte = config['use_attention']
        self.atte_loc = config['attention_loc']
        self.atte = SelfAttention(config['filters'][self.atte_loc])
        self.out = Conv1D(1, 4, strides=1,
                           activation= activation,
                           padding='same',
                           kernel_constraint = self.constraint,
                           use_bias = False)
        
    def call(self, x):
        for i in range(self.n_layers):
            x = self.conv[i](x)
            if self.atte_loc == i and self.use_atte:
                x, a_w = self.atte(x)
        x = self.out(x)
        return x, a_w
    
class Classifier(Model):
    def __init__(self, config):
        super(Classifier, self).__init__()
        
        self.use_temporal_enc = config['use_global_pos'] 
        self.n_layers = config['n_layers']
        self.constraint = Spectral_Norm()
        self.use_atte = config['use_attention']
        self.atte_loc = config['attention_loc']
        self.batch_size = config['batch_size']
        self.max_length = config['max_length']
        self.model_type = config['model_type']
        
        assert self.model_type in ['ResPreAct', "Res"], "model type {} not implemented needs to be [ResPreAct, Res]".format(self.model_type)  
        
        if self.use_temporal_enc:
            self.get_time()
            self.emb  = Embedding(config['max_length'], config['temporal_encode'])
            
        self.inp = Input((config['max_length'], config['vocab_size']), batch_size=self.batch_size)
        self.conv1 = Conv1D(config['filters'][0], 6, padding= 'same', activation = 'relu')
        self.conv2 = Conv1D(config['filters'][config['n_layers']], 6, padding= 'same', activation = 'relu')
        
        if self.model_type == "Res":
            self.res = [ResMod(config['filters'][i],
                           config['kernels'][i],
                           strides=config['strides'][i],
                           dilation = config['dilations'][i],
                           constrains=None,
                           l1=config['l1'],
                           l2=config['l2'],
                           rate = config['rate']) for i in range(self.n_layers)]
        elif self.model_type == "ResPreAct":
            self.res = [ResMod(config['filters'][i],
                           config['kernels'][i],
                           strides=config['strides'][i],
                           dilation = config['dilations'][i],
                           constrains=None,
                           l1=config['l1'],
                           l2=config['l2'],
                           rate = config['rate']) for i in range(self.n_layers)]
        
        self.atte = SelfAttention(config['filters'][self.atte_loc])    
        self.flatten = Flatten()
        self.out1 = Dense(1, activation = None) #, kernel_constraint=self.constraint)
        

        self.out = self.call(self.inp)                         
        # Reinitial
        super(Classifier, self).__init__(
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
                x, self.a_w = self.atte(x)
                
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.out1(x)
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
    
class CycleGan(tf.keras.Model):

    def __init__(self, config, callbacks=None):
        super(CycleGan, self).__init__()
        self.G, self.F, self.D_x, self.D_y = self.load_models(config['CycleGan'])
        self.classifier = self.load_classifier(config['Classifier'])
        self.lambda_cycle = config['CycleGan']['lambda_cycle']
        self.lambda_id    = config['CycleGan']['lambda_id'] 
        self.add  = tf.keras.layers.Add()
        self.pcaobj = callbacks
    def compile( self, loss_obj, optimizers):
        
        super(CycleGan, self).compile()
        
        self.gen_G_optimizer = optimizers['opt_G']
        self.gen_F_optimizer = optimizers['opt_F']
        self.disc_X_optimizer = optimizers['opt_D_x']
        self.disc_Y_optimizer = optimizers['opt_D_y']
        
        self.generator_loss_fn = loss_obj.generator_loss_fn
        self.discriminator_loss_fn = loss_obj.discriminator_loss_fn
        self.cycle_loss_fn = loss_obj.cycle_loss_fn
        self.identity_loss_fn = loss_obj.cycle_loss_fn
        
    def load_models(self, config):
        """Create all models that is used in cycle gan""" 

        if config["Losses"]["loss"] == 'Non-Reducing':
            D_activation = 'sigmoid'
        else:
            D_activation = 'linear'

        vocab = config["Vocab_size"] 

        G    = Generator_res(config["Generator"], vocab)
        F    = Generator_res(config["Generator"], vocab) 
        D_x  = Discriminator(config["Discriminator"], vocab, activation = D_activation)
        D_y  = Discriminator(config["Discriminator"], vocab, activation = D_activation)

        return G, F, D_x, D_y
    
    def load_classifier(self, config):
        dir_trained_classifiers = config['dir']
        files          = os.listdir(dir_trained_classifiers)
        reg_model = []
        y = []
        inputs = tf.keras.Input(shape=(512,21))
        for index, file in enumerate(files):
            reg_model.append(Classifier(config))
            reg_model[index].load_weights(os.path.join(dir_trained_classifiers,file))
            y.append(reg_model[index](inputs))

        outputs = tf.keras.layers.average([mod for mod in y])
        ensemble_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        return ensemble_model
    
    @tf.function
    def train_step(self, batch_data):

        _, X_bin, W_x= batch_data[0]
        _, Y_bin, W_y= batch_data[1]


        with tf.GradientTape(persistent=True) as tape:
            _, X_bin, W_x = batch_data[0]
            _, Y_bin, W_y= batch_data[1]
            #print("X_bin", X_bin)
            
            fake_y, _ = self.G(X_bin, training=True)
            fake_x, _ = self.F(Y_bin, training=True)
            #print("fake_y", fake_y)
            # Identity mapping
            same_x, _ = self.F(X_bin, training=True)
            same_y, _ = self.G(Y_bin, training=True)
            #print("same_x", same_x)
            # Cycle: x -> y -> x
            cycled_x, _ = self.F(fake_y, training=True)
            cycled_y, _ = self.G(fake_x, training=True)
            #print("cycled_x", cycled_x)
            # Discriminator output
            disc_real_y, _ = self.D_y(Y_bin, training=True)
            disc_fake_y, _ = self.D_y(fake_y, training=True)
            disc_real_x, _ = self.D_x(X_bin, training=True)
            disc_fake_x, _ = self.D_x(fake_x, training=True)


            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)
            #print('Loss G:', gen_G_loss)

            id_G_loss = self.cycle_loss_fn(Y_bin, same_y, W_y) # * self.lambda_cycle * self.lambda_id
            id_F_loss = self.cycle_loss_fn(X_bin, same_x, W_x) # * self.lambda_cycle * self.lambda_id
            #print('Id loss G:', id_G_loss)
            
            gen_cycle_x_loss = self.cycle_loss_fn(X_bin, cycled_x, W_x) # * self.lambda_cycle 
            gen_cycle_y_loss = self.cycle_loss_fn(Y_bin, cycled_y, W_y) # * self.lambda_cycle 
            #print('C loss G', gen_cycle_x_loss)


            # Discriminator loss
            tot_loss_G = gen_G_loss + gen_cycle_x_loss * self.lambda_cycle + id_G_loss * self.lambda_cycle * self.lambda_id
            tot_loss_F = gen_F_loss + gen_cycle_y_loss * self.lambda_cycle + id_F_loss * self.lambda_cycle * self.lambda_id
            #print('total loss G', tot_loss_G)
            loss_D_y = self.discriminator_loss_fn(disc_real_y, disc_fake_y)
            loss_D_x = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            #print('total loss D_X', loss_D_x)
        grads_G_gen = tape.gradient(tot_loss_G, self.G.trainable_variables)
        grads_F_gen = tape.gradient(tot_loss_F, self.F.trainable_variables)
        
        
        #grads_G_gen = tape.gradient(gen_G_loss, self.G.trainable_variables)
        #grads_F_gen = tape.gradient(gen_F_loss, self.F.trainable_variables)
        #grads_G_cyc = tape.gradient(gen_cycle_x_loss, self.G.trainable_variables)
        #grads_F_cyc = tape.gradient(gen_cycle_y_loss, self.F.trainable_variables)
        #grads_G_id = tape.gradient(id_G_loss, self.G.trainable_variables)
        #grads_F_id = tape.gradient(id_F_loss, self.F.trainable_variables)

        # Get the gradients for the discriminators
        grads_disc_y = tape.gradient(loss_D_y, self.D_y.trainable_variables)
        grads_disc_x = tape.gradient(loss_D_x, self.D_x.trainable_variables)

        # Update the weights of the generators 
        self.gen_G_optimizer.apply_gradients(zip(grads_G_gen, self.G.trainable_variables))  
        self.gen_F_optimizer.apply_gradients(zip(grads_F_gen, self.F.trainable_variables))
        
        #self.gen_G_optimizer.apply_gradients(zip(grads_G_cyc, self.G.trainable_variables))  
        #self.gen_F_optimizer.apply_gradients(zip(grads_F_cyc, self.F.trainable_variables))
        #self.gen_G_optimizer.apply_gradients(zip(grads_G_id, self.G.trainable_variables))  
        #self.gen_F_optimizer.apply_gradients(zip(grads_F_id, self.F.trainable_variables))

        # Update the weights of the discriminators
        self.disc_Y_optimizer.apply_gradients(zip(grads_disc_y, self.D_y.trainable_variables))
        self.disc_X_optimizer.apply_gradients(zip(grads_disc_x, self.D_x.trainable_variables))

        return {
            "Gen_G_loss": gen_G_loss,
            "Cycle_X_loss": gen_cycle_x_loss,
            "Disc_X_loss": loss_D_x,
            "Gen_F_loss": gen_F_loss,
            "Cycle_Y_loss": gen_cycle_y_loss,
            "Disc_Y_loss": loss_D_y
        }, ((fake_y, fake_x),(cycled_x, cycled_y))
    
    #@tf.function
    def validate_step(self, val_x, val_y,data, step):
        # PCA clustering to measure diversity
        W_x = np.zeros((data['n_meso_val'],512)) #TODO
        W_y = np.zeros((data['n_thermo_val'],512)) #TODO
        gen_x = np.zeros((data['n_thermo_val'],512,21))
        gen_y = np.zeros((data['n_meso_val'],512,21))

        
        for k, item in enumerate(val_x):
            _, X_bin, w_x = item    
            logits, _ = self.G(X_bin)
            tmp = tf.math.argmax(logits, axis = -1).numpy()
            gen_y[k,:,:] = logits.numpy()    
            W_x[k,:] = w_x.numpy()

        #print(data['n_meso_val'])
        for k, item in enumerate(val_y):
            _, Y_bin, w_y = item    
            logits, _ = self.F(Y_bin)
            tmp = tf.math.argmax(logits, axis = -1).numpy()
            gen_x[k,:,:] = logits.numpy()
            W_y[k,:] = w_y.numpy()


        df_gen_y = zip(list(gen_y), gen_y, list(W_x))
        df_gen_x = zip(list(gen_x), gen_x, list(W_y)) 

        self.pcaobj(df_gen_y, df_gen_x, data['n_thermo_val'], data['n_meso_val'], step=step)

        # Get temp dif
        diff=0
        for k, item in enumerate(val_x):
            _, X_bin, W_x = item 
            logit, _ = self.G(X_bin)
            W_x = tf.reshape(W_x, shape=(1,512,1))
            W_x = tf.repeat(W_x, repeats=21, axis=2)
            trans_x = tf.math.multiply(W_x, logit)
            logits_real  = self.classifier(X_bin)
            logits_trans = self.classifier(trans_x)
            diff += tf.math.reduce_mean(tf.math.subtract(logits_trans,logits_real))
            diff_x = diff/k
        diff=0
        for k, item in enumerate(val_y):
            _, Y_bin, W_y = item 
            logit, _ = self.F(Y_bin)
            W_y = tf.reshape(W_y, shape=(1,512,1))
            W_y = tf.repeat(W_y, repeats=21, axis=2)
            trans_y = tf.math.multiply(W_y, logit)
            logits_real  = self.classifier(Y_bin)
            logits_trans = self.classifier(trans_y)
            diff += tf.math.reduce_mean(tf.math.subtract(logits_trans,logits_real))
            diff_y = diff/k

        return diff_x.numpy(), diff_y.numpy()
    
    @tf.function
    def generate_step(self, batch_data):

        
        id_x, X_bin, W_x = batch_data[0]
        id_y, Y_bin, W_y= batch_data[1]

        fake_y, _ = self.G(X_bin, training=True)
        fake_x, _ = self.F(Y_bin, training=True)
            
        return  (fake_y, fake_x), (id_x, id_y), (W_x, W_y)