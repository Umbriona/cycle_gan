import os, sys
currentdir = os.path.dirname(os.getcwd())
sys.path.append(currentdir)

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

from utils.preprocessing import prepare_dataset
from utils import models_new
from utils import callbacks
from utils import losses



def load_data(config):
    """ Function to load all the data """
    # Parameters
    file_thermo = config['file_thermo']
    file_meso   = config['file_meso']
    seq_length  = config['seq_length']
    max_samples = config['max_sampels']
    
    thermo_train, thermo_val, n_thermo_train, n_thermo_val = prepare_dataset(file_thermo, 
                                                                             seq_length = seq_length,
                                                                             max_samples = max_samples)
    
    meso_train, meso_val, n_meso_train, n_meso_val = prepare_dataset(file_meso,
                                                                     seq_length = seq_length,
                                                                     max_samples = max_samples)


    return thermo_train, meso_train

def load_models(config):
"""Create all models that is used in cycle gan""" 
    
    model_type = config["Generator"]["type"]
    G_filters = config["Generator"]["filters"]
    G_sizes   = config["Generator"]["kernels"]
    G_dilation= config["Generator"]["dilations"]


    D_filters = config["Discriminator"]["filters"]
    D_sizes   = config["Discriminator"]["kernels"]
    D_dilation= config["Discriminator"]["dilations"]
    D_strides = config["Discriminator"]["strides"]
    
    vocab = config["Vocab_size"] 

    G = models_new.Generator_res(G_filters, G_sizes, G_dilation, vocab)
    F = models_new.Generator_res(G_filters, G_sizes, G_dilation, vocab)
    D_x  = models_new.Discriminator(D_filters, D_sizes, D_strides, D_dilation, vocab)
    D_y  = models_new.Discriminator(D_filters, D_sizes, D_strides, D_dilation, vocab)
    
    return G, F, D_x, D_y

def load_classifier(config):
    vocab         = config['Vocab_size']
    filters       = config['filters']
    kernels       = config['kernels']
    dilations     = config['dilations']
    strides       = config['strides']
    use_attention = config['use_attention']
    file          = config['file']
    
    reg_model = models_new.Classifier(filters, kernels, strides, dilation, vocab)
    reg_model.load_weights(file)
    return reg_model


class CycleGan(tf.keras.Model):

    def __init__(self, config):
        super(CycleGan, self).__init__()
        self.G, self.F, D_x, D_y = load_models(config['CycleGan'])
        self.classifier = load_classifier(config['Classifier'])
        self.lambda_cycle = config['lambda_cycle']
        self.add  = tf.keras.layers.Add()
        
    def compile( self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()
    
    @tf.function
    def train_step(self, batch_data):

        _, X_bin, W_x= batch_data[0]
        _, Y_bin, W_y= batch_data[1]


        with tf.GradientTape(persistent=True) as tape:
            _, X_bin, W_x = batch_data[0]
            _, Y_bin, W_y= batch_data[1]

            fake_y, _ = self.G(X_bin, training=training)
            fake_x, _ = self.F(Y_bin, training=training)

            # Identity mapping
            same_x, _ = self.F(X_bin, training=True)
            same_y, _ = self.G(Y_bin, training=True)

            # Cycle: x -> y -> x
            cycled_x, _ = self.F(fake_y, training=training)
            cycled_y, _ = self.G(fake_x, training=training)

            # Discriminator output
            disc_real_y, _ = self.D_y(Y_bin, training=training)
            disc_fake_y, _ = self.D_y(fake_y, training=training)
            disc_real_x, _ = self.D_x(X_bin, training=training)
            disc_fake_x, _ = self.D_x(fake_x, training=training)


            gen_G_loss = loss_obj.generator_loss_fn(disc_fake_y)
            gen_F_loss = loss_obj.generator_loss_fn(disc_fake_x)

            id_G_loss = loss_obj.cycle_loss_fn(Y_bin, same_y, W_y) 
            id_F_loss = loss_obj.cycle_loss_fn(X_bin, same_x, W_x) 

            gen_cycle_x_loss = loss_obj.cycle_loss_fn(X_bin, cycled_x, W_x) 
            gen_cycle_y_loss = loss_obj.cycle_loss_fn(Y_bin, cycled_y, W_y)



            # Discriminator loss
            tot_loss_G = gen_G_loss + gen_cycle_x_loss * lambda_cycle + id_G_loss * lambda_cycle * lambda_id
            tot_loss_F = gen_F_loss + gen_cycle_y_loss * lambda_cycle + id_F_loss * lambda_cycle * lambda_id

            loss_D_y = loss_obj.discriminator_loss_fn(disc_real_y, disc_fake_y)
            loss_D_x = loss_obj.discriminator_loss_fn(disc_real_x, disc_fake_x)

        grads_G = tape.gradient(tot_loss_G, self.G.trainable_variables)

        grads_F = tape.gradient(tot_loss_F, self.F.trainable_variables)

        # Get the gradients for the discriminators
        grads_disc_y = tape.gradient(loss_D_y, self.D_y.trainable_variables)
        grads_disc_x = tape.gradient(loss_D_x, self.D_x.trainable_variables)

        # Update the weights of the generators 
        self.gen_G_optimizer.apply_gradients(zip(grads_G, self.G.trainable_variables))  
        self.gen_F_optimizer.apply_gradients(zip(grads_F, self.F.trainable_variables))

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
        }, logits
    
    @tf.function
    def validate_step(self, batch_data, step):
        # PCA clustering to measure diversity


        W_y = np.zeros((n_meso_val,512)) #TODO
        W_x = np.zeros((n_thermo_val,512)) #TODO
        gen_y = np.zeros((n_thermo_val,512))
        gen_x = np.zeros((n_meso_val,512))


        for k, item in enumerate(val_x):
            _, X_bin, w_x = item    
            logits, _ = self.G(X_bin)
            tmp = tf.math.argmax(logits, axis = -1).numpy()
            gen_y[k,:] = tmp    
            W_x[k,:] = w_x.numpy()


        for k, item in enumerate(val_y):
            _, Y_bin, w_y = item    
            logits, _ = self.F(Y_bin)
            tmp = tf.math.argmax(logits, axis = -1).numpy()
            gen_x[k,:] = tmp
            W_y[k,:] = w_y.numpy()


        df_gen_y = zip(list(gen_y), list(W_x), list(W_x))
        df_gen_x = zip(list(gen_x), list(W_y), list(W_y)) 

        pcaobj(df_gen_y, df_gen_x, n_thermo_val, n_meso_val, step=step)

        # Get temp dif
        for k, item in enumerate(batches_x):
            _, X_bin, W_x = item 
            logit, _ = self.G(X_bin)
            W_x = tf.reshape(W_x, shape=(32,512,1))
            W_x = tf.repeat(W_x, repeats=21, axis=2)
            trans_x = tf.math.multiply(W_x, logit)
            logits_real  = self.classifier(X_bin)
            logits_trans = self.classifier(trans_x)
            diff = tf.math.reduce_mean(tf.math.subtract(logits_trans,logits_real))
            temp_diff_x(diff)
        for k, item in enumerate(batches_y):
            _, Y_bin, W_y = item 
            logit, _ = self.F(Y_bin)
            W_y = tf.reshape(W_y, shape=(32,512,1))
            W_y = tf.repeat(W_y, repeats=21, axis=2)
            trans_y = tf.math.multiply(W_y, logit)
            logits_real  = self.classifier(Y_bin)
            logits_trans = self.classifier(trans_y)
            diff = tf.math.reduce_mean(tf.math.subtract(logits_trans,logits_real))
            temp_diff_y(diff)

        return 

def train(config):
    
    history = {
        "Gen_G_loss": [],
        "Cycle_X_loss": [],
        "Disc_X_loss": [],
        "Gen_F_loss": [],
        "Cycle_Y_loss": [],
        "Disc_Y_loss": [],
        "x_acc":[],
        "y_acc":[],
        "x_c_acc":[],
        "y_c_acc":[]
    }

    for i in range(400):
    batches_x = meso_train.shuffle(buffer_size = 40000).batch(32, drop_remainder=True) 
    batches_y = thermo_train.shuffle(buffer_size = 40000).batch(32, drop_remainder=True)
    for step, x in enumerate(zip(batches_x,batches_y)):
    
        losses_, logits = train_step(model = cycle_gan_model, batch_data = x,
                            loss_obj = loss_obj,
                            optimizers = [G_opt, F_opt, Dx_opt, Dy_opt],
                            lambda_cycle = 10,
                            lambda_id = 0.2)

        train_loss_G(losses_["Gen_G_loss"]) 
        train_loss_cycle_x(losses_["Cycle_X_loss"])
        train_loss_disc_y(losses_["Disc_X_loss"])
        train_loss_F(losses_["Gen_F_loss"]) 
        train_loss_cycle_y(losses_["Cycle_Y_loss"])
        train_loss_disc_x(losses_["Disc_Y_loss"])
        
        train_acc_x(x[0][1], logits[0][0], x[0][2])
        train_acc_y(x[1][1], logits[0][1], x[1][2])
        train_cycled_acc_x(x[0][1], logits[1][0], x[0][2])
        train_cycled_acc_y(x[1][1], logits[1][1], x[1][2])
        
    
    if i % 2 == 0:
        val_x = thermo_val.shuffle(buffer_size = 40000).batch(1, drop_remainder=False)
        val_y = meso_val.shuffle(buffer_size = 40000).batch(1, drop_remainder=False)
        
        
        with temp_diff_summary_x.as_default():
            tf.summary.scalar('temp_diff', temp_diff_x.result(), step=i, description = 'temp_diff_x')
        with temp_diff_summary_y.as_default():
            tf.summary.scalar('temp_diff', temp_diff_y.result(), step=i, description = 'temp_diff_y')
        temp_diff_x.reset_states()
        temp_diff_y.reset_states()
           
        
    print("Epoch: %d Loss_G: %2.4f Loss_F: %2.4f Loss_cycle_X: %2.4f Loss_cycle_Y: %2.4f Loss_D_Y: %2.4f Loss_D_X %2.4f" % 
          (i, float(train_loss_G.result()), float(train_loss_F.result()), float(train_loss_cycle_x.result()), float(train_loss_cycle_y.result()), float(train_loss_disc_y.result()), float(train_loss_disc_x.result())))
    print("Epoch: %d acc trans x: %2.4f acc trans y: %2.4f acc cycled x : %2.4f acc cycled y: %2.4f" % 
          (i, train_acc_x.result(), train_acc_y.result(), train_cycled_acc_x.result(), train_cycled_acc_y.result()))
    
    # Write log file
    with G_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss_G.result(), step = i, description = 'X transform')
            tf.summary.scalar('acc', train_acc_x.result(), step = i, description = 'X transform' )
            
            
    with F_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss_F.result(), step = i, description = 'Y transform')
        tf.summary.scalar('acc', train_acc_y.result(), step = i, description = 'Y transform' )
                  
    with D_x_summary_writer.as_default():         
        tf.summary.scalar('loss', train_loss_disc_y.result(), step = i, description = 'X discriminator')        
    with D_y_summary_writer.as_default():        
        tf.summary.scalar('loss', train_loss_disc_x.result(), step = i, description = 'Y discriminator')    
    with X_c_summary_writer.as_default(): 
        tf.summary.scalar('loss', train_loss_cycle_x.result(), step = i, description = 'X cycle')
        tf.summary.scalar('acc', train_cycled_acc_x.result(), step = i, description = 'X cycle' )         
    with Y_c_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss_cycle_y.result(), step = i, description = 'Y cycle')
        tf.summary.scalar('acc', train_cycled_acc_y.result(), step = i, description = 'Y cycle' )
        
    # Save history object
    history["Gen_G_loss"].append(train_loss_G.result())
    history["Cycle_X_loss"].append(train_loss_cycle_x.result())
    history["Disc_X_loss"].append(train_loss_disk_x.result())
    history["Gen_F_loss"].append(train_loss_F.result())
    history["Cycle_Y_loss"].append(train_loss_cycle_y.result())
    history["Disc_Y_loss"].append(train_loss_disk_y.result())
    history["x_acc"].append(train_acc_x.result())
    history["x_c_acc"].append(train_cycled_acc_x.result())
    history["y_acc"].append(train_acc_y.result())
    history["y_c_acc"].append(train_cycled_acc_y.result())

    # Reset states
    train_loss_G.reset_states()
    train_loss_cycle_x.reset_states()
    train_loss_disc_y.reset_states()
    train_loss_F.reset_states() 
    train_loss_cycle_y.reset_states()
    train_loss_disc_x.reset_states()
    
    train_acc_x.reset_states()
    train_acc_y.reset_states()
    train_cycled_acc_x.reset_states()
    train_cycled_acc_y.reset_states()
    
    return 

def load_classifier(config):
    """ Load the clasifier that will estimate the temperature difference"""
    
    
    return 
def main():
    cycle_gan_model = MyModel(G,F, D_x, D_y, lambda_cycle=10)

    loss_obj = losses.HingeLoss()

    # Callbacks

    train_loss_G       = tf.keras.metrics.Mean('loss_G', dtype=tf.float32)
    train_loss_cycle_x = tf.keras.metrics.Mean('loss_cycle_x', dtype=tf.float32)
    train_loss_disc_y  = tf.keras.metrics.Mean('loss_disc_y', dtype=tf.float32)
    train_loss_F       = tf.keras.metrics.Mean('loss_F', dtype=tf.float32)
    train_loss_cycle_y = tf.keras.metrics.Mean('loss_cycle_y', dtype=tf.float32)
    train_loss_disc_x  = tf.keras.metrics.Mean('loss_disc_x', dtype=tf.float32)

    temp_diff_x  = tf.keras.metrics.Mean('temp_diff_x', dtype=tf.float32)
    temp_diff_y  = tf.keras.metrics.Mean('temp_diff_y', dtype=tf.float32)

    train_acc_x        = tf.keras.metrics.CategoricalAccuracy()
    train_cycled_acc_x = tf.keras.metrics.CategoricalAccuracy()
    train_acc_y        = tf.keras.metrics.CategoricalAccuracy()
    train_cycled_acc_y = tf.keras.metrics.CategoricalAccuracy()

    # Optimizers
    G_opt=keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5)
    F_opt=keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5)
    Dx_opt=keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5)
    Dy_opt=keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5)

    #file writer

    base_dir = 'log/exp_all3'
    G_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'G'))
    F_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'F'))

    D_x_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'D_x'))
    D_y_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'D_y'))

    X_c_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'X_c'))
    Y_c_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'Y_c'))

    temp_diff_summary_x = tf.summary.create_file_writer(os.path.join(base_dir,'temp_diff_x'))
    temp_diff_summary_y = tf.summary.create_file_writer(os.path.join(base_dir,'temp_diff_y'))


    pcaobj = callbacks.PCAPlot(thermo_train.as_numpy_iterator(), meso_train.as_numpy_iterator(), n_thermo_train, n_meso_train, logdir=os.path.join(base_dir,'img')) 



# Training

    
if __name__ == "__main__":
    main()