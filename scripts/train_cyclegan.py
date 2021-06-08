import os, sys
currentdir = os.path.dirname(os.getcwd())
sys.path.append(currentdir)

import argparse
import yaml
import datetime
import pandas as pd

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


parser = argparse.ArgumentParser(""" """)

parser.add_argument('-c', '--config', type=str, default = 'config.yaml',
                   help = 'Configuration file that configures all parameters')

parser.add_argument('-v', '--verbose', action="store_true",
                   help = "Verbosity")
parser.add_argument('-g', '--gpu', type=str, choices=['0', '1','0,1'], default='0,1')

args = parser.parse_args()

def load_data(config):
    """ Function to load all the data """
    # Parameters
    file_thermo = config['file_thermo']
    file_meso   = config['file_meso']
    seq_length  = config['seq_length']
    max_samples = config['max_samples']
    
    thermo_train, thermo_val, n_thermo_train, n_thermo_val = prepare_dataset(file_thermo, 
                                                                             seq_length = seq_length,
                                                                             max_samples = max_samples)
    
    meso_train, meso_val, n_meso_train, n_meso_val = prepare_dataset(file_meso,
                                                                     seq_length = seq_length,
                                                                     max_samples = max_samples)

    data = {'thermo_train': thermo_train,
            'meso_train': meso_train,
            'thermo_val': thermo_val,
            'meso_val': meso_val,
            'n_thermo_train': n_thermo_train,
            'n_meso_train': n_meso_train,
            'n_thermo_val': n_thermo_val,
            'n_meso_val': n_meso_val}
    
    return data

def load_models(config):
    """Create all models that is used in cycle gan""" 
    
    model_type = config["Generator"]["type"]
    G_filters = config["Generator"]["filters"]
    G_sizes   = config["Generator"]["kernels"]
    G_dilation= config["Generator"]["dilations"]
    G_gumbel = config["Generator"]["use_gumbel"]
    G_temperature = config["Generator"]["temperature"]


    D_filters = config["Discriminator"]["filters"]
    D_sizes   = config["Discriminator"]["kernels"]
    D_dilation= config["Discriminator"]["dilations"]
    D_strides = config["Discriminator"]["strides"]
    
    if config["Losses"]["loss"] == 'Non-Reducing':
        D_activation = 'sigmoid'
    else:
        D_activation = 'linear'
    
    vocab = config["Vocab_size"] 

    G    = models_new.Generator_res(G_filters, G_sizes, G_dilation, vocab, use_gumbel = G_gumbel, temperature = G_temperature)
    F    = models_new.Generator_res(G_filters, G_sizes, G_dilation, vocab, use_gumbel = G_gumbel, temperature = G_temperature) 
    D_x  = models_new.Discriminator(D_filters, D_sizes, D_strides, D_dilation, vocab, activation = D_activation)
    D_y  = models_new.Discriminator(D_filters, D_sizes, D_strides, D_dilation, vocab, activation = D_activation)
    
    return G, F, D_x, D_y

def load_classifier(config):
    vocab         = config['Vocab_size']
    filters       = config['filters']
    kernels       = config['kernels']
    dilations     = config['dilations']
    strides       = config['strides']
    use_attention = config['use_attention']
    file          = config['file']
    
    reg_model = models_new.Classifier(filters, kernels, strides, dilations, vocab)
    reg_model.load_weights(file)
    return reg_model

def load_losses(config):
    if config['loss'] == 'Non-Reduceing':
        loss_obj = losses.NonReduceingLoss()
    elif config['loss'] == 'Wasserstein':
        loss_obj = losses.WassersteinLoss()
    elif config['loss'] == 'Hinge':
        loss_obj = losses.HingeLoss()
    else:
        loss_obj = losses.NonReduceingLoss()
    return loss_obj

def load_optimizers(config):
    lr_D   = config['learning_rate_discriminator']
    lr_G   = config['learning_rate_generator']
    beta_D = config['beta_1_discriminator']
    beta_G = config['beta_1_generator']
    optimizers = {}
    if config['optimizer_discriminator'] == 'Adam':
        optimizers['opt_D_x'] = keras.optimizers.Adam(learning_rate = lr_D, beta_1 = beta_D) 
        optimizers['opt_D_y'] = keras.optimizers.Adam(learning_rate = lr_D, beta_1 = beta_D)
    else:
        optimizers['opt_D_x'] = keras.optimizers.SGD(learning_rate = lr_D, momentum = beta_D) 
        optimizers['opt_D_y'] = keras.optimizers.SGD(learning_rate = lr_D, momentum = beta_D)
        
    if config['optimizer_generator'] == 'Adam':
        optimizers['opt_G'] = keras.optimizers.Adam(learning_rate = lr_G, beta_1 = beta_G) 
        optimizers['opt_F'] = keras.optimizers.Adam(learning_rate = lr_G, beta_1 = beta_G)
    else: 
        optimizers['opt_G'] = keras.optimizers.SGD(learning_rate = lr_G, momentum = beta_G) 
        optimizers['opt_F'] = keras.optimizers.SGD(learning_rate = lr_G, momentum = beta_G)
        
    return optimizers

def load_metrics(config):
    metrics = {}
    metrics['loss_G']       = tf.keras.metrics.Mean('loss_G', dtype=tf.float32)
    metrics['loss_cycle_x'] = tf.keras.metrics.Mean('loss_cycle_x', dtype=tf.float32)
    metrics['loss_disc_y']  = tf.keras.metrics.Mean('loss_disc_y', dtype=tf.float32)
    metrics['loss_F']       = tf.keras.metrics.Mean('loss_F', dtype=tf.float32)
    metrics['loss_cycle_y'] = tf.keras.metrics.Mean('loss_cycle_y', dtype=tf.float32)
    metrics['loss_disc_x']  = tf.keras.metrics.Mean('loss_disc_x', dtype=tf.float32)

    metrics['temp_diff_x']  = tf.keras.metrics.Mean('temp_diff_x', dtype=tf.float32)
    metrics['temp_diff_y']  = tf.keras.metrics.Mean('temp_diff_y', dtype=tf.float32)

    metrics['acc_x']        = tf.keras.metrics.CategoricalAccuracy()
    metrics['cycled_acc_x'] = tf.keras.metrics.CategoricalAccuracy()
    metrics['acc_y']        = tf.keras.metrics.CategoricalAccuracy()
    metrics['cycled_acc_y'] = tf.keras.metrics.CategoricalAccuracy()
    
    return metrics

class CycleGan(tf.keras.Model):

    def __init__(self, config, callbacks=None):
        super(CycleGan, self).__init__()
        self.G, self.F, self.D_x, self.D_y = load_models(config['CycleGan'])
        self.classifier = load_classifier(config['Classifier'])
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
    
    @tf.function
    def train_step(self, batch_data):

        _, X_bin, W_x= batch_data[0]
        _, Y_bin, W_y= batch_data[1]


        with tf.GradientTape(persistent=True) as tape:
            _, X_bin, W_x = batch_data[0]
            _, Y_bin, W_y= batch_data[1]

            fake_y, _ = self.G(X_bin, training=True)
            #print(fake_y.numpy()[0,:20,:])
            fake_x, _ = self.F(Y_bin, training=True)
            #print('fake_y', fake_y.numpy()[0,:20,:])
            # Identity mapping
            same_x, _ = self.F(X_bin, training=True)
            same_y, _ = self.G(Y_bin, training=True)

            # Cycle: x -> y -> x
            cycled_x, _ = self.F(fake_y, training=True)
            cycled_y, _ = self.G(fake_x, training=True)

            # Discriminator output
            disc_real_y, _ = self.D_y(Y_bin, training=True)
            disc_fake_y, _ = self.D_y(fake_y, training=True)
            disc_real_x, _ = self.D_x(X_bin, training=True)
            disc_fake_x, _ = self.D_x(fake_x, training=True)


            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)
            #print('Loss G:', gen_G_loss)

            id_G_loss = self.cycle_loss_fn(Y_bin, same_y, W_y) 
            id_F_loss = self.cycle_loss_fn(X_bin, same_x, W_x) 
            #print('Id loss G:', id_G_loss)
            
            gen_cycle_x_loss = self.cycle_loss_fn(X_bin, cycled_x, W_x) 
            gen_cycle_y_loss = self.cycle_loss_fn(Y_bin, cycled_y, W_y)
            #print('C loss G', gen_cycle_x_loss)


            # Discriminator loss
            tot_loss_G = gen_G_loss + gen_cycle_x_loss * self.lambda_cycle + id_G_loss * self.lambda_cycle * self.lambda_id
            tot_loss_F = gen_F_loss + gen_cycle_y_loss * self.lambda_cycle + id_F_loss * self.lambda_cycle * self.lambda_id
            #print('total loss G', tot_loss_G)
            loss_D_y = self.discriminator_loss_fn(disc_real_y, disc_fake_y)
            loss_D_x = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
        
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
            #tmp = tf.math.argmax(logits, axis = -1).numpy()
            gen_y[k,:, :] = logits #tmp    
            W_x[k,:] = w_x.numpy()

        #print(data['n_meso_val'])
        for k, item in enumerate(val_y):
            _, Y_bin, w_y = item    
            logits, _ = self.F(Y_bin)
            #tmp = tf.math.argmax(logits, axis = -1).numpy()
            gen_x[k,:, :] = logits #tmp
            W_y[k,:] = w_y.numpy()


        df_gen_y = zip(list(gen_y), list(gen_y), list(W_x))
        df_gen_x = zip(list(gen_x), list(gen_x), list(W_y)) 

        self.pcaobj(df_gen_y, df_gen_x, data['n_thermo_val'], data['n_meso_val'], step=step)

        # Get temp dif
        diff=0
        for k, item in enumerate(val_x):
            _, X_bin, W_x = item 
            logit, _ = self.G(X_bin)
            logit = tf.math.argmax(logits, axis=-1, output_type=tf.dtypes.int64)
            logit = tf.one_hot(logit, 21, dtype=tf.float32)
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
            logit = tf.math.argmax(logits, axis=-1, output_type=tf.dtypes.int64)
            logit = tf.one_hot(logit, 21, dtype=tf.float32)
            W_y = tf.reshape(W_y, shape=(1,512,1))
            W_y = tf.repeat(W_y, repeats=21, axis=2)
            trans_y = tf.math.multiply(W_y, logit)
            logits_real  = self.classifier(Y_bin)
            logits_trans = self.classifier(trans_y)
            diff += tf.math.reduce_mean(tf.math.subtract(logits_trans,logits_real))
            diff_y = diff/k

        return diff_x, diff_y

def train(config, model, data, time):
    
    #file writers

    base_dir = os.path.join(config['Log']['base_dir'],time)
    G_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'G'))
    F_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'F'))

    D_x_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'D_x'))
    D_y_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'D_y'))

    X_c_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'X_c'))
    Y_c_summary_writer = tf.summary.create_file_writer(os.path.join(base_dir,'Y_c'))

    temp_diff_summary_x = tf.summary.create_file_writer(os.path.join(base_dir,'temp_diff_x'))
    temp_diff_summary_y = tf.summary.create_file_writer(os.path.join(base_dir,'temp_diff_y'))


    
    metrics = load_metrics(config['CycleGan']['Metrics'])
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
        "y_c_acc":[],
        "temp_diff_x":[],
        "temp_diff_y":[]
        
    }
    diff_x = 0
    diff_y = 0
    for epoch in range(config['CycleGan']['epochs']):
        batches_x = data['meso_train'].shuffle(buffer_size = 40000).batch(config['CycleGan']['batch_size'], drop_remainder=True) 
        batches_y = data['thermo_train'].shuffle(buffer_size = 40000).batch(config['CycleGan']['batch_size'], drop_remainder=True)
        
        #Anneal schedule for gumbel
        if config['CycleGan']['Generator']['use_gumbel']:
                model.G.gms.tau = max(0.1, np.exp(-0.01*epoch))
                model.F.gms.tau = max(0.1, np.exp(-0.01*epoch))
                
        for step, x in enumerate(zip(batches_x,batches_y)):
            
            
            

            losses_, logits = model.train_step( batch_data = x)

            metrics['loss_G'](losses_["Gen_G_loss"]) 
            metrics['loss_cycle_x'](losses_["Cycle_X_loss"])
            metrics['loss_disc_y'](losses_["Disc_X_loss"])
            metrics['loss_F'](losses_["Gen_F_loss"]) 
            metrics['loss_cycle_y'](losses_["Cycle_Y_loss"])
            metrics['loss_disc_x'](losses_["Disc_Y_loss"])

            metrics['acc_x'](x[0][1], logits[0][0], x[0][2])
            metrics['acc_y'](x[1][1], logits[0][1], x[1][2])
            metrics['cycled_acc_x'](x[0][1], logits[1][0], x[0][2])
            metrics['cycled_acc_y'](x[1][1], logits[1][1], x[1][2])
        
        
        if epoch % 10 == 0:
            val_x = data['meso_val'].shuffle(buffer_size = 40000).batch(1, drop_remainder=False)
            val_y = data['thermo_val'].shuffle(buffer_size = 40000).batch(1, drop_remainder=False)
            
            diff_x, diff_y = model.validate_step( val_x, val_y,data, epoch)

            with temp_diff_summary_x.as_default():
                tf.summary.scalar('temp_diff', diff_x, step=epoch, description = 'temp_diff_x')
            with temp_diff_summary_y.as_default():
                tf.summary.scalar('temp_diff', diff_y, step=epoch, description = 'temp_diff_y')
            
            model.save_weights(os.path.join(config['Results']['base_dir'],time,'weights','cycle_gan_model_'+str(epoch)))
        if args.verbose:    
            print("Epoch: %d Loss_G: %2.4f Loss_F: %2.4f Loss_cycle_X: %2.4f Loss_cycle_Y: %2.4f Loss_D_Y: %2.4f Loss_D_X %2.4f" % 
              (epoch, float(metrics['loss_G'].result()),
               float(metrics['loss_F'].result()),
               float(metrics['loss_cycle_x'].result()),
               float(metrics['loss_cycle_y'].result()),
               float(metrics['loss_disc_y'].result()),
               float(metrics['loss_disc_x'].result())))
            print("Epoch: %d acc trans x: %2.4f acc trans y: %2.4f acc cycled x : %2.4f acc cycled y: %2.4f" % 
              (epoch, metrics['acc_x'].result(),
               metrics['acc_y'].result(),
               metrics['cycled_acc_x'].result(),
               metrics['cycled_acc_y'].result()))

        # Write log file
        with G_summary_writer.as_default():
                tf.summary.scalar('loss', metrics['loss_G'].result(), step = epoch, description = 'X transform')
                tf.summary.scalar('acc', metrics['acc_x'].result(), step = epoch, description = 'X transform' )


        with F_summary_writer.as_default():
            tf.summary.scalar('loss', metrics['loss_F'].result(), step = epoch, description = 'Y transform')
            tf.summary.scalar('acc', metrics['acc_y'].result(), step = epoch, description = 'Y transform' )

        with D_x_summary_writer.as_default():         
            tf.summary.scalar('loss', metrics['loss_disc_y'].result(), step = epoch, description = 'X discriminator')        
        with D_y_summary_writer.as_default():        
            tf.summary.scalar('loss', metrics['loss_disc_x'].result(), step = epoch, description = 'Y discriminator')    
        with X_c_summary_writer.as_default(): 
            tf.summary.scalar('loss', metrics['loss_cycle_x'].result(), step = epoch, description = 'X cycle')
            tf.summary.scalar('acc', metrics['cycled_acc_x'].result(), step = epoch, description = 'X cycle' )         
        with Y_c_summary_writer.as_default():
            tf.summary.scalar('loss', metrics['loss_cycle_y'].result(), step = epoch, description = 'Y cycle')
            tf.summary.scalar('acc', metrics['cycled_acc_y'].result(), step = epoch, description = 'Y cycle' )

        # Save history object
        history["Gen_G_loss"].append(metrics['loss_G'].result().numpy())
        history["Cycle_X_loss"].append(metrics['loss_cycle_x'].result().numpy())
        history["Disc_X_loss"].append(metrics['loss_disc_x'].result().numpy())
        history["Gen_F_loss"].append(metrics['loss_F'].result().numpy())
        history["Cycle_Y_loss"].append(metrics['loss_cycle_y'].result().numpy())
        history["Disc_Y_loss"].append(metrics['loss_disc_y'].result().numpy())
        history["x_acc"].append(metrics['acc_x'].result().numpy())
        history["x_c_acc"].append(metrics['cycled_acc_x'].result().numpy())
        history["y_acc"].append(metrics['acc_y'].result().numpy())
        history["y_c_acc"].append(metrics['cycled_acc_y'].result().numpy())
        history["temp_diff_x"].append(diff_x.numpy())
        history["temp_diff_y"].append(diff_y.numpy())
        # Reset states
        metrics['loss_G'].reset_states()
        metrics['loss_cycle_x'].reset_states()
        metrics['loss_disc_y'].reset_states()
        metrics['loss_F'].reset_states() 
        metrics['loss_cycle_y'].reset_states()
        metrics['loss_disc_x'].reset_states()

        metrics['acc_x'].reset_states()
        metrics['acc_y'].reset_states()
        metrics['cycled_acc_x'].reset_states()
        metrics['cycled_acc_y'].reset_states()
    
    return history


def main():
    
    # GPU setting

    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    
    # Get time
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Load configuration file
    with open(args.config, 'r') as file_descriptor:
        config = yaml.load(file_descriptor, Loader=yaml.FullLoader)
        
    with open(args.config, 'r') as file_descriptor:
        config_str = file_descriptor.read()

    # Load training data
    data = load_data(config['Data'])
    
    # Callbacks
    cb = callbacks.PCAPlot(data['thermo_train'].as_numpy_iterator(), data['meso_train'].as_numpy_iterator(), data['n_thermo_train'], data['n_meso_train'], logdir=os.path.join(config['Log']['base_dir'],time,'img')) 
    
    # Initiate model
    model = CycleGan(config, callbacks = cb)
    
    loss_obj  = load_losses(config['CycleGan']['Losses'])
    optimizers = load_optimizers(config['CycleGan']['Optimizers'])
    model.compile(loss_obj, optimizers)
    
    result_dir = os.path.join(config['Results']['base_dir'],time)
    os.mkdir(os.path.join(result_dir))
    os.mkdir(os.path.join(result_dir,'weights'))
    
    # Initiate Training

    history = train(config, model, data, time)
    
    #writing results
    

    # Save model
    model.save_weights(os.path.join(result_dir,'weights','cycle_gan_model'))
    # Write history obj
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(result_dir,'history.csv'))
    # Save config_file
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as file_descriptor:
        file_descriptor.write(config_str)
        
    return 0





# Training

    
if __name__ == "__main__":
    main()
