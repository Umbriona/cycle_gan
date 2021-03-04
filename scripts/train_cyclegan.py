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

from utils.loaders import load_data, load_optimizers, load_metrics, load_losses
from utils import models_new
from utils import callbacks



parser = argparse.ArgumentParser(""" """)

parser.add_argument('-c', '--config', type=str, default = 'config.yaml',
                   help = 'Configuration file that configures all parameters')

parser.add_argument('-v', '--verbose', action="store_true",
                   help = "Verbosity")
parser.add_argument('-g', '--gpu', type=str, choices=['0', '1','0,1'], default='0,1')

args = parser.parse_args()


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
    diff_x=0
    diff_y=0
    for epoch in range(config['CycleGan']['epochs']):
        batches_x = data['meso_train'].shuffle(buffer_size = 40000).batch(config['CycleGan']['batch_size'], drop_remainder=True) 
        batches_y = data['thermo_train'].shuffle(buffer_size = 40000).batch(config['CycleGan']['batch_size'], drop_remainder=True)
        
        #Anneal schedule for gumbel
        if config['CycleGan']['Generator']['use_gumbel']:
                model.G.gms.tau = max(0.1, np.exp(-0.01*epoch))
                model.G.gms.tau = max(0.1, np.exp(-0.01*epoch))
                
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
        
        

        if epoch % 10 == 0 or epoch == config['CycleGan']['epochs']-1:
            val_x = data['meso_val'].shuffle(buffer_size = 40000).batch(1, drop_remainder=False)
            val_y = data['thermo_val'].shuffle(buffer_size = 40000).batch(1, drop_remainder=False)
            
            diff_x, diff_y = model.validate_step( val_x, val_y,data, epoch)

            with temp_diff_summary_x.as_default():
                tf.summary.scalar('temp_diff', diff_x, step=epoch, description = 'temp_diff_x')
            with temp_diff_summary_y.as_default():
                tf.summary.scalar('temp_diff', diff_y, step=epoch, description = 'temp_diff_y')


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
        history["temp_diff_x"].append(diff_x)
        history["temp_diff_y"].append(diff_y)
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
    model = models_new.CycleGan(config, callbacks = cb)
    
    loss_obj  = load_losses(config['CycleGan']['Losses'])
    optimizers = load_optimizers(config['CycleGan']['Optimizers'])
    model.compile(loss_obj, optimizers)
    
    # Initiate Training

    history = train(config, model, data, time)
    
    #writing results
    
    result_dir = os.path.join(config['Results']['base_dir'],time)
    os.mkdir(os.path.join(result_dir))
    os.mkdir(os.path.join(result_dir,'weights'))
    # Save model
    model.save_weights(os.path.join(result_dir,'weights','cycle_gan_model'))
    # Write history obj
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(result_dir,'history.csv'))
    # Save config_file
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as file_descriptor:
        file_descriptor.write(config_str)
        
    return 0
    
if __name__ == "__main__":
    main()