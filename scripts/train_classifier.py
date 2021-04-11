#! /usr/bin/env python
import os, sys
currentdir = os.getcwd()
updir = os.path.dirname(currentdir)
sys.path.append(updir)

import argparse

import numpy as np
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers


from utils import preprocessing as pre
from utils import models_new
from utils.callbacks import coef_det_k
import pandas as pd
import yaml


parser = argparse.ArgumentParser(""" """)

parser.add_argument('-c', '--config', type=str, default = 'config.yaml',
                   help = 'Configuration file that configures all parameters')
parser.add_argument('-o', '--output', type=str, default = '../weights/classifier1.h5')

parser.add_argument('-v', '--verbose', action="store_true",
                   help = "Verbosity")
parser.add_argument('-g', '--gpu', type=str, choices=['0', '1','0,1'], default='0,1')




def main(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    
    with open(args.config, 'r') as file_descriptor:
        config = yaml.load(file_descriptor, Loader=yaml.FullLoader)
        
        
    
    data_train, data_val = pre.prepare_dataset_reg(config['Data']['dir'], config['Data']['names_reg'],
                               seq_length = config['Data']['max_length'],
                               t_v_split = 0.1,
                               max_samples = config['Data']['max_samples'])
    
    
    opt = keras.optimizers.Adam(learning_rate=config['Classifier']['learning_rate'])
    model = models_new.Classifier(config['Classifier'])
    model.compile(optimizer=opt, loss='mse', metrics=[coef_det_k])
    model.summary()
    
    x_train = data_train.shuffle(buffer_size = 150000, reshuffle_each_iteration=True).batch(config['Classifier']['batch_size'],
                                                             drop_remainder=True) 
    x_val = data_val.shuffle(buffer_size = 150000, reshuffle_each_iteration=True).batch(config['Classifier']['batch_size'],
                                                         drop_remainder=True)
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.000001)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta = 0.01)
    
    history = model.fit(x_train, epochs=config['Classifier']['epochs'], validation_data = x_val, callbacks=[reduce_lr, early_stop])
    
    model.save(args.output)
    df = pd.DataFrame(history.history)
    df.to_csv(args.output[:-2]+'csv')
    
    return 0

if __name__ == "__main__":
    
    args = parser.parse_args()
    main(args)