import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Activation, Embedding, Input, LeakyReLU, LayerNormalization, BatchNormalization, Softmax, Dropout, Concatenate
from utils.layers_new import  SelfAttentionSN, GumbelSoftmax, SpectralNormalization
from utils.layers_residual import ResModPreAct
from utils import models_generator as models_gen
from utils import models_discriminator as models_dis
from utils import preprocessing as pre

class CycleGan(tf.keras.Model):

    def __init__(self, config, callbacks=None, name = "gan"):
        super(CycleGan, self).__init__(name = name)
        self.G, self.F, self.D_x, self.D_y = self.load_models(config['CycleGan'])
        
        # Build models
        inp = Input(shape=(512,21))
        output_G = self.G(inp)
        output_F = self.F(inp)
        output_Dx = self.D_x(inp)
        output_Dy = self.D_y(inp)
        
        # Build summary
        self.G.summary()
        self.F.summary()
        self.D_x.summary()
        self.D_y.summary()
        

        self.lambda_cycle = tf.Variable(config['CycleGan']['lambda_cycle'], dtype=tf.float32, trainable=False)
        self.lambda_id    = tf.Variable(config['CycleGan']['lambda_id'], dtype=tf.float32, trainable=False) 
        self.add  = tf.keras.layers.Add()
        
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

        G    = models_gen.Generator_res(config["Generator"], vocab, name = "Generator_thermo")
        F    = models_gen.Generator_res(config["Generator"], vocab, name = "Generator_meso") 
        D_x  = models_dis.Discriminator(config["Discriminator"], vocab, activation = D_activation, name = "Discriminator_thermo")
        D_y  = models_dis.Discriminator(config["Discriminator"], vocab, activation = D_activation, name = "Discriminator_meso")

        return G, F, D_x, D_y
    
    def gradient_penalty(self, fake_y, Y_bin, fake_x, X_bin):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        alpha_meso = tf.random.normal([32, 1, 1], 0.0, 1.0)
        alpha_thermo = tf.random.normal([32, 1, 1], 0.0, 1.0)
        
        diff_meso = fake_y - Y_bin
        diff_thermo = fake_x - X_bin
        
        interpolated_meso = Y_bin + alpha_meso * diff_meso
        interpolated_thermo = X_bin + alpha_thermo * diff_thermo
        
        # 1. Get the discriminator output for this interpolated image.
        with tf.GradientTape() as gp_tape_y:
            gp_tape_y.watch(interpolated_meso)
            pred_y = self.D_y(interpolated_meso, training=True)
            
        with tf.GradientTape() as gp_tape_x:
            gp_tape_x.watch(interpolated_thermo)
            pred_x = self.D_x(interpolated_thermo, training=True)
            
        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads_y = gp_tape_y.gradient(pred_y, [interpolated_meso])[0]
        grads_x = gp_tape_x.gradient(pred_x, [interpolated_thermo])[0]
        
        # 3. Calculate the norm of the gradients.
        norm_y = tf.sqrt(tf.reduce_sum(tf.square(grads_y), axis=[1, 2]))
        norm_x = tf.sqrt(tf.reduce_sum(tf.square(grads_x), axis=[1, 2]))
        gp_y = tf.reduce_mean((norm_y - 1.0) ** 2)
        gp_x = tf.reduce_mean((norm_x - 1.0) ** 2)
        return gp_y, gp_x
    
    @tf.function
    def train_step(self, batch_data):

        X_bin, _, W_x= batch_data[0]
        Y_bin, _, W_y= batch_data[1]

        with tf.GradientTape(persistent=True) as tape:
            X_bin, _, W_x = batch_data[0]
            Y_bin, _, W_y= batch_data[1]
            #print("X_bin", X_bin)
            
            fake_y = self.G(X_bin, training=True)
            fake_x = self.F(Y_bin, training=True)
            
            #Apply mask
            mask_x = tf.repeat(W_x, 21, axis=-1)
            mask_y = tf.repeat(W_y, 21, axis=-1)
                
            fake_y = tf.math.multiply(fake_y, mask_x)
            fake_x = tf.math.multiply(fake_x, mask_y)
            
            #print("Fake y", fake_y)
            #print("Fake x", fake_x)
            
            # Identity mapping
            same_x = self.F(X_bin, training=True)
            same_y = self.G(Y_bin, training=True)
            #print("same_x", same_x)
            # Cycle: x -> y -> x
            cycled_x = self.F(fake_y, training=True)
            cycled_y = self.G(fake_x, training=True)
            #print("cycled_x", cycled_x)
            
            # Discriminator output
            disc_real_y = self.D_y(Y_bin, training=True)
            disc_fake_y = self.D_y(fake_y, training=True)
            #print("disc_real x", disc_real_x)
            #print("disc_fake x", disc_fake_x)
        
            disc_real_x = self.D_x(X_bin, training=True)
            disc_fake_x = self.D_x(fake_x, training=True)
            #print("disc_real y", disc_real_y)
            #print("disc_fake y", disc_fake_y)

            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)
            #print('Loss G:', gen_G_loss)

            id_G_loss = self.cycle_loss_fn(Y_bin, same_y, W_y)  * self.lambda_cycle * self.lambda_id
            id_F_loss = self.cycle_loss_fn(X_bin, same_x, W_x)  * self.lambda_cycle * self.lambda_id
            #print('Id loss G:', id_G_loss)
            
            gen_cycle_x_loss = self.cycle_loss_fn(X_bin, cycled_x, W_x)  * self.lambda_cycle 
            gen_cycle_y_loss = self.cycle_loss_fn(Y_bin, cycled_y, W_y)  * self.lambda_cycle 
            #print('C loss G', gen_cycle_x_loss)


            # Generator total loss
            tot_loss_G = gen_G_loss  + gen_cycle_x_loss  + id_G_loss 
            tot_loss_F = gen_F_loss  + gen_cycle_y_loss  + id_F_loss 
            #print('total loss G', tot_loss_G)
            
            # Discriminator loss
            gp_y, gp_x = self.gradient_penalty(fake_y, Y_bin, fake_x, X_bin)
            #print("gp_y", gp_y)
            #print("gp_x", gp_x)
            loss_D_y = self.discriminator_loss_fn(disc_real_y, disc_fake_y) + gp_y * 10
            loss_D_x = self.discriminator_loss_fn(disc_real_x, disc_fake_x) + gp_x * 10
            
            #print("disc_real x", disc_real_x)
            #print("disc_fake x", disc_fake_x)
            #print("disc_real y", disc_real_y)
            #print("disc_fake y", disc_fake_y)
            #print('total loss D_X', loss_D_x)
            #print('total loss D_Y', loss_D_y)
            
            
        grads_G_gen = tape.gradient(tot_loss_G, self.G.trainable_variables)
        grads_F_gen = tape.gradient(tot_loss_F, self.F.trainable_variables)
        
        # Get the gradients for the discriminators
        grads_disc_y = tape.gradient(loss_D_y, self.D_y.trainable_variables)
        grads_disc_x = tape.gradient(loss_D_x, self.D_x.trainable_variables)

        # Update the weights of the generators 
        self.gen_G_optimizer.apply_gradients(zip(grads_G_gen, self.G.trainable_variables))  
        self.gen_F_optimizer.apply_gradients(zip(grads_F_gen, self.F.trainable_variables))
        

        # Update the weights of the discriminators
        self.disc_Y_optimizer.apply_gradients(zip(grads_disc_y, self.D_y.trainable_variables))
        self.disc_X_optimizer.apply_gradients(zip(grads_disc_x, self.D_x.trainable_variables))

        return {
            "Gen_G_loss": gen_G_loss,
            "Cycle_X_loss": gen_cycle_x_loss,
            "Id_X_loss": id_G_loss,
            "Disc_X_loss": loss_D_x,
            "Gen_F_loss": gen_F_loss,
            "Cycle_Y_loss": gen_cycle_y_loss,
            "Id_Y_loss": id_F_loss,
            "Disc_Y_loss": loss_D_y
        }, ((fake_y, fake_x),(cycled_x, cycled_y), (same_x, same_y))
    
    @tf.function
    def validate_step(self, batch_data):
        X_bin, _, W_x= batch_data[0]
        Y_bin, _, W_y= batch_data[1]
        
        shape = tf.shape(X_bin)
        
        logit_x = self.G(X_bin)

        W_x = tf.reshape(W_x, shape=(shape[0],shape[1],1))
        W_x = tf.repeat(W_x, repeats=21, axis=2)
        trans_x = tf.math.multiply(W_x, logit_x)
        temp_real_x  = self.classifier(X_bin)
        temp_fake_x = self.classifier(trans_x)
        diff = tf.math.reduce_mean(tf.math.subtract(temp_real_x, temp_fake_x))
        diff_x = diff
        
        logit_y = self.F(Y_bin)
        W_y = tf.reshape(W_y, shape=(shape[0],shape[1],1))
        W_y = tf.repeat(W_y, repeats=21, axis=2)
        trans_y = tf.math.multiply(W_y, logit_y)
        temp_real_y  = self.classifier(Y_bin)
        temp_fake_y = self.classifier(trans_y)
        diff = tf.math.reduce_mean(tf.math.subtract(temp_real_y, temp_fake_y))
        diff_y = diff
        
        return diff_x, diff_y
    
    def validate_step_old(self, val_x, val_y,data, step):
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
    
    #@tf.function
    def generate_step(self, batch_data):

        
        X_bin, _, W_x = batch_data[0]
        Y_bin, _, W_y= batch_data[1]

        fake_y = self.G(X_bin, training=True)
        fake_x = self.F(Y_bin, training=True)
        seqs = []

        for seq, w in zip(list(tf.math.argmax(fake_y,axis=-1).numpy()), list(W_x.numpy())):
                #print("seq", seq)
                #print("mask", w)
                #print("masked seq", seq[w==1])
                seqs.append(pre.convert_table(seq, tf.reshape(w, shape=(512,))))    
        return seqs 
    
    def save_gan(self, file):
        self.D_x.save_weights(file+"_discrim_x.h5")
        self.D_y.save_weights(file+"_discrim_y.h5")
        self.G.save_weights(file+"_generator_G.h5")
        self.F.save_weights(file+"_generator_x.h5")
    
    def load_gan(self, file):
        self.D_x.load_weights(file+"_discrim_x.h5")
        self.D_y.load_weights(file+"_discrim_y.h5")
        self.G.load_weights(file+"_generator_G.h5")
        self.F.load_weights(file+"_generator_x.h5")