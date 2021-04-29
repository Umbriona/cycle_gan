import tensorflow as tf
from tensorflow.keras.losses import Loss, CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.backend import softplus
from tensorflow.keras import backend as K

class WassersteinLoss(Loss):
    def __init__(self, ):
        super(WassersteinLoss, self).__init__()
        self.cross = CategoricalCrossentropy(from_logits=False, 
                                                             label_smoothing=0.0,
                                                             reduction = tf.keras.losses.Reduction.NONE,
                                                             name='cat_cross')

    def cycle_loss_fn(self, real, cycled, w):
        return tf.reduce_mean(self.cross( real, cycled, w), axis = 0)
    
    # Define the loss function for the generators
    def generator_loss_fn(self, fake):
        return -tf.reduce_mean(fake, axis=0)

    # Define the loss function for the discriminators
    def discriminator_loss_fn(self, real, fake):
        real_loss = tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)
        return fake_loss - real_loss
    
class NonReduceingLoss(Loss):
    def __init__(self, ):
        super(NonReduceingLoss, self).__init__()
        self.cross = CategoricalCrossentropy(from_logits=False, 
                                                             label_smoothing=0.0,
                                                             reduction = tf.keras.losses.Reduction.NONE,
                                                             name='cat_cross')
    def cycle_loss_fn(self, real, cycled, w):
        return tf.reduce_mean(self.cross( real, cycled, w), axis = 0)
        #return self.cross( real, cycled, w)
    
    def generator_loss_fn(self, fake):
        return K.mean(K.softplus(-fake), axis=0)
    
    def discriminator_loss_fn(self, real, fake):
        L1 = K.mean(K.softplus(-real), axis=0)
        L2 = K.mean(K.softplus(fake), axis=0)
        return L1 + L2
    
class HingeLoss(Loss):
    def __init__(self ):
        super(HingeLoss,self).__init__()
        self.cross = CategoricalCrossentropy(from_logits=False, 
                                                             label_smoothing=0.0,
                                                             reduction = tf.keras.losses.Reduction.NONE,
                                                             name='cat_cross')

    def cycle_loss_fn(self, real, cycled, w):
        return tf.reduce_mean(self.cross( real, cycled, w), axis = 0)
    
    # Define the loss function for the generators
    def generator_loss_fn(self, fake):
        return -1 * K.mean(fake, axis=0)

    # Define the loss function for the discriminators
    def discriminator_loss_fn(self, real, fake):
        loss = K.mean(K.relu(1. - real),axis=0)
        loss += K.mean(K.relu(1. + fake),axis=0)
        return loss
