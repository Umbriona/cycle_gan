import tensorflow as tf

class Loss_secondary_MSE():
    
    def __init__(self, sample_weight = False, name = 'loss_sec_mse'):
        self.name = name
        self.sample_weight = sample_weight
        
    def __call__(self, y, y_pred, sample_weight = None):
        
        if self.sample_weight:
            assert type(sample_weight) != None, 'sample_weight needs to be supplied'
        
        y = tf.cast(y, dtype = tf.float32)
        loss = tf.math.squared_difference(y, y_pred)
        loss = tf.reduce_mean(loss, axis=-1)
        if self.sample_weight:
            loss = tf.math.multiply(loss,sample_weight)
        loss = tf.reduce_mean(loss, axis=0)
        return loss
    
class Loss_secondary_cross():
    
    def __init__(self, sample_weight = False, name = 'loss_sec_cross'):
        self.name = name
        self.sample_weight = sample_weight
        
    def __call__(self, y, y_pred, sample_weight = None):
        
        if self.sample_weight:
            assert type(sample_weight) != None, 'sample_weight needs to be supplied'
        
        y = tf.cast(y, dtype = tf.float32)
        loss = tf.math.multiply(y, tf.math.log(y_pred+1e-7))                         
        loss = -tf.reduce_sum(loss, axis=-1)
        if self.sample_weight:
            loss = tf.math.multiply(loss,sample_weight)
        loss = tf.reduce_sum(loss, axis=0)
        return loss
    
class Loss_torsion():
    def __init__(self, size, sample_weight = False, name = 'loss_tor'):
        self.name = name
        self.sample_weight = sample_weight
        self.size = size
        self.flatten = tf.keras.layers.Flatten()  
    
    def __call__(self, y_true, y_pred, sample_weight):
        
        if self.sample_weight:
            assert type(sample_weight) != None, 'sample_weight needs to be supplied'

        y_true = tf.transpose(y_true, perm=[0,2,1])
        y_pred = tf.transpose(y_pred, perm=[0,2,1])
        

        y_true_csum = tf.keras.backend.cumsum(y_true, axis=-1)
        y_pred_csum = tf.keras.backend.cumsum(y_pred, axis=-1)

        
        y_true_mat_0 = tf.keras.backend.repeat(y_true_csum[:,0,:], self.size)
        y_true_mat_1 = tf.keras.backend.repeat(y_true_csum[:,1,:], self.size)
        y_pred_mat_0 = tf.keras.backend.repeat(y_pred_csum[:,0,:], self.size)
        y_pred_mat_1 = tf.keras.backend.repeat(y_pred_csum[:,1,:], self.size)
        

        y_pred_mat_0 = tf.transpose(y_pred_mat_0, perm=[0,2,1])
        y_pred_mat_1 = tf.transpose(y_pred_mat_1, perm=[0,2,1])

        vec_0 = tf.math.squared_difference(y_true_mat_0,y_pred_mat_0)
        vec_1 = tf.math.squared_difference(y_true_mat_1,y_pred_mat_1)
        
        dist_mat = tf.math.sqrt(tf.math.add(vec_0, vec_1))
        loss = tf.math.reduce_sum(tf.math.l2_normalize(dist_mat),axis=-1)
        if self.sample_weight:
            loss = tf.math.multiply(loss,sample_weight)
        loss = tf.reduce_mean(loss, axis=0)
        return loss

