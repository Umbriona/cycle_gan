import os
import io
import tensorflow as tf
from tensorflow.keras import backend as K
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def coef_det_k(y_true, y_pred):
    """Computer coefficient of determination R^2
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

class KLMonitor():
    """Callback that calculates residue frequency from transformed sequences and computes the KL-divergence with the true residue frequences"""

    def __init__(self, data_a, data_b, vocabulary=[0,1,2,3,4], ):
        self.counts_a = data_a.shape[0]
        self.counts_b = data_b.shape[0]
        self.dist_a = np.zeros((len(vocabulary),), dtype=np.float32)
        self.dist_b = np.zeros((len(vocabulary),), dtype=np.float32)
        self.vocabulary = vocabulary
    
        for i in vocabulary:
            self.dist_a[i] = np.sum(data_a==i)
        self.dist_a /= data_a.size
        for i in vocabulary:
            self.dist_b[i] = np.sum(data_b==i)
        self.dist_b /= data_b.size
        
        self.kl = tf.keras.metrics.KLDivergence(name='kullback_leibler_divergence', dtype=None)

    def __call__(self, data_a, data_b, G, F):
        dist_trans_x = np.zeros((len(self.vocabulary),), dtype=np.float32)
        dist_trans_y = np.zeros((len(self.vocabulary),), dtype=np.float32)
        for i, batch in enumerate(zip(data_a, data_b)):
            _, X_bin, W_x= batch[0]
            _, Y_bin, W_y= batch[1]
            y_transform, _ = G(X_bin)
            x_transform, _ = F(Y_bin)
            y_transform = tf.argmax(y_transform).numpy()
            x_transform = tf.argmax(x_transform).numpy()
            for i in self.vocabulary:
                dist_trans_x = np.sum(x_transform==i)
            for i in self.vocabulary:
                dist_trans_y = np.sum(y_transform==i)
                
        dist_trans_x /= self.counts_b
        dist_trans_y /= self.counts_a
        
        
        return self.kl(self.dist_a, dist_trans_x), self.kl(self.dist_b, dist_trans_y)
    
class PCAPlot():
    def __init__(self, data_thermo, data_meso, n_thermo, n_meso, word_length=1, logdir = "log"):
        
        self.file_writer = tf.summary.create_file_writer(os.path.join(logdir,'PCA_plot'))

        
        self.word_length = word_length
        
        self.features_thermo = self.calc_freq(data_thermo, n_thermo)
        self.features_meso   = self.calc_freq(data_meso, n_meso)
        
        self.pca = PCA(n_components=2)
        X= np.concatenate((self.features_thermo, self.features_meso))
        self.pca.fit(X)
        
        self.pc_thermo = self.pca.transform(self.features_thermo)
        self.pc_meso   = self.pca.transform(self.features_meso)
        self.plot_pca(self.pc_thermo, self.pc_meso)
        
             
    def calc_freq(self, data, n_items): 
        
        
        tmp = np.zeros((n_items,20), dtype = np.float32)
        for i, item in enumerate(data):
            seq_len = int(np.sum(item[2]))
            seq = np.argmax(item[1], axis=-1)
            for x in range(seq_len):         
                if seq[x] >= 20:
                    continue
                tmp[i, int(seq[x])] += 1
            tmp[i, :] /= seq_len
        return tmp
    
    def plot2img(self, figure):
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image
    
    def plot_pca(self, pc_gen_thermo, pc_gen_meso):
        
        #concatinate data 
        X_pca = np.concatenate((self.pc_thermo, self.pc_meso, pc_gen_thermo, pc_gen_meso))
        
        idx_thermo     = self.pc_thermo.shape[0]
        idx_meso       = idx_thermo + self.pc_meso.shape[0]
        idx_gen_thermo = idx_meso + pc_gen_thermo.shape[0]
        idx_gen_meso   = idx_gen_thermo + pc_gen_meso.shape[0]
        # color classes
        y = np.ones((idx_gen_meso,))
        y[idx_thermo:idx_meso] = 2
        y[idx_meso:idx_gen_thermo] = 3
        y[idx_gen_thermo:idx_gen_meso] = 4
        
        
        
        fig = plt.figure(figsize=(15,15))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s = 10, cmap ='jet')

        plt.title('PCA plot of {}-gram frequency'.format(self.word_length))
        handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
        legend2 = plt.legend(handles, ['Thermophiles', 'Mesophiles', 'Gen Thermophiles', 'Gen Mesophiles'], loc="upper right", title="Distributions")
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        #plt.show()
        img = self.plot2img(fig)
        return img

    
    
    def __call__(self, gen_thermo, gen_meso, n_thermo, n_meso, step):
        features_gen_thermo = self.calc_freq(gen_thermo, n_meso)
        features_gen_meso   = self.calc_freq(gen_meso, n_thermo)
        
        pc_gen_thermo = self.pca.transform(features_gen_thermo)
        pc_gen_meso   = self.pca.transform(features_gen_meso)
        
        img = self.plot_pca(pc_gen_thermo, pc_gen_meso)
        
        with self.file_writer.as_default():
            tf.summary.image("Training data", img, step=step)
        
        
        