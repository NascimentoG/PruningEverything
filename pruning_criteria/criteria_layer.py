import numpy as np
from numpy.linalg import matrix_rank
import copy
import time
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.utils import gen_batches
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.extmath import softmax
import gc

n_samples = 500

class klDivergence():
    __name__ = 'Neural Network Pruning with Residual-Connections and Limited-Data, CVPR 2020'
    # Code adapted from https://github.com/Roll920/CURL

    def __init__(self):
        pass

    def bn_idx(self, model, layer_idx):
        idx = -1
        #Looking for the closest relu activatin
        for i in range(layer_idx, len(model.layers)):
            layer = model.get_layer(index=i)
            if isinstance(layer, BatchNormalization):
                idx = i
                break

        return idx

    def zeroed_out(self, model, layer_idx):
        layer = model.get_layer(index=layer_idx)
        w = layer.get_weights()

        # Zeroed out the conv2d filter
        for i in range(0, len(w)):
            w[i] = np.zeros(w[i].shape)

        layer.set_weights(w)

        return None

    def scores(self, model, X_train=None, y_train=None, allowed_layers=[]):
        output = []

        if n_samples:
            y_ = np.argmax(y_train, axis=1)
            sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in
                            np.unique(y_)]
            sub_sampling = np.array(sub_sampling).reshape(-1)
        else:  # It uses the full training data
            sub_sampling = np.arange(X_train.shape[0])

        p = softmax(model.predict(X_train[sub_sampling], verbose=0))  # Softmax -- Logits

        unchaged_weights = model.get_weights()

        for layer_idx in allowed_layers:

            #'Removes' the layers. The weights are updated by reference
            self.zeroed_out(model, layer_idx-1)#i=Add i-1 is the batch index

            q = softmax(model.predict(X_train[sub_sampling], verbose=0))

            # Compute KL Divergence -- See generate_mask.py line 129
            kl_loss = q * (np.log(q) - np.log(p))
            kl_loss = np.sum(kl_loss, axis=1)
            kl_loss = np.mean(kl_loss)

            # Restore the original weights (unpruned)
            model.set_weights(unchaged_weights)

            #print('Layer [{}] Score[{:.4f}]'.format(i, np.mean(kl_loss)), flush=True)
            output.append((layer_idx, kl_loss))

        return output


class L1():
    __name__ = 'Pruning Layers for Efficient ConvNets'

    def __init__(self):
        pass

    def compute_l1(self, weights):
        filter_w, filter_h, n_channels, n_filters =  weights[0].shape[0],  weights[0].shape[1], weights[0].shape[2], weights[0].shape[3]
        l1 = np.zeros((n_filters))
        for channel in range(0, n_channels):
            for filter in range(0, n_filters):
                kernel = weights[0][:, :, channel, filter]
                l1[filter] += np.sum(np.absolute(kernel))

        return l1

    def scores(self, model, X_train=None, y_train=None, allowed_layers=[]):
        output = []
        idx_Conv2D = 0
        for layer_idx in allowed_layers:

            layer = model.get_layer(index=layer_idx-2)
            weights = layer.get_weights()# weights have the format: w,h, channel, filters
            score1 = self.compute_l1(weights)

            layer = model.get_layer(index=layer_idx-5)
            weights = layer.get_weights()# weights have the format: w,h, channel, filters
            score2 = self.compute_l1(weights)
            
            output.append((layer_idx, np.mean(score1) + np.mean(score2)))
        return output

def criteria(method='random'):
    if method == 'klDivergence':
        return klDivergence()
    
    if method == 'L1':
        return L1()