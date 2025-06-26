import numpy as np
import copy
import time
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import os.path
import sys

from numpy.linalg import matrix_rank
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.extmath import softmax
from sklearn.utils import gen_batches

n_samples = 500

class L1():
    __name__ = 'Pruning Filters for Efficient ConvNets'

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

            layer = model.get_layer(index=layer_idx)

            weights = layer.get_weights()# weights have the format: w,h, channel, filters
            score = self.compute_l1(weights)

            output.append((layer_idx, score))

        return output


class klDivergence():
    __name__ = 'Neural Network Pruning with Residual-Connections and Limited-Data'

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

    def zeroed_out(self, model, layer_idx, filter_idx):
        layer = model.get_layer(index=layer_idx)
        w = layer.get_weights()

        # Zeroed out the conv2d filter
        w[0][:, :, :, filter_idx] = np.zeros(w[0].shape[0:-1])
        w[1][filter_idx] = 0
        layer.set_weights(w)

        #Find the index of the BN layer based on layer_idx
        layer_idx = self.bn_idx(model, layer_idx)
        layer = model.get_layer(index=layer_idx)

        #VGG16 on ImageNet224x224 does not contain BN layers.
        if isinstance(layer, BatchNormalization):
            # Zeroed out the batch norm filter
            w = layer.get_weights()
            w[0][filter_idx] = 0
            w[1][filter_idx] = 0
            w[2][filter_idx] = 0
            w[3][filter_idx] = 0
            layer.set_weights(w)

        return None

    def scores(self, model, X_train=None, y_train=None, allowed_layers=[]):
        output = []

        #n_samples = 256 # Original paper subsample is 256
        if n_samples:
            y_ = np.argmax(y_train, axis=1)
            sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in
                            np.unique(y_)]
            sub_sampling = np.array(sub_sampling).reshape(-1)
        else:  # It uses the full training data
            sub_sampling = np.arange(X_train.shape[0])

        p = softmax(model.predict(X_train[sub_sampling], verbose=False))  # Softmax -- Logits
        unchaged_weights = model.get_weights()

        for layer_idx in allowed_layers:

            layer = model.get_layer(index=layer_idx)

            scores = np.zeros((layer.filters))
            for filter_idx in range(0, layer.filters):
                self.zeroed_out(model, layer_idx, filter_idx) #The weights are updated by reference
                q = softmax(model.predict(X_train[sub_sampling], verbose=False))

                # Compute KL Divergence -- See generate_mask.py line 129
                kl_loss = q * (np.log(q) - np.log(p))
                kl_loss = np.sum(kl_loss, axis=1)
                kl_loss = np.mean(kl_loss)
                scores[filter_idx] = kl_loss

                #Restore the original weights (unpruned)
                model.set_weights(unchaged_weights)

            output.append((layer_idx, scores))

        return output

def criteria(method='random'):

    if method == 'L1':
        return L1()

    if method == 'klDivergence':
        return klDivergence()