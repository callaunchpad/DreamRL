# Author: Joey Hejna, Jihan
# Resource: https://github.com/yanji84/keras-mdn/blob/master/mdn.py

import numpy as np
import keras
from keras import backend as K
from keras.layers import Layer, Dense, Dropout, Activation
from keras.activations import softmax
from keras.models import Sequential

## CONSTANTS
logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))

class MDN(Layer):

    def __init__(self, num_mix, output_dim, **kwargs):
        self.output_dim = output_dim
        self.num_mix = num_mix
        self.mdn_layer_mus = Dense(self.num_mix * self.output_dim, name='mdn_mus')  
        self.mdn_layer_sigmas = Dense(self.num_mix * self.output_dim, activation=exp, name='mdn_sigmas')  # mix*output vals exp activation
        self.mdn_layer_pi = Dense(self.num_mix, activation = softmax, name='mdn_pi')  
        super(MDN, self).__init__(**kwargs)

    def model(self, input_shape, layer_params):
        # initialize empty model
        self.model = Sequential()

        # add dense layers
        cur_shape = input_shape[1]
        for layer_dense in layer_params['dense']:
            self.model.add(Dense(layer_dense['neurons'],
                           batch_input_shape=(None, cur_shape),
                           activation=layer_dense['activation']))
            cur_shape = layer_dense['neurons']

            if layer_dense['dropout'] > 0:
                self.model.add(Dropout(layer_dense['dropout']))

        # add RNN layers



    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                    shape=(input_shape[1], self.output_dim),
                                    initializer='truncated_normal',
                                    trainable=True)
        self.bias = self.add_weight(name='kernel', 
                                    shape=(input_shape[1], self.output_dim),
                                    initializer='zeros',
                                    trainable=True)

        super(MDN, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        output = K.dot(x, self.kernel)
        output = K.add_bias(output, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

'''
def get_mdn_coef(output):
      logmix, mean, logstd = tf.split(output, 3, 1)
      logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
      return logmix, mean, logstd
'''

def get_mdn_coef(output):
    # first column is the batch dimension
    assert output.shape[0] % 3 == 0

    num_components = output.shape[1] / 3
    
    logmix = output[:, :num_components]
    mean = output[:,num_components: 2*num_components]
    logstd = output[:, 2*num_components:]

    logmix = logmix - K.logsumexp(logmix, axis=1, keepdims=True)

    return logmix, mean, logstd

def lognormal(y, mean, logstd):
    return -0.5 * ((y - mean) / K.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

def get_lossfunc(logmix, mean, logstd, y):
    v = logmix + lognormal(y, mean, logstd)
    v = K.logsumexp(v, 1, keepdims=True)
    return -K.mean(v) #perhaps axis=1 and keepdims?

def mdn_loss():
    def loss(y, output):
        logmix, mean, logstd = get_mdn_coef(output)
        return get_lossfunc(logmix, mean, logstd, y)
    return loss

def exp(x):
    return e ** x

def main():
    # TODO: Write MDN test here
    model = MDN(100, 100)
    return

if __name__ == "__main__":
    main()
