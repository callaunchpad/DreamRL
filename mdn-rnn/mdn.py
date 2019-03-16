# Author: Joey Hejna, Jihan
# Resource: https://github.com/yanji84/keras-mdn/blob/master/mdn.py

import numpy as np
import keras
from keras import backend as K
from keras.layers import Layer, Dense, Dropout, Activation, LSTM
from keras.activations import softmax, tanh
from keras.models import Sequential

## CONSTANTS
logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
PARAM_DEFAULTS = {
	'rnn': {
		'dropout': 0,
		'neurons': 512,
		'activation' : tanh
	},
	'dense': {
		'dropout': 0,
		'neurons': 256,
		'activation': tanh
	}
}

class MDN(Layer):

	# output_dim should be 3 * num_mix
    def __init__(self, num_mix, output_dim, **kwargs):
    	assert num_mix * 3 == output_dim, "output_dim must be 3x num_mix!"
        self.output_dim = output_dim
        self.num_mix = num_mix
        super(MDN, self).__init__(**kwargs)

    def model(self, input_shape, layer_params):
    	# set default params
    	for layer_type in PARAM_DEFAULTS:
    		for i in range(len(layer_params[layer_type])):
    			for param in PARAM_DEFAULTS[layer_type]:
    				if param not in layer_params[layer_type][i]:
    					layer_params[layer_type][i][param] = PARAM_DEFAULTS[layer_type][param]

    	# assert last dense layer neurons matches w output_dim
    	assert layer_params['dense'][-1]['neurons'] == self.output_dim, 'last dense layer must match up with output_dim!'

        # initialize empty model
        self.model = Sequential()

        # add RNN layers
        for layer_rnn in layer_params['rnn']:
        	self.model.add(LSTM(layer_rnn['neurons'],
        				   activation=layer_rnn['activation']))
        	if layer_rnn['dropout'] > 0:
        		self.model.add(Dropout(layer_rnn['dropout']))

        # add dense layers
        cur_shape = input_shape[1]
        for layer_dense in layer_params['dense']:
            self.model.add(Dense(layer_dense['neurons'],
                           batch_input_shape=(None, cur_shape),
                           activation=layer_dense['activation']))
            cur_shape = layer_dense['neurons']

            if layer_dense['dropout'] > 0:
                self.model.add(Dropout(layer_dense['dropout']))

        return self.model


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
