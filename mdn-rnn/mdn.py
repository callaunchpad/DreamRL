# Author: Joey Hejna, Jihan
# Resource: https://github.com/yanji84/keras-mdn/blob/master/mdn.py

import numpy as np
import keras
from keras import backend as K
<<<<<<< HEAD
from keras import layers
from keras.models import Model, Sequential
=======
from keras.layers import Layer, Dense, Dropout, Activation, LSTM
from keras.activations import softmax, tanh
from keras.models import Sequential
>>>>>>> 1788777564a337a6cb290e589425416d24c9a8ad

## CONSTANTS
logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))

class MDN(layers.Layer):
	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		super(MDN, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.kernel = self.add_weight(name='kernel', 
									shape=(input_shape[1], self.output_dim),
									initializer='truncated_normal',
									trainable=True)
		self.bias = self.add_weight(name='kernel', 
									shape=(self.output_dim,),
									initializer='zeros',
									trainable=True)

		super(MDN, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		output = K.dot(x, self.kernel)
		output = K.bias_add(output, self.bias)
		return output

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)

'''
def get_mdn_coef(output):
      logmix, mean, logstd = tf.split(output, 3, 1)
      logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
      return logmix, mean, logstd
'''

class MDNRNN():
	def __init__(self, hyperparameters):
		self.hps = hyperparameters
		self.build_model()

	def build_model(self):
		# Batch dimension????
		self.input = layers.Input(shape=(self.hps['max_seq_len'], self.hps['in_width']), dtype='float32')
		# hidden_state, self.last_state
		rnn_out, self.hidden_state, self.cell_state = layers.LSTM(self.hps['rnn_size'], return_sequences=True, return_state=True, dropout=0.0, recurrent_dropout=0.0)(self.input)
		self.output = layers.TimeDistributed(MDN(self.hps['out_width'] * self.hps['kmix'] * 3))(rnn_out)
		self.model = Model(self.input, self.output)
		print("Input:", self.model.input)
		print("Output:", self.model.output)
		self.model.compile(optimizer='adam', loss=MDNRNN.mdn_loss())
		
	def train(self, x, y):

		self.model.fit(x, y, batch_size=self.hps['batch_size'], validation_split=self.hps['validation_split'])

	def test(self, x, y):
		loss, acc = self.model.evaluate(x, y, batch_size=self.hps[batch_size])

	def get_mdn_coef(output):
		# first column is the batch dimension
		assert output.shape[2] % 3 == 0
		print(' in get mdn coef')
		num_components = int(int(output.shape[2]) / 3)
		
		logmix = output[:, :, :num_components]
		mean = output[:, :, num_components: 2*num_components]
		logstd = output[:, :, 2*num_components:]

		logmix = logmix - K.logsumexp(logmix, axis=2, keepdims=True)

		return logmix, mean, logstd

	def lognormal(y, mean, logstd):

		return -0.5 * ((y - mean) / K.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

	def get_lossfunc(logmix, mean, logstd, y):
		print(' in get loss')

		v = logmix + MDNRNN.lognormal(y, mean, logstd)
		print('log normal shape', v.shape)
		v = K.logsumexp(v, 2, keepdims=True)
		print('log normal shape', v.shape)

		out = -K.mean(v)
		return out #perhaps axis=1 and keepdims?

	def mdn_loss():
		def loss(y, output):
			print("OUTPUT SHAPE", output.shape)
			logmix, mean, logstd = MDNRNN.get_mdn_coef(output)
			print('IN LOSS - mix, mean, std:', logmix.shape, mean.shape, logstd.shape)
			return MDNRNN.get_lossfunc(logmix, mean, logstd, y)
		return loss

def main():
	# TODO: Write MDN test here
	hps = {}
	hps['batch_size'] = 7
	hps['max_seq_len'] = 16
	hps['in_width'] = 8 # latent + action
	hps['out_width'] = 6 # Latent
	hps['rnn_size'] = 10
	hps['kmix'] = 3
	hps['validation_split'] = 0.1

	mdnrnn = MDNRNN(hps)
	print("####################################")
	print("FINISHED BUILD")
	# X Size = (DIMS, max seq, in_width)
	X = np.random.normal(size=(1000, hps['max_seq_len'], hps['in_width']))
	
	Y = np.random.normal(size=(1000, hps['max_seq_len'], hps['out_width']))

	mdnrnn.train(X, Y)


if __name__ == "__main__":
    main()
