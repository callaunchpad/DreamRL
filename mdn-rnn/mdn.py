# Author: Joey Hejna, Jihan
# Resources: https://github.com/yanji84/keras-mdn/blob/master/mdn.py
#            https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/rnn/rnn.py

import numpy as np
import keras
import sys
from keras import backend as K
from keras import layers
from keras.models import Model

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
		# Create Input layer: out shape = (batch, maxlen, latent_size + actions)
		self.input = layers.Input(shape=(self.hps['max_seq_len'], self.hps['in_width']), dtype='float32')
		# Create RNN Cell (with droput!). out shape = (batch, maxlen, RNN size)
		rnn_out, self.hidden_state, self.cell_state = layers.LSTM(self.hps['rnn_size'], return_sequences=True, 
													return_state=True, dropout=self.hps['dropout'], 
													recurrent_dropout=self.hps['recurrent_dropout'])(self.input)
		# Reshape RNN out to distribute sequences over batch dim. out shape = (batch*maxlen, RNN size)
		rnn_out = self.output = layers.Lambda(
								lambda x: K.reshape(x, (-1, self.hps['rnn_size'])), 
								output_shape=(self.hps['rnn_size'],))(rnn_out)
		# Apply MDN to each vector of every sequence seperatly. 
		# Each RNN vector corresponds to out params, each with k gaussians defined by 3 params
		# out shape = (batch * maxlen, out*kmix*3)
		self.output = MDN(self.hps['out_width'] * self.hps['kmix'] * 3)(rnn_out)
		# Again push everything to the batch dimension
		# out shape = (batch * maxlen * out, kmix*3)

		self.output = layers.Lambda(
								lambda x: K.reshape(x, (-1, self.hps['kmix'] * 3)), 
								output_shape=(self.hps['kmix'] * 3,))(self.output)

		self.model = Model(self.input, self.output)
		print("Input:", self.model.input)
		print("Output:", self.model.output)

		self.model.compile(optimizer='adam', loss=MDNRNN.mdn_loss())
	
	def train(self, x, y):

		self.model.fit(x, y, batch_size=self.hps['batch_size'], validation_split=self.hps['validation_split'])

	def test(self, x, y):
		loss = self.model.evaluate(x, y, batch_size=self.hps['batch_size'])
		return loss

	def save(self, path):
		self.model.save_weights(path + ".h5")
		print("Saved!")

	def restore(self, path):
		print("Restoring from " + path)
		self.model.load_weights(path + ".h5")
		print("Restored!")


	def get_mdn_coef(output):
		# first column is the batch dimension
		assert output.shape[1] % 3 == 0
		num_components = int(int(output.shape[1]) / 3)
		
		logmix = output[:, :num_components]
		mean = output[:, num_components: 2*num_components]
		logstd = output[:, 2*num_components:]

		logmix = logmix - K.logsumexp(logmix, axis=1, keepdims=True)

		return logmix, mean, logstd

	def lognormal(y, mean, logstd):
		return -0.5 * ((y - mean) / K.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

	def get_lossfunc(logmix, mean, logstd, y):
		v = logmix + MDNRNN.lognormal(y, mean, logstd)
		v = K.logsumexp(v, 1, keepdims=True)
		out = -K.mean(v)
		return out

	def mdn_loss():
		def loss(y, output):
			# reshape the y vector from (batch, maxlen, out) to (batch * maxlen * out, 1)
			y = K.reshape(y, (-1, 1))
			logmix, mean, logstd = MDNRNN.get_mdn_coef(output)
			return MDNRNN.get_lossfunc(logmix, mean, logstd, y)
		return loss

def main():
	# TODO: Write MDN test here
	hps = {}
	hps['batch_size'] = 100
	hps['max_seq_len'] = 1000
	hps['in_width'] = 35 # latent + action
	hps['out_width'] = 32 # Latent
	hps['rnn_size'] = 256
	hps['kmix'] = 5
	hps['dropout'] = 0.9
	hps['recurrent_dropout'] = 0.9
	hps['validation_split'] = 0.1

	mdnrnn = MDNRNN(hps)
	print("FINISHED BUILD")
	X = np.random.normal(size=(10, hps['max_seq_len'], hps['in_width']))
	Y = np.random.normal(size=(10, hps['max_seq_len'], hps['out_width']))

	mdnrnn.train(X, Y)
	mdnrnn.save("checkpoints/test1")
	print("Loss", mdnrnn.test(X,Y))


if __name__ == "__main__":
    main()
