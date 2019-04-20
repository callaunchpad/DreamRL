# Author: Joey Hejna, Jihan
# Resources: https://github.com/yanji84/keras-mdn/blob/master/mdn.py
#            https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/rnn/rnn.py
#			 https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

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

class MDNRNN():
	def __init__(self, hyperparameters):
		self.hps = hyperparameters
		self.build_model()

	def build_model(self):
		# Create Input layer: out shape = (batch, Length, latent_size + actions)
		self.input = layers.Input(shape=(None, self.hps['in_width']), dtype='float32')
		# Create RNN Cell (with droput!). out shape = (batch, maxlen, RNN size)
		self.lstm = layers.LSTM(self.hps['rnn_size'], return_sequences=True, 
													return_state=True, dropout=self.hps['dropout'], 
													recurrent_dropout=self.hps['recurrent_dropout'])
		rnn_out, self.hidden_state, self.cell_state = self.lstm(self.input)
		self.output = layers.TimeDistributed(MDN(self.hps['out_width'] * self.hps['kmix'] * 3))(rnn_out)
		
		self.model = Model(self.input, self.output)
		print("Input:", self.model.input)
		print("Output:", self.model.output)

		self.model.compile(optimizer='adam', loss=self.mdn_loss())
	
		self.get_rnn_states = K.function([self.input], [self.hidden_state, self.cell_state])

	def train(self, x, y):
		self.model.fit(x, y, batch_size=self.hps['batch_size'], validation_split=self.hps['validation_split'], epochs=self.hps['epochs'])

	def evaluate(self, x, y):
		loss = self.model.evaluate(x, y, batch_size=self.hps['batch_size'])
		return loss

	def predict(self, x):
		return self.model.predict(x)

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

	def mdn_loss(self):
		def loss(y, output):
			# reshape the y vector from (batch, maxlen, out) to (batch * maxlen * out, 1)
			y = K.reshape(y, (-1, 1))
			# out shape = (batch * maxlen * out, kmix*3)
			output = K.reshape(output, (-1, self.hps['kmix'] * 3))
			logmix, mean, logstd = MDNRNN.get_mdn_coef(output)
			return MDNRNN.get_lossfunc(logmix, mean, logstd, y)
		return loss

	def get_lstm_value(self):
		return self.hidden_state, self.cell_state

	def set_stateful(self, boolean):
		self.lstm.stateful = boolean

	def reset_states(self):
		self.model.reset_states()		

	def rnn_next_state_stateful_init_seq(self, z, a, seq):		
		out = self.model.predict(seq)
		return self.rnn_next_state_stateful(z, a)

	# Potential Alternative
	'''
	old_states = [state_h, state_c]
	lstm_layer = model.get_layer('lstm')
	lstm_layer.reset_states(states=old_states)
	pred = model.predict(x=x)
	new_states_to_save = [pred[1], pred[2]]
	'''

	def rnn_next_state(self, z, a, h, c):
		# Note that to run this stateful should be true
		input_x = np.concatenate((z.reshape((1, 1, self.hps['out_width'])),
									a.reshape((1, 1, self.hps['action_size']))), axis=2)
		self.lstm.reset_states(states=[h, c])
		new_h, new_c = self.get_rnn_states([input_x])
		return new_h, new_c

	def rnn_next_state_stateful(self, z, a):
		# Note that to run this stateful must be True!
		# It will automatically continue from the last state
		input_x = np.concatenate((z.reshape((1, 1, self.hps['out_width'])),
									a.reshape((1, 1, self.hps['action_size']))), axis=2)
		new_h, new_c = self.get_rnn_states([input_x])
		print(new_h[0][:5], new_c[0][:5])
		return new_h, new_c

	def get_pi_idx(self, x, pdf):
		# samples from a categorial distribution
		N = pdf.size
		accumulate = 0
		for i in range(0, N):
			accumulate += pdf[i]
			if (accumulate >= x):
		  		return i
		print('error with sampling ensemble')
		return -1

	def sample_sequence(self, init_z, actions, temperature=1.0, length=1000):
		self.lstm.stateful = True
		self.reset_states()
		strokes = np.zeros((length, self.hps['out_width']), dtype=np.float32)
		z = init_z.reshape((1, 1, self.hps['out_width']))
		for i in range(length):
			in_vec = np.concatenate((z, actions[i].reshape((1, 1, self.hps['action_size']))), axis=2)
			out_vec = self.model.predict(in_vec)
			out_vec = np.reshape(out_vec, (-1, self.hps['kmix'] * 3))
			logmix, mean, logstd = MDNRNN.get_mdn_coef(out_vec)
			logmix = K.eval(logmix)
			logmix2 = np.copy(logmix)/temperature
			logmix2 -= logmix2.max()
			logmix2 = np.exp(logmix2)
			logmix2 /= logmix2.sum(axis=1).reshape(self.hps['out_width'], 1)

			mixture_idx = np.zeros(self.hps['out_width'])
			chosen_mean = np.zeros(self.hps['out_width'])
			chosen_logstd = np.zeros(self.hps['out_width'])
			for j in range(self.hps['out_width']):
				idx = self.get_pi_idx(np.random.rand(), logmix2[j])
				mixture_idx[j] = idx
				chosen_mean[j] = mean[j][idx]
				chosen_logstd[j] = logstd[j][idx]

			rand_gaussian = np.random.randn(self.hps['out_width'])*np.sqrt(temperature)
			next_x = chosen_mean + np.exp(chosen_logstd)*rand_gaussian

			strokes[i,:] = next_x

			z = np.reshape(next_x, (1, 1, self.hps['out_width']))

		self.lstm.stateful = False
		return strokes

	def sample_output(self, output, temperature=1.0):
		out_vec = np.reshape(out_vec, (-1, self.hps['kmix'] * 3))
		logmix, mean, logstd = MDNRNN.get_mdn_coef(out_vec)
		logmix = K.eval(logmix)
		logmix2 = np.copy(logmix)/temperature
		logmix2 -= logmix2.max()
		logmix2 = np.exp(logmix2)
		logmix2 /= logmix2.sum(axis=1).reshape(self.hps['out_width'], 1)

		mixture_idx = np.zeros(self.hps['out_width'])
		chosen_mean = np.zeros(self.hps['out_width'])
		chosen_logstd = np.zeros(self.hps['out_width'])
		for j in range(self.hps['out_width']):
			idx = self.get_pi_idx(np.random.rand(), logmix2[j])
			mixture_idx[j] = idx
			chosen_mean[j] = mean[j][idx]
			chosen_logstd[j] = logstd[j][idx]

		rand_gaussian = np.random.randn(self.hps['out_width'])*np.sqrt(temperature)
		return chosen_mean + np.exp(chosen_logstd)*rand_gaussian

def main():
	# MDN Parameters
	hps = {}
	hps['batch_size'] = 5
	hps['max_seq_len'] = 150
	hps['in_width'] = 68 # latent + action
	hps['out_width'] = 64 # Latent
	hps['action_size'] = 4 # in width - out width
	hps['rnn_size'] = 128
	hps['kmix'] = 5
	hps['dropout'] = 0.5
	hps['recurrent_dropout'] = 0.5
	hps['validation_split'] = 0.1
	hps['epochs'] = 5

	mdnrnn = MDNRNN(hps)
	print("FINISHED BUILD")
	X = np.load("data/LunarLander_MDN_input.npy")
	Y = np.load("data/LunarLander_MDN_output.npy")

	mdnrnn.train(X, Y)
	print("FINISH TRAIN")

	reset_seq = np.zeros(shape=(1, 1, hps['in_width']))
	
	# for test_ind in [0, 1]:
	# 	# Reset the MDNRNN
	# 	mdnrnn.stateful = False
	# 	mdnrnn.predict(reset_seq)

	# 	X_test = np.expand_dims(X[test_ind, :75, :], axis=0)
	# 	X_init_z = np.expand_dims(X[test_ind, 75, :64], axis=0)
	# 	X_actions =X[test_ind, :75, 64:]
		
	# 	# Make predictions, start from halfway through sequence
	# 	mdnrnn.set_stateful(True)
	# 	mdnrnn.predict(X_test)
	# 	outs = mdnrnn.sample_sequence(X_init_z, X_actions, length=25)
	# 	np.save("results/LunarLanderSeq75-100_input_" + str(test_ind) + ".npy", outs)

	# Test stateful stuff
	mdnrnn.set_stateful(True)
	z = np.random.normal(size=(1, 1, hps['out_width']))
	a = np.random.normal(size=(1, hps['action_size']))
	h = np.zeros((1, hps['rnn_size']))
	c = np.zeros((1, hps['rnn_size']))
	h, c = mdnrnn.rnn_next_state_stateful(z, a)
	h, c = mdnrnn.rnn_next_state_stateful(z + 1, a)
	h, c = mdnrnn.rnn_next_state_stateful(z + 2, a)


if __name__ == "__main__":
	main()
