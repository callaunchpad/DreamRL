# Author: Joey Hejna, Jihan
# Resources: https://github.com/yanji84/keras-mdn/blob/master/mdn.py
#            https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/rnn/rnn.py
#			 https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

import numpy as np
import tensorflow as tf
import time
import json

## CONSTANTS
logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))


class MDNRNN():
	def __init__(self, hyperparameters):
		self.hps = hyperparameters
		self.g = tf.Graph()
		with self.g.as_default():
			self.build_model()
		self.init_sess()

	def build_model(self):
		if self.hps['training']:
			self.global_step = tf.Variable(0, name='global_step', trainable=False)

		self.input = tf.placeholder(dtype=tf.float32, shape=[self.hps['batch_size'], self.hps['max_seq_len'], self.hps['in_width']])
		self.output = tf.placeholder(dtype=tf.float32, shape=[self.hps['batch_size'], self.hps['max_seq_len'], self.hps['out_width']])

		# RNN
		cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hps['rnn_size'], dropout_keep_prob=self.hps['recurrent_dropout'])
		self.cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.hps['dropout'])

		self.initial_state = self.cell.zero_state(batch_size=self.hps['batch_size'], dtype=tf.float32)

		output, last_state = tf.nn.dynamic_rnn(self.cell, self.input, initial_state=self.initial_state,
										   time_major=False, swap_memory=True, dtype=tf.float32)

		self.final_state = last_state
		# MDN
		output = tf.reshape(output, [-1, self.hps['rnn_size']])
		output_w = tf.get_variable("output_w", [self.hps['rnn_size'], self.hps['out_width'] * self.hps['kmix'] * 3])
		output_b = tf.get_variable("output_b", [self.hps['out_width'] * self.hps['kmix'] * 3])
		output = tf.nn.xw_plus_b(output, output_w, output_b)
		output = tf.reshape(output, [-1, self.hps['kmix'] * 3])

		self.out_logmix, self.out_mean, self.out_logstd = MDNRNN.get_mdn_coef(output)

		flat_target_data = tf.reshape(self.output, [-1, 1])

		lossfunc = MDNRNN.get_lossfunc(self.out_logmix, self.out_mean, self.out_logstd, flat_target_data)

		self.cost = tf.reduce_mean(lossfunc)

		if self.hps['training']:
			self.lr = tf.Variable(self.hps['lr'], trainable=False)
			optimizer = tf.train.AdamOptimizer(self.lr)
			gvs = optimizer.compute_gradients(self.cost)
			capped_gvs = [(tf.clip_by_value(grad, -self.hps['grad_clip'], self.hps['grad_clip']), var) for grad, var in gvs]
			self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')

		self.init = tf.global_variables_initializer()

		t_vars = tf.trainable_variables()

		self.assign_ops = {}
		for var in t_vars:
			#if var.name.startswith('mdn_rnn'):
			pshape = var.get_shape()
			pl = tf.placeholder(tf.float32, pshape, var.name[:-2]+'_placeholder')
			assign_op = var.assign(pl)
			self.assign_ops[var] = (assign_op, pl)

	def init_sess(self):
		self.sess = tf.Session(graph=self.g)
		self.sess.run(self.init)

	def get_mdn_coef(output):
		# first column is the batch dimension
		logmix, mean, logstd = tf.split(output, 3, 1)
		logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
		return logmix, mean, logstd

	def lognormal(y, mean, logstd):
		return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

	def get_lossfunc(logmix, mean, logstd, y):
		v = logmix + MDNRNN.lognormal(y, mean, logstd)
		v = tf.reduce_logsumexp(v, 1, keepdims=True)
		out = -tf.reduce_mean(v)
		return out

	def get_model_params(self):
		# get trainable params.
		model_names = []
		model_params = []
		model_shapes = []
		with self.g.as_default():
			t_vars = tf.trainable_variables()
			for var in t_vars:
				#if var.name.startswith('mdn_rnn'):
				param_name = var.name
				p = self.sess.run(var)
				model_names.append(param_name)
				params = np.round(p*10000).astype(np.int).tolist()
				model_params.append(params)
				model_shapes.append(p.shape)
		return model_params, model_shapes, model_names

	def set_model_params(self, params):
		with self.g.as_default():
			t_vars = tf.trainable_variables()
			idx = 0
			for var in t_vars:
				#if var.name.startswith('mdn_rnn'):
				pshape = tuple(var.get_shape().as_list())
				p = np.array(params[idx])
				assert pshape == p.shape, "inconsistent shape"
				assign_op, pl = self.assign_ops[var]
				self.sess.run(assign_op, feed_dict={pl.name: p/10000.})
				idx += 1

	def load(self, path):
		with open(path, 'r') as f:
			params = json.load(f)
			self.set_model_params(params)

	def save(self, path):
		model_params, model_shapes, model_names = self.get_model_params()
		qparams = []
		for p in model_params:
			qparams.append(p)
		with open(path, 'wt') as outfile:
			json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

	def train(self, x, y):
		start = time.time()

		for step in range(self.hps['num_steps']):
			s = self.sess.run(self.global_step)

			# Can Adjust Later
			lr = self.hps['lr']

			indices = np.random.permutation(len(x))[:self.hps['batch_size']]
			batch_x = x[indices]
			batch_y = y[indices]

			feed = {self.input: batch_x, self.output: batch_y, self.lr: lr}
			(train_cost, state, train_step, _) = self.sess.run([self.cost, self.final_state, self.global_step, self.train_op], feed)

			if (step%20==0 and step > 0):
				end = time.time()
				time_taken = end-start
				start = time.time()
				output_log = "step: %d, lr: %.6f, cost: %.4f, train_time_taken: %.4f" % (step, lr, train_cost, time_taken)
				print(output_log)

	def rnn_init_state(self):
		return self.sess.run(self.initial_state)

	def rnn_next_state(self, z, a, prev_state):
		input_x = np.concatenate((z.reshape((1, 1, self.hps['out_width'])),
									a.reshape((1, 1, self.hps['action_size']))), axis=2)
		feed = {self.input: input_x, self.initial_state: prev_state}
		return self.sess.run(self.final_state, feed)

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

	def sample_sequence(self, init_z, actions, temperature=1.0, length=1000, prev_state=None):
		if prev_state is None:
			prev_state = self.sess.run(self.initial_state)
		strokes = np.zeros((length, self.hps['out_width']), dtype=np.float32)
		z = init_z.reshape((1, 1, self.hps['out_width']))

		for i in range(length):
			in_vec = np.concatenate((z, actions[i].reshape((1, 1, self.hps['action_size']))), axis=2)
			feed = {self.input: in_vec, self.initial_state : prev_state}
			[logmix, mean, logstd, next_state] = self.sess.run([self.out_logmix, self.out_mean, self.out_logstd, self.final_state], feed)

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
			prev_state = next_state
			z = np.reshape(next_x, (1, 1, self.hps['out_width']))

		return strokes

	def sample_z(self, z, a, prev_state, temperature=1.0):
		in_vec = np.concatenate((z.reshape((1, 1, self.hps['out_width'])),
									a.reshape((1, 1, self.hps['action_size']))), axis=2)
		feed = {self.input: in_vec, self.initial_state : prev_state}

		[logmix, mean, logstd, next_state] = self.sess.run([self.out_logmix, self.out_mean, self.out_logstd, self.final_state], feed)

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
		z = chosen_mean + np.exp(chosen_logstd)*rand_gaussian
		z = np.reshape(z, (1, 1, self.hps['out_width']))
		return z, next_state

	def set_hps_to_inference(hps):
		hps = hps.copy()
		hps['batch_size'] = 1
		hps['max_seq_len'] = 1
		hps['use_recurrent_dropout'] = 0
		hps['training'] = 0
		return hps

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
	hps['num_steps'] = 50
	hps['training'] = True
	hps['lr'] = 0.001
	hps['grad_clip'] = 1.0

	mdnrnn = MDNRNN(hps)
	print("FINISHED BUILD")
	X = np.load("data/LunarLander_MDN_input.npy")
	Y = np.load("data/LunarLander_MDN_output.npy")

	mdnrnn.train(X, Y)
	print("FINISH TRAIN")
	mdnrnn.save("checkpoints/lunar.json")

	hps_inf = MDNRNN.set_hps_to_inference(hps)
	mdnrnn_inf = MDNRNN(hps_inf)
	mdnrnn_inf.load("checkpoints/lunar.json")

	state = mdnrnn_inf.rnn_init_state()
	print(state)
	z = np.random.normal(size=(1, 1, hps['out_width']))
	a = np.random.normal(size=(1, hps['action_size']))

	state = mdnrnn_inf.rnn_next_state(z, a, state)
	print(state)

	# Test Sample Z
	z = np.random.normal(size=(1, 1, hps['out_width']))
	a = np.random.normal(size=(1, hps['action_size']))
	z, state = mdnrnn_inf.sample_z(z, a, state)
	print(z)

	# Test Sample Seq
	init_z = np.random.normal(size=(1, 1, hps['out_width']))
	actions = np.random.normal(size=(10, hps['action_size']))
	zs = mdnrnn_inf.sample_sequence(init_z, actions, length=10)
	print(zs)


if __name__ == "__main__":
	main()
