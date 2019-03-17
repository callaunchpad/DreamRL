import random
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

SAVED_MODEL_FNAME = "rlnetlinear.model-weights"

'''
Class that represents a simple linear net to predict actions taken based
on latent z and hidden state h vector output.
'''
class RLNetLinear:
	'''
	Initialize net given size of h and z vectors, size of action space, and
	optionally, the hidden layer size
	'''
	def __init__(self, h_size, z_size, action_size, hidden_size=64):
		self.hsize, self.zsize, self.asize = h_size, z_size, action_size
		self.sess = tf.Session()
		self.hidden_size = hidden_size
		self._build_model()
		self.init_op = tf.global_variables_initializer()
		self.saver = tf.train.Saver()

	'''
	Internal method to build the computation graph
	'''
	def _build_model(self):
		with tf.variable_scope("rlnetlinear", reuse=tf.AUTO_REUSE):
			self.x = tf.placeholder(tf.float32, [None, self.hsize + self.zsize], name="X")
			self.y = tf.placeholder(tf.float32, [None, self.asize], name="Y")
			hidden_layer = tf.layers.dense(self.x, self.hidden_size, use_bias=True, activation=tf.nn.tanh, name="hidden")
			self.y_hat = tf.layers.dense(hidden_layer, self.asize, name="output")
			self.loss = tf.losses.mean_squared_error(self.y_hat, self.y)
			self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

	'''
	Load saved model weights from optionally specified filename
	'''
	def load_model(self, fname=SAVED_MODEL_FNAME):
		self.saver.restore(self.sess, fname)

	'''
	Save model weights to optionally specified filename
	'''
	def save_model(self, fname=SAVED_MODEL_FNAME):
		assert self.saver.save(self.sess, fname) == fname, "Sanity check failed - saved to filename other than specified one"

	'''
	Train model on specified X (size N x (h + z)) and Y (size N x a) output with optionally
	specified number of epochs and batch size
	'''
	def train(self, xtrain, ytrain, num_epochs=2500, batch_size=128):
		self.sess.run(self.init_op)
		assert xtrain.shape[1] == (self.hsize + self.zsize), "Input X for training should have column shape of <size of h> + <size of z>"
		assert ytrain.shape == (xtrain.shape[0], self.asize), "Output Y for training should have size N x <size of a> where N = # of rows in input X for training"
		for epoch in range(1, num_epochs + 1):
			batch_indices = np.array(random.sample(range(xtrain.shape[0]), batch_size))
			batchx, batchy = xtrain[batch_indices], ytrain[batch_indices]
			loss, _ = self.sess.run([self.loss, self.optimizer], {self.x: batchx, self.y: batchy})
			if epoch % 500 == 0:
				print("Epoch: %d | Loss: %f\n" % (epoch, loss))

	'''
	Evaluate model on given x of size (n x (h + z)) to give output of size (n x a)
	'''
	def evaluate(self, x):
		return self.sess.run(self.y_hat, {self.x: x})

'''
Test model on function y = sin(x) + x/3 in range [-2*pi, 2*pi]
'''
def test_model(use_saved=False):
	'''
	Plot output of model versus actual function
	'''
	def plot(model):
		x = np.linspace(-2 * np.pi, 2 * np.pi, num=1000)
		y_actual = np.sin(x) + x / 3
		print("Plotting model...")
		y = np.array([model.evaluate(np.array([[i]])) for i in x])
		print("Showing plot...")
		y = y[:,0,0]
		plt.plot(x, y)
		plt.plot(x, y_actual)
		plt.show()

	# Generate training data from function with artifical Gaussian noise
	x_train = np.linspace(-np.pi * 2, np.pi * 2, num=10000)
	y_train = np.sin(x_train) + x_train / 3 + np.random.randn(10000) / 10

	# Train model on training data and save it OR read from saved model file if specified
	sin_pred = RLNetLinear(0, 1, 1, hidden_size=128)
	if not use_saved:
		sin_pred.train(
			np.reshape(x_train, (len(x_train), 1)),
			np.reshape(y_train, (len(y_train), 1)),
			num_epochs=15000,
			batch_size=128
		)
		sin_pred.save_model()
	else:
		sin_pred.load_model()

	# Plot model
	plot(sin_pred)

if __name__ == '__main__':
	print("Training RLNetLinear on y = sin(x) + x/3 in range [-2*pi, 2*pi] with Gaussian noise")
	test_model(use_saved=False)
	print("Testing save / load model functionality")
	test_model(use_saved=True)


