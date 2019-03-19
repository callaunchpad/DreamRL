import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Net:
	def __init__(self):
		# Your training data
		self.x_train = np.linspace(-np.pi, np.pi * 2, num=10000)
		self.y_train = np.sin(self.x_train) + self.x_train / 3 + np.random.randn(10000) / 10

		# Some parameters (feel free to edit the hidden sizes and number of epochs)
		self.input_size, self.output_size = 1, 1
		self.hidden_size1, self.hidden_size2 = 200, 50
		self.num_epochs = 2000

		# Starting a TensorFlow session
		self.sess = tf.Session()
		self._build_model()

	def _build_model(self):
		# TODO: Build your fully-connected neural net!
		self.x = tf.placeholder("float", [None, 1])
		self.y = tf.placeholder("float", [None, 1])
		x1 = tf.contrib.layers.fully_connected(self.x, self.hidden_size1, activation_fn=tf.nn.relu) #dense same thing, bias=True
		x2 = tf.contrib.layers.fully_connected(x1, self.hidden_size2, activation_fn=tf.nn.relu)
		self.result = tf.contrib.layers.fully_connected(x2, self.output_size, activation_fn=None)
		self.loss = tf.losses.mean_squared_error(self.y, self.result)
		self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

		# pass

	def train(self):
		# TODO: Train your model. No need to do any batching.
		self.sess.run(tf.global_variables_initializer())
		for _ in range(self.num_epochs):
			xpts = self.x_train
			ypts = self.y_train
			i, loss_result = self.sess.run(fetches=[self.train_op, self.loss], feed_dict={self.x: xpts[:, None], self.y: ypts[:, None]})
			print('iteration {}, loss={}'.format(_, loss_result))

	def evaluate(self, x):
		# TODO: Output your model's prediction for the input x
		y = self.sess.run(self.result, feed_dict={self.x:x})
		#print(y)
		return y

# Plots the function your model represents (should look like the function f(x) = sin(x) + x / 3)
def show_plot(model):
	x = np.linspace(-2 * np.pi, 2 * np.pi, num=1000)
	y = np.array([model.evaluate([[i]]) for i in x])
	y = y[:,0,0]
	plt.plot(x, y)
	plt.show()

sin_pred = Net()
sin_pred.train()
show_plot(sin_pred)

