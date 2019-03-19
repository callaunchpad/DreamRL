import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Net:
	def __init__(self):
		# Your training data
		self.x_train = np.linspace(-np.pi * 2, np.pi * 2, num=10000)
		self.y_train = np.sin(self.x_train) + self.x_train / 3 + np.random.randn(10000) / 10

		# Some parameters (feel free to edit the hidden sizes and number of epochs)
		self.input_size, self.output_size = 1, 1
		self.hidden_size1, self.hidden_size2 = 128, 128
		self.num_epochs = 5000

		# Starting a TensorFlow session
		self.sess = tf.Session()
		self._build_model()

	def _build_model(self):
		# TODO: Build your fully-connected neural net!
		pass
		
	def train(self):
		# TODO: Train your model. No need to do any batching.
		for _ in range(self.num_epochs):
			pass

	def evaluate(self, x):
		# TODO: Output your model's prediction for the input x
		return 0

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

