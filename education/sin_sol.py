import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Net:
	def __init__(self):
		self.x_train = np.linspace(-np.pi * 2, np.pi * 2, num=10000)
		self.y_train = np.sin(self.x_train) + self.x_train / 3 + np.random.randn(10000) / 10
		self.input_size, self.output_size = 1, 1
		self.hidden_size = 32
		self.num_epochs = 5000
		self.sess = tf.Session()
		self._build_model()

	def _build_model(self):
		self.x = tf.placeholder(tf.float32, [None, self.input_size])
		self.y_truth = tf.placeholder(tf.float32, [None, self.output_size])
		hidden_layer = tf.layers.dense(self.x, self.hidden_size, use_bias=True, activation=tf.nn.tanh)
		self.y_hat = tf.layers.dense(hidden_layer, self.output_size)
		self.loss = tf.losses.mean_squared_error(self.y_hat, self.y_truth)
		self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
		self.sess.run(tf.global_variables_initializer())

	def train(self):
		x_train, y_train = np.array([self.x_train]).T, np.array([self.y_train]).T
		for epoch in range(self.num_epochs):
			loss, _ = self.sess.run([self.loss, self.optimizer], {self.x: x_train, self.y_truth: y_train})
			if epoch % 500 == 0:
				print(loss)

	def evaluate(self, x):
		return self.sess.run(self.y_hat, {self.x: x})

def plot(model):
	x = np.linspace(-2 * np.pi, 2 * np.pi, num=1000)
	y = np.array([model.evaluate([[i]]) for i in x])
	y = y[:,0,0]
	plt.plot(model.x_train, model.y_train)
	plt.plot(x, y)
	plt.show()

sin_pred = Net()
sin_pred.train()
plot(sin_pred)

