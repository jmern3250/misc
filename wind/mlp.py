import numpy as np 
import tensorflow as tf
import pickle 

def lrelu(x, alpha=0.1):
    return tf.nn.relu(x)*(1-alpha) - alpha*tf.nn.relu(-x)

class MLP(object):
	def __init__(self, n_layers, layer_sizes, activations, scope='mlp'):
		self.scope = scope
		self.n_layers = n_layers
		if not isinstance(layer_sizes, list):
			self.layer_sizes = [layer_sizes]*self.n_layers
		else:
			self.layer_sizes = layer_sizes
		if not isinstance(activations, list):
			self.activations = [activations]*self.n_layers
		self.build_graph()
		self.initialized = False 
		self.session = tf.Session()

	def build_graph(self):
		self.X = tf.placeholder(tf.float32, [None, 4])
		self.is_training = tf.placeholder(tf.bool)
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
			h = self.X 
			for i in range(self.n_layers):
				name = 'h' + str(i)
				h = tf.layers.dense(
						    h,
						    self.layer_sizes[i],
						    activation=self.activations[i],
						    use_bias=True,
						    kernel_initializer=tf.contrib.layers.xavier_initializer(),
						    bias_initializer=tf.zeros_initializer(),
						    name=name,
						)
				h = tf.layers.dropout(
						    h,
						    rate=0.10,
						    noise_shape=None,
						    seed=None,
						    training=self.is_training,
						    name=name+'droput'
						)
			self.Y_ = tf.layers.dense(
						    h,
						    1,
						    activation=None,
						    use_bias=True,
						    kernel_initializer=tf.contrib.layers.xavier_initializer(),
						    bias_initializer=tf.zeros_initializer(),
						    name='Y_',
						)
		var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		self.saver = tf.train.Saver(var_list=var_list)

	def restore_graph(self, checkpoint=None):
		if checkpoint is None: 
			self.session.run(tf.global_variables_initializer())
			self.initialized = True
		else:
			self.session.run(tf.global_variables_initializer())
			self.saver.restore(self.session, checkpoint)
			self.initialized = True

	def train(self, Xd, Yd, n_epochs, batch_size, learning_rate, 
				filename, checkpoint=None):
		
		Y = tf.placeholder(tf.float32, [None, 1])
		L2 = tf.reduce_mean((Y - self.Y_)**2)
		L1 = tf.reduce_mean(tf.abs(Y - self.Y_))
		lam = 1.0
		loss = lam*L2 + (1. - lam)*L1
		adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
		train = adam.minimize(loss)

		train_vars = [loss, train]

		self.restore_graph(checkpoint)
		n_samples = Xd.shape[0]
		n_batches = n_samples//batch_size
		train_itrs = np.arange(n_batches)
		for e in range(n_epochs):
			losses = []
			np.random.shuffle(train_itrs)
			for itr in train_itrs: 
				itr_start = itr*batch_size
				itr_end = (itr+1)*batch_size
				x = Xd[itr_start:itr_end, ...]
				y = Yd[itr_start:itr_end, ...]
				feed_dict = {self.X:x,
							Y:y,
							self.is_training:True}
				loss, _ = self.session.run(train_vars, feed_dict)
				losses.append(loss)
			mean_loss = np.mean(losses)
			print('Epoch %r complete with mean loss %r' % (e, mean_loss))
		print('Training complete, saving graph...')
		self.saver.save(self.session, filename)

	def predict(self, Xd):
		Y = []
		n_samples = Xd.shape[0]
		for i in range(n_samples):
			x = Xd[i:i+1, ...]
			feed_dict = {self.X:x,
						 self.is_training:False}
			y = self.session.run(self.Y_, feed_dict)
			Y.append(y.squeeze())
		Y = np.stack(Y)
		return Y

if __name__ == '__main__':
	with open('./data/Xmlp.p', 'rb') as f: 
		Xd = pickle.load(f)
	with open('./data/Ymlp.p', 'rb') as f: 
		Yd = pickle.load(f)
	Xmean = np.mean(Xd, axis=0)
	Xstd = np.std(Xd, axis=0)
	Ymean = np.mean(Yd, axis=0)
	Ystd = np.std(Yd, axis=0)

	Xd_ = Xd - Xmean
	Xd_ /= Xstd

	Yd_ = Yd - Ymean
	Yd_ /= Ystd

	n_samples = Xd.shape[0]

	Xtrain = Xd_[:(n_samples-100), ...]
	Ytrain = Yd_[:(n_samples-100), ...]
	Xtest = Xd_[(n_samples-100):, ...]
	Ytest = Yd[(n_samples-100):, ...]

	mlp = MLP(4, 256, lrelu, scope='mlp')
	mlp.train(Xtrain, Ytrain, 20000, 100, 1e-3, 
				'./models/model2', checkpoint='./models/model1')

	# import pdb; pdb.set_trace()
	# mlp.restore_graph('./models/model1')
	import matplotlib.pyplot as plt
	### Training Validation ###
	y_ = mlp.predict(Xtrain)
	y_ *= Ystd
	y_ += Ymean
	plt.figure()
	plt.plot(y_.squeeze())
	Ytrain *= Ystd
	Ytrain += Ymean
	plt.plot(Ytrain.squeeze(), '.')
	plt.grid(True)
	plt.show()	
	
	y_ = mlp.predict(Xtest)
	y_ *= Ystd
	y_ += Ymean
	plt.figure()
	plt.plot(y_.squeeze())
	plt.plot(Ytest.squeeze(), '.')
	plt.show()	
