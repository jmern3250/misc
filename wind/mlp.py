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
		self.X = tf.placeholder(tf.float32, [None, 7])
		self.M = tf.placeholder(tf.float32, [None, 360])
		self.is_training = tf.placeholder(tf.bool)
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
			m0 = tf.layers.dense(
						    self.M,
						    128,
						    activation=lrelu,
						    use_bias=True,
						    kernel_initializer=tf.contrib.layers.xavier_initializer(),
						    bias_initializer=tf.zeros_initializer(),
						    name='m0'
						)
			m1 = tf.layers.dense(
						    m0,
						    32,
						    activation=lrelu,
						    use_bias=True,
						    kernel_initializer=tf.contrib.layers.xavier_initializer(),
						    bias_initializer=tf.zeros_initializer(),
						    name='m1'
						) 
			h = tf.concat([m1, self.X], axis=1) 
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
						    rate=0.25,
						    noise_shape=None,
						    seed=None,
						    training=self.is_training,
						    name=name+'droput'
						)
			self.Y_ = tf.layers.dense(
						    h,
						    5,
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

	def train(self, Xd, Md, Yd, n_epochs, batch_size, learning_rate, 
				filename, checkpoint=None):
		
		Y = tf.placeholder(tf.float32, [None, 5])
		L2 = tf.reduce_mean((Y - self.Y_)**2)
		L1 = tf.reduce_mean(tf.abs(Y - self.Y_))
		lam = 0.75
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
				m = Md[itr_start:itr_end, ...]
				y = Yd[itr_start:itr_end, ...]
				feed_dict = {self.X:x,
							self.M:m,
							Y:y,
							self.is_training:True}
				loss, _ = self.session.run(train_vars, feed_dict)
				losses.append(loss)
			mean_loss = np.mean(losses)
			print('Epoch %r complete with mean loss %r' % (e, mean_loss))
		print('Training complete, saving graph...')
		self.saver.save(self.session, filename)

	def predict(self, Xd, Md):
		Y = []
		n_samples = Xd.shape[0]
		for i in range(n_samples):
			x = Xd[i:i+1, ...]
			m = Md[i:i+1, ...]
			feed_dict = {self.X:x,
						 self.M:m,
						 self.is_training:False}
			y = self.session.run(self.Y_, feed_dict)
			Y.append(y.squeeze())
		Y = np.stack(Y)
		return Y

if __name__ == '__main__':
	with open('./data/X2.p', 'rb') as f: 
		Xd = pickle.load(f)
	with open('./data/M2.p', 'rb') as f: 
		Md = pickle.load(f)
	with open('./data/Y2.p', 'rb') as f: 
		Yd = pickle.load(f)
	Xmean = np.mean(Xd, axis=0)
	Xstd = np.std(Xd, axis=0)
	Mmean = np.mean(Md, axis=0)
	Mstd = np.std(Md, axis=0)
	Ymean = np.mean(Yd, axis=0)
	Ystd = np.std(Yd, axis=0)

	Xd_ = Xd - Xmean
	Xd_ /= Xstd

	Md_ = Md - Mmean
	Md_ /= Mstd

	Yd_ = Yd - Ymean
	Yd_ /= Ystd

	n_samples = Xd.shape[0]

	# np.random.seed(0)

	# test_idxs = np.random.choice(np.arange(n_samples), size=100, replace=False)
	# train_idxs = np.delete(np.arange(n_samples))

	Xtrain = Xd_[:(n_samples-100), ...]
	Mtrain = Md_[:(n_samples-100), ...]
	Ytrain = Yd_[:(n_samples-100), ...]
	Xtest = Xd_[(n_samples-100):, ...]
	Mtest = Md_[(n_samples-100):, ...]
	Ytest = Yd[(n_samples-100):, ...]

	mlp = MLP(5, 256, lrelu, scope='mlp')
	mlp.train(Xtrain, Mtrain, Ytrain, 5000, 100, 1e-4, 
				'./models_v2/model1', checkpoint='./models_v2/model0')

	# mlp.restore_graph('./models_v2/model1')
	import matplotlib.pyplot as plt
	### Training Validation ###
	y_ = mlp.predict(Xtrain, Mtrain)
	y_ *= Ystd
	y_ += Ymean

	plt.figure()
	Ytrain *= Ystd
	Ytrain += Ymean
	plt.plot(Ytrain[:,0].squeeze(), '-')
	plt.plot(y_[:,0].squeeze(), '.')
	plt.grid(True)
	plt.show()	
	
	y_ = mlp.predict(Xtest, Mtest)
	y_ *= Ystd
	y_ += Ymean

	MSE = np.linalg.norm(y_[:,0] - Ytest[:,0])
	print('Test MSE:', MSE)
	plt.figure()
	plt.plot(Ytest[:,0].squeeze(), '-')
	plt.plot(y_[:,0].squeeze(), '.')
	plt.show()	
