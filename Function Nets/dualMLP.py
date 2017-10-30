import numpy as np 
import tensorflow as tf 
import itertools 

import pdb

def encoder(X, sizes, bits=8):
	layers = {}
	layers[0] = tf.layers.dense(X, 
								sizes[0],
								activation=tf.nn.relu,
								use_bias=False
								) 
	for j, size in enumerate(sizes[1:]):
		i = j+1
		layers[i] = tf.layers.dense(layers[i-1], 
								size,
								activation=tf.nn.relu,
								use_bias=False
								) 

	binary = tf.layers.dense(layers[len(sizes)-1], 
									bits,
                                    activation = tf.tanh,
                                    use_bias=False,
                                    ) 
	return binary 

def decoder(binary):
	output = tf.layers.dense(binary, 1, activation=None, use_bias=False)
	return output 

def maxset(B):
	span = np.ones([0,B])
	i = 0
	for row in itertools.product([-1,1], repeat=8):
		span = np.vstack([span,row])
		i += 1
	return span 

def solveB(X, W, c, Bmax):
	def loss(X, W, c, b):
		x_hat = W.T.dot(b) + c
		# pdb.set_trace()
		loss = np.linalg.norm(X - x_hat)**2
		return loss

	Bloss = []
	for b in Bmax: 
		bloss = loss(X,W,c,b)
		Bloss.append(bloss)
	idx = np.argmin(Bloss)
	B = Bmax[idx]
	return B 

SIZES = [256, 512, 256]
BITS = 8
INTS_OUT = 10
INTS_IN = 4000
NU = np.array(10.0).reshape([1,])

# First build the computation graph

X = tf.placeholder(tf.float32, [None, 1])
B = tf.placeholder(tf.float32, [None, BITS])
nu = tf.placeholder(tf.float32, [1])

with tf.variable_scope('Encoder') as enc: 
	binary = encoder(X, SIZES, BITS)

with tf.variable_scope('Decoder') as dec: 
	x_B = decoder(B)

with tf.variable_scope(dec, reuse=True):
	x_H = decoder(binary)

# weights = tf.get_default_graph().get_tensor_by_name(
#   os.path.split(x.name)[0] + '/kernel:0')

train_error = tf.nn.l2_loss(X - x_B)**2 + nu*tf.nn.l2_loss(B - binary)**2 
# + 0.1*tf.nn.l2_loss(weights)**2
tf.summary.scalar('train_loss', train_error)

test_error = tf.nn.l2_loss(X- x_H)**2
tf.summary.scalar('test_loss', test_error)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
train_step = optimizer.minimize(train_error)

# Now initialize the graph 
sess = tf.Session()
saver = tf.train.Saver()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./tb',sess.graph)

sess.run(tf.global_variables_initializer())

# Solve the problem... 
print('Generating binary permutations...')
Bmax = maxset(BITS)
print('Binary permutations complete, starting iterations...')
Xd = np.arange(-1.5, 1.5, 0.01)
Xd = Xd.reshape([-1,1])
iter_cnt = 0
for i in range(INTS_OUT):
	c = 0.0
	print('Iteration %r started...' % (i+1))
	params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Decoder')
	for param in params:
		if 'kernel:' in param.name: 
			W = sess.run(param)
		elif 'bias:' in param.name: 
			c = sess.run(param)

	print('Solving for B vector...')
	b = solveB(Xd, W, c, Bmax)
	print('B vector found. Begining SGD iterations...')
	for j in range(INTS_IN):
		pdb.set_trace()
		b = b.reshape([1,8])
		feed_dict = {X:Xd, B:b, nu:NU}
		_, tr_err, tst_err, summary = sess.run([train_step, train_error, test_error, merged], feed_dict=feed_dict)
		writer.add_summary(summary, iter_cnt)
		iter_cnt += 1
		if (j+1)%500 == 0:
			print('SGD iteration %r complete with training error %r' % (j+1, tr_err))
	print('Iteration %r complete with train error %r and test error %r' % (i+1, tr_err, tst_err))

saver.save(sess, './models/model') #, global_step=e+start_chk)
print('Training complete, model saved')





