import numpy as np 
import tensorflow as tf 
import itertools 
import os

import pdb

def encoder(X, sizes, bits=8):
	layers = {}
	layers[0] = tf.layers.dense(X, 
								sizes[0],
								activation=tf.nn.relu,
								use_bias=True
								) 
	for j, size in enumerate(sizes[1:]):
		i = j+1
		layers[i] = tf.layers.dense(layers[i-1], 
								size,
								activation=tf.nn.relu,
								use_bias=True
								) 

	binary = tf.layers.dense(layers[len(sizes)-1], 
									bits,
                                    activation = tf.tanh,
                                    use_bias=True,
                                    ) 
	return binary 

def decoder(binary):
	output = tf.layers.dense(binary, 1, activation=None, use_bias=False)
	return output 

def maxset(B):
	span = np.ones([0,B])
	i = 0
	for row in itertools.product([-1,1], repeat=B):
		span = np.vstack([span,row])
		i += 1
	return span 

def solveB(H, X, W, c, Bmax, nu):
	def loss(H, X, W, c, b, nu):
		x_hat = W.T.dot(b) + c
		loss = np.linalg.norm(X - x_hat)**2 + nu*np.linalg.norm(b - H)**2
		return loss
	B = np.zeros([0,Bmax.shape[1]])
	for i, x in enumerate(X): 
		Bloss = []
		for b in Bmax: 
			bloss = loss(H[i,:],x,W,c,b,nu)
			Bloss.append(bloss)
		idx = np.argmin(Bloss)
		B = np.vstack([B, Bmax[idx]])
	return B 

SIZES = [512, 512, 512]
BITS = 8
INTS_OUT = 100
INTS_IN = 2000
NU = np.array(1e-2).reshape([1,])

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



train_error = (tf.nn.l2_loss(X - x_B) + tf.reduce_sum(tf.abs(X - x_B))) + nu*(tf.nn.l2_loss(B - binary) + tf.reduce_sum(tf.abs(B - binary))) 
train_sum = tf.squeeze(train_error) 


tf.summary.scalar('train_error', train_sum)

test_error = tf.reduce_mean((X - x_H)**2)
test_sum = tf.squeeze(test_error)
tf.summary.scalar('test_error', test_sum)

bin_error = (X - x_B)**2
# final_error = (X - x_H)**2

# optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-2)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
train_step = optimizer.minimize(train_error)
pre_train_step = optimizer.minimize(test_error)

# Now initialize the graph 
sess = tf.Session()
saver = tf.train.Saver()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./tb',sess.graph)

sess.run(tf.global_variables_initializer())

Xd = np.arange(-1.5, 1.5, 0.01)
Xd = Xd.reshape([-1,1])

# Initialize with pure SGD
print('Pre-training...')
for _ in range(4000):
	_ = sess.run(pre_train_step, feed_dict={X:Xd})

# Solve the problem... 
print('Generating binary permutations...')
Bmax = maxset(BITS)
print('Binary permutations complete, starting iterations...')

iter_cnt = 0
done = False 
tst_err_0 = 1e8
i = 0
while not done:
	c = 0.0
	print('Iteration %r started...' % (i+1))
	params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Decoder')
	for param in params:
		if 'kernel:' in param.name: 
			W = sess.run(param)
		elif 'bias:' in param.name: 
			c = sess.run(param)
	h = sess.run(binary, feed_dict={X:Xd})

	print('Solving for B vector...')
	b = solveB(h, Xd, W, c, Bmax, NU)
	print('B vector found. Begining SGD iterations...')
	for j in range(INTS_IN):
		feed_dict = {X:Xd, B:b, nu:NU}
		_, tr_err, tst_err, summary = sess.run([train_step, train_error, test_error, merged], feed_dict=feed_dict)
		writer.add_summary(summary, iter_cnt)
		iter_cnt += 1
		if j%500 == 0:
			print('SGD iteration %r complete with training error %r' % (j, tr_err[0]))
	print('Iteration %r complete with train error %r and test error %r' % (i+1, tr_err[0], tst_err))
	i += 1
	d_tst_err = np.abs(tst_err - tst_err_0)
	tst_err_0 = tst_err
	if d_tst_err <= 1e-3 or i >= INTS_OUT:
		done = True
h = sess.run(binary, feed_dict={X:Xd})
h_ = np.unique(h)
bin_err = sess.run(bin_error, feed_dict={X:Xd, B:b})
max_err = np.amax(bin_err)
pdb.set_trace()
saver.save(sess, './models/model') #, global_step=e+start_chk)
print('Training complete, model saved')





