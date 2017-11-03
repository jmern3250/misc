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
								# use_bias=False
								) 
	for j, size in enumerate(sizes[1:]):
		i = j+1
		layers[i] = tf.layers.dense(layers[i-1], 
								size,
								activation=tf.nn.relu,
								use_bias=True
								# use_bias=False
								) 

	binary = tf.layers.dense(layers[len(sizes)-1], 
									bits,
									# activation=tf.tanh,
                                    activation = None,
                                    use_bias=True
                                    # use_bias=False
                                    ) 
	return binary 

def decoder(binary):
	output = tf.layers.dense(binary, 1, activation=None, use_bias=False)
	return output 


SIZES = [512, 512, 512]
BITS = 8
N = 10000
NU = np.array(1.0).reshape([1,])

# First build the computation graph

X = tf.placeholder(tf.float32, [None, 1])
nu = tf.placeholder(tf.float32, [1])

with tf.variable_scope('Encoder') as enc: 
	binary = encoder(X, SIZES, BITS)

with tf.variable_scope('Decoder') as dec: 
	x_B = decoder(binary)


reg_error = 0.0
params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
for param in params:
	if 'kernel:' in param.name: 
		reg_error += tf.nn.l2_loss(param)
reg_error *= 1e-2

trans_error = 10.0*(tf.nn.l2_loss(X - x_B) + tf.reduce_sum(tf.abs(X - x_B))) 
bin_error = 5e-4*tf.nn.l2_loss(1 - binary**2) + tf.reduce_sum(tf.abs(1 - binary**2))
# idp_error = tf.nn.l2_loss(tf.reduce_mean(tf.matmul(tf.transpose(binary), binary)) - np.eye(BITS))
idp_error = 0.0
bal_error = 2e-5*tf.nn.l2_loss(tf.matmul(binary, np.ones([BITS, 1], dtype=np.float32)))
# bal_error = 0.0

error = reg_error + trans_error + bin_error + 1e-5*idp_error + bal_error
trans_sum = tf.squeeze(trans_error) 
tf.summary.scalar('reg_error', tf.squeeze(reg_error))
tf.summary.scalar('trans_error', trans_sum)
tf.summary.scalar('bin_error', tf.squeeze(bin_error))
tf.summary.scalar('bal_error', tf.squeeze(bal_error))

final_error = np.abs(X - x_B)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-2)

train_step = optimizer.minimize(error)

# Now initialize the graph 
sess = tf.Session()
saver = tf.train.Saver()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./tb',sess.graph)

sess.run(tf.global_variables_initializer())

Xd = np.arange(-1.5, 1.5, 0.01)
Xd = Xd.reshape([-1,1])

i = 0
for i in range(N):
	feed_dict = {X:Xd, nu:NU}
	_, err, summary = sess.run([train_step, error, merged], feed_dict=feed_dict)
	writer.add_summary(summary, i)
	if i%500 == 0:
		print('SGD iteration %r complete with error %r' % (i, err))

h = sess.run(binary, feed_dict={X:Xd})
h_ = np.unique(h)
fin_err = sess.run(final_error, feed_dict={X:Xd})
max_err = np.amax(fin_err)
mean_err = np.mean(fin_err)
print('Maximum error: %r; Mean error: %r' % (max_err, mean_err))
print('Mean of binary: %r' % (np.mean(np.abs(h_))))
pdb.set_trace()
saver.save(sess, './models/model') #, global_step=e+start_chk)
print('Training complete, model saved')





