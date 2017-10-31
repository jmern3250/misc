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
								# use_bias=True
								use_bias=False
								) 
	for j, size in enumerate(sizes[1:]):
		i = j+1
		layers[i] = tf.layers.dense(layers[i-1], 
								size,
								activation=tf.nn.relu,
								# use_bias=True
								use_bias=False
								) 

	binary = tf.layers.dense(layers[len(sizes)-1], 
									bits,
                                    activation = tf.tanh,
                                    # use_bias=True
                                    use_bias=False
                                    ) 
	return binary 

def decoder(binary):
	output = tf.layers.dense(binary, 1, activation=None, use_bias=False)
	return output 



def real2bits(value,bits=8,radius=1.5):
	binary = np.ones([bits,])*-1.0
	step = radius/(2**bits/2)
	if value < 0:
	    binary[0] = 1
	integer = int(np.floor(np.abs(value/step)))
	done = False 
	i = bits-1
	while not done: 
	    binary[i] = (integer%2)*2 - 1
	    integer = int(integer/2)
	    if integer == 0:
	        done = True 
	    else:
	        i -= 1 
	return binary 

def bits2real(bit_array,bits=8,radius=1.5):
    step = radius/(2**(bits-1))
    weights = np.zeros([bits,])
    for i in range(bits-1):
        weights[bits-i-1] = 2**i
    value = weights.dot((bit_array+1.0)/2.0)*step
    if bool(bit_array[0]):
        value *= -1.0
    return value


SIZES = [256, 512, 1024]
BITS = 8
N = 2000
NU = np.array(1.0).reshape([1,])

# First build the computation graph

X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, BITS])
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

reg_error *= 1e-4
trans_error = 1e-2*(tf.nn.l2_loss(binary - Y) + tf.reduce_sum(tf.abs(binary - Y)))
# bin_error = 1e-4*(tf.nn.l2_loss(binary**2 - 1) + tf.reduce_sum(tf.abs(binary**2 - 1)))
# bin_error = 1e-3*(tf.nn.l2_loss(binary - Y) + tf.reduce_sum(tf.abs(binary - Y)))
# trans_error = 1e-2*(tf.nn.l2_loss(X - x_B) + tf.reduce_sum(tf.abs(X - x_B))) 

error = reg_error + trans_error 
# + bin_error
trans_sum = tf.squeeze(trans_error) 
# tf.summary.scalar('bin_error', tf.squeeze(bin_error))
tf.summary.scalar('reg_error', tf.squeeze(reg_error))
tf.summary.scalar('trans_error', trans_sum)

# optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
train_step = optimizer.minimize(error)


# Now initialize the graph 
sess = tf.Session()
saver = tf.train.Saver()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./tb',sess.graph)

sess.run(tf.global_variables_initializer())

Xd = np.arange(-1.5, 1.5, 0.005)
Xd = Xd.reshape([-1,1])

Yd = np.zeros([0,BITS])
for i, x in enumerate(Xd): 
	y = real2bits(x, bits=BITS, radius=1.5)
	Yd = np.vstack([Yd, y])

# pdb.set_trace()

i = 0
for i in range(N):
	feed_dict = {X:Xd, Y:Yd}
	_, err, summary = sess.run([train_step, error, merged], feed_dict=feed_dict)
	writer.add_summary(summary, i)
	if i%500 == 0:
		print('SGD iteration %r complete with error %r' % (i, err))

h = sess.run(binary, feed_dict={X:Xd})
h_ = np.unique(h)
print('Mean of binary: %r' % (np.mean(np.abs(h))))
fin_bin = sess.run(binary, feed_dict={X:Xd})
fin_error = np.linalg.norm(fin_bin - Yd, axis=1)

max_err = np.amax(fin_error)
mean_err = np.mean(fin_error)
print('Maximum error: %r; Mean error: %r' % (max_err, mean_err))
saver.save(sess, './models/model') #, global_step=e+start_chk)
print('Training complete, model saved')





