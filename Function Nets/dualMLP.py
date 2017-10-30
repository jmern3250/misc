import numpy as np 
import tensorflow as tf 

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
    basis = np.ones([input_dims*2,])
    basis[input_dims:] = -1
    comb = itertools.permutations(basis,input_dims)
    full_set = np.zeros([0,input_dims])
    for c in comb: 
        full_set = np.vstack([full_set, c])
    combs = np.unique(full_set,axis=0)
    return combs

def solveB(X, W, c, Bmax):
	def loss(X, W, c, b):
		x_hat = W.dot(b) + c
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

# First build the computation graph

X = tf.placeholder(tf.float32, [None, 1])
B = tf.placeholder(tf.float32, [None, BITS])
nu = tf.placeholder(tf.float32, [1])

with tf.variable_scope('Encoder') as enc: 
	binary = encoder(X, SIZES, BITS)

with tf.variable_scope('Decoder') as dec: 
	x_B = decoder(B)
	x_H = decoder(binary)

# weights = tf.get_default_graph().get_tensor_by_name(
#   os.path.split(x.name)[0] + '/kernel:0')

train_error = tf.nn.l2_loss(X - X_B)**2 + nu*tf.nn.l2_loss(B - binary)**2 
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

sess.run(global_variables_initializer())

# Solve the problem... 
Bmax = maxset(BITS)
# TODO generate Xd 
for i in range(10):
	# TODO Extract W & c
	b = solveB(Xd, W, c, Bmax)


