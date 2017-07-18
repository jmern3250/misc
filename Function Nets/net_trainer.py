import numpy as np 
import tensorflow as tf
import pickle 
import math 
import bitstring

# import bitstring
# f1 = bitstring.BitArray(float=1.0, length=32)
# print f1.read('bin') 

def basic_MLP(X,y,sizes,is_training,threshold=0,dropout=False): 
	layers = {}
	layers[0] = tf.layers.dense(X, sizes[0],
								activation = tf.nn.relu,
								use_bias=True,
								)
	if dropout:
		dropout_key = str(0)+'d' 
		layers[dropout_key] = tf.layers.dropout(layers[0],
												rate=0.25,
												training=is_training)
	for i,  size in enumerate(sizes[1:]): 
		if dropout: 
			dropout_key_prior = str(i-1)+'d' 
			layers[i] = tf.layers.dense(layers[dropout_key_prior], size,
									activation = tf.nn.relu,
									use_bias=True,
									)
			dropout_key = str(i)+'d' 
			layers[dropout_key] = tf.layers.dropout(layers[dropout_key],
												rate=0.25,
												training=is_training)
		else:
			layers[i] = tf.layers.dense(layers[i-1], size,
									activation = tf.nn.relu,
									use_bias=True,
									)

	output_value = tf.layers.dense(layers[-1], 16,
							 	   activation = None,
							       use_bias=False,
									)

	threshold_list = [threshold]*16
	threshold_tensor = tf.constant(threshold_list)
	output = tf.greater_equal(output_value, threshold_tensor)
	loss = output - y 
	return output


tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=1)
y = tf.placeholder(tf.uint8, shape=16)
is_training = tf.placeholder(tf.bool, shape=1)
output = basic_MLP(X,y,is_training,threshold=0,dropout=False)


