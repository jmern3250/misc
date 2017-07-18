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
	output_bool = tf.greater_equal(output_value, threshold_tensor)
    output = tf.cast(output_bool,dtype=tf.uint8) 
	return output

def build_code_table(bits=8, lo=-1.5, hi=1.5):
    code_dict = {}
    half_bins = int(2**bits/2)
    prev_code_array = np.zeros([1,bits],dtype=np.dtype(int)).squeeze()
    prev_code = ''.join(map(str, prev_code_array))
    step = hi/half_bins
    v_range = [0.0, step]
    code_dict[prev_code] = v_range
    for i in range(half_bins-1): 
        print('i:',i)
        idx = bits-1
        done = False
        right_code = str()
        while not done:
            targ_bit = int(prev_code[idx])
            targ_bit += 1
            if targ_bit == 2:
                right_code = str(0) + right_code 
                idx -= 1
                done = False
            else: 
                right_code = str(targ_bit) + right_code 
                done = True
        code = prev_code[:idx] + right_code 
        v_range = [v_range[1], v_range[1]+step]
        code_dict[code] = v_range
        prev_code = code[:]
    
    
    prev_code_array = np.zeros([1,bits],dtype=np.dtype(int)).squeeze()
    prev_code_array[0] = 1
    prev_code = ''.join(map(str, prev_code_array))
    step = lo/half_bins
    v_range = [0.0, step]
    code_dict[prev_code] = v_range
    for i in range(half_bins-1): 
        print('i:',i)
        idx = bits-1
        done = False
        right_code = str()
        while not done:
            targ_bit = int(prev_code[idx])
            targ_bit += 1
            if targ_bit == 2:
                right_code = str(0) + right_code 
                idx -= 1
                done = False
            else: 
                right_code = str(targ_bit) + right_code 
                done = True
        code = prev_code[:idx] + right_code 
        v_range = [v_range[1], v_range[1]+step]
        code_dict[code] = v_range
        prev_code = code[:] 
        print(prev_code)
    return code_dict 


tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=1)
y = tf.placeholder(tf.uint8, shape=16)
is_training = tf.placeholder(tf.bool, shape=1)
output = basic_MLP(X,y,is_training,threshold=0,dropout=False)


