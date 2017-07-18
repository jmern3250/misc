import numpy as np 
import tensorflow as tf
import pickle 
import math 
import bitstring

# import bitstring
# f1 = bitstring.BitArray(float=1.0, length=32)
# print f1.read('bin') 

def basic_MLP(X,y,sizes,is_training,bits=8,threshold=0,dropout=False): 
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

	output_value = tf.layers.dense(layers[-1], bits,
							 	   activation = None,
							       use_bias=False,
									)

	threshold_list = [threshold]*bits
	threshold_tensor = tf.constant(threshold_list)
	output_bool = tf.greater_equal(output_value, threshold_tensor)
    output = tf.cast(output_bool,dtype=tf.uint8) 
	return output

def bit2real(X, bits=8):
    weight_list = [-1]
    for i in range(bits-1):
        weight_list.append(2**i)
    conversion_tensor = tf.constant(weight_list)
    real_output = tf.matmul(X, conversion_tensor,
                            transpose_a=True)
    return real_output

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
    return code_dict 

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%X_train.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[i:i+batch_size].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct

# with tf.Session() as sess:
#     with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 
#         sess.run(tf.global_variables_initializer())
#         print('Training')
#         run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,True)
#         print('Validation')
#         run_model(sess,y_out,mean_loss,X_val,y_val,1,64)


tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=1)
y = tf.placeholder(tf.uint8, shape=1)
is_training = tf.placeholder(tf.bool, shape=1)
output = basic_MLP(X,y,is_training,bits=8,threshold=0,dropout=False)
real_output = bit2real(output,bits=8)
difference = real_output - y
loss = tf.abs(difference)
optimizer = tf.train.AdamOptimizer(1e-3) 
train_step = optimizer.minimize(loss)


