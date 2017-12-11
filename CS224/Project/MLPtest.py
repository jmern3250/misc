import tensorflow as tf
import pickle
import argparse

import numpy as np
import math

def main():
    Xd, Yd = load_data()
    # import pdb; pdb.set_trace()
    # if args.GPU == 0:
    #     config = tf.ConfigProto(
    #             device_count = {'GPU': 0}
    #             )
    # else:
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = False

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 28])
    # Y = tf.placeholder(tf.float32, [None, 1])
    output = MLP(X,None)

    sess = tf.Session()
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    model_name = './MLP/MLP'
    saver.restore(sess, model_name)

    feed_dict = {X:Xd}

    Y_ = sess.run(output, feed_dict)

    Y_ = Y_.squeeze()
    np.savetxt('MLPout0.csv', Y_, delimiter=',')
    import pdb; pdb.set_trace()
    
    
def load_data():
    data = np.loadtxt('./Data/extdata.csv', delimiter=',')
    Y = data[:,0].reshape([-1,1])
    X = data[:,1:]
    return X, Y 



def l1_norm(X):
    # X = tf.sqrt(X**2)
    X = tf.abs(X)
    norm = tf.reduce_sum(X)
    return norm

def MLP(X, regularizer=None):
    h1 = tf.layers.dense(X, 64, 
                        activation=tf.nn.relu, 
                        use_bias=True,
                        kernel_regularizer=regularizer,
                        name='h1'
                        )
    h2 = tf.layers.dense(h1, 128, 
                        activation=tf.nn.relu, 
                        use_bias=True,
                        kernel_regularizer=regularizer,
                        name='h2'
                        )
    h3 = tf.layers.dense(h2, 256, 
                        activation=tf.nn.relu, 
                        use_bias=True,
                        kernel_regularizer=regularizer,
                        name='h3'
                        )
    h4 = tf.layers.dense(h3, 512, 
                        activation=tf.nn.relu, 
                        use_bias=True,
                        kernel_regularizer=regularizer,
                        name='h4'
                        )
    h5 = tf.layers.dense(h4, 64, 
                        activation=tf.nn.relu, 
                        use_bias=True,
                        kernel_regularizer=regularizer,
                        name='h5'
                        )
    output = tf.layers.dense(h5, 1, 
                        activation=None, 
                        use_bias=True,
                        kernel_regularizer=regularizer,
                        name='output'
                        )
    return output 

if __name__ == '__main__':
    main()
