import tensorflow as tf
import scipy.io as sio
import pickle
import glob 
import argparse

from autoencoder import * 

import numpy as np
from numpy import matlib
import math
import timeit

def main(args):
    U, S = load_data()

    if args.GPU == 0:
        config = tf.ConfigProto(
                device_count = {'GPU': 0}
                )
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 100000, 200, 1])
    
    with tf.variable_scope('Encoder') as enc: 
        latent_y = encoder(X, is_training)
    
    sess = tf.Session(config=config)
    enc_saver = tf.train.Saver(var_list=enc_vars)
    

    model_name = './CNN/CNN'
    enc_saver.restore(sess, model_name+'_enc')

    E = run_model(sess, X, U, S)
    np.savetxt('Embeddings.csv', E, delimiter=',')

    

def load_data():
    with open('./Pickles/Sdata0.p', 'rb') as f:
        S = pickle.load(f)
    with open('./Pickles/Udata0.p', 'rb') as f: 
        U = pickle.load(f) 
    return U, S

def run_model(session, X, U, S):
    m = len(U)
    E = np.zeros([m, 128])
    for day in range(m):
        Xd = np.zeros([1,100000,200,1])
        u = U[day]
        s = S[day]
        m = u.shape[0]
        s_ = np.tile(s, (m,1))
        u_ = np.hstack([u, s_]).reshape([1,-1,200,1])
        Xd[:, :m, :, :] = u_ 
        feed_dict = {X:Xd}
        e = session.run(latent_y,feed_dict=feed_dict)
        E[i,:] = e.squeeze()
    return E


        
    

def l1_norm(X):
    # X = tf.sqrt(X**2)
    X = tf.abs(X)
    norm = tf.reduce_sum(X)
    return norm 

if __name__ == '__main__':
    main()
