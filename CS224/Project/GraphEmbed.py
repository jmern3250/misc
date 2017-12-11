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

    feed_dict = 

def load_data():
    with open('./Pickles/Sdata0.p', 'rb') as f:
        S = pickle.load(f)
    with open('./Pickles/Udata0.p', 'rb') as f: 
        U = pickle.load(f) 
    return U, S

def run_model(session, X, is_training, loss_val, U, S, 
              epochs=1, batch_size=1, print_every=100,
              training=None, plot_losses=False,writer=None, sum_vars=None):
        
    

def l1_norm(X):
    # X = tf.sqrt(X**2)
    X = tf.abs(X)
    norm = tf.reduce_sum(X)
    return norm 

if __name__ == '__main__':
    main()
