import tensorflow as tf
# import PIL 
# from PIL import Image
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
    # Y = tf.placeholder(tf.float32, [None, 100000, 200, 1])
    is_training = tf.placeholder(tf.bool)
    
    with tf.variable_scope('Encoder') as enc: 
        latent_y = encoder(X, is_training)
    with tf.variable_scope('Decoder') as dec:
        output = decoder(latent_y, is_training)
    X_sum = tf.reduce_sum(X)
    l2_loss = tf.nn.l2_loss((output-X)/X_sum)/(100000.0*200.0)
    # l1_loss = l1_norm(output-X)
    # trans_loss = 0.9*l2_loss + 0.1*l1_loss
    mean_loss = tf.reduce_mean(l2_loss)
    tf.summary.scalar('loss', mean_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.rate)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    enc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Encoder')
    dec_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Decoder')
    with tf.control_dependencies(extra_update_ops):
        train_full = optimizer.minimize(mean_loss)
    
    sess = tf.Session(config=config)
    enc_saver = tf.train.Saver(var_list=enc_vars)
    dec_saver = tf.train.Saver(var_list=dec_vars)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tb',sess.graph)

    sess.run(tf.global_variables_initializer())
    _ = run_model(sess, X, is_training, mean_loss, U, S, 
              epochs=args.epochs, print_every=10, 
              training=train_full, plot_losses=False,
              writer=writer, sum_vars=merged)

    model_name = './CNN/CNN'
    enc_saver.save(sess, model_name+'_enc')
    dec_saver.save(sess, model_name+'_dec')

def load_data():
    with open('./Pickles/Sdata0.p', 'rb') as f:
        S = pickle.load(f)
    with open('./Pickles/Udata0.p', 'rb') as f: 
        U = pickle.load(f) 
    return U, S
# def load_data():
#     with open('./Pickles/Sdata0.p', 'rb') as f:
#         S = pickle.load(f)
#     with open('./Pickles/Udata0.p', 'rb') as f: 
#         U = pickle.load(f)
#     X = np.zeros([365,100000,200,1])
#     for day, u in U.items(): 
#         s = S[day]
#         m = u.shape[0]
#         s_ = np.tile(s, (m,1))
#         u_ = np.hstack([u, s_])
#         X[day, :m, :, :] = u_ 
#     return X

def run_model(session, X, is_training, loss_val, U, S, 
              epochs=1, batch_size=1, print_every=100,
              training=None, plot_losses=False,writer=None, sum_vars=None):
        
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [loss_val, training]
    if writer is not None: 
        variables.append(sum_vars)

    # counter 
    iter_cnt = 0
    for e in range(epochs):
        losses = []
        idxs = np.random.permutation(np.arange(365))
        # make sure we iterate over the dataset once
        for i in range(len(U)):
            # generate indicies for the batch
            # start_idx = (i*batch_size)%Xd.shape[0]
            # idx = train_indicies[start_idx:start_idx+batch_size]
            day = idxs[i]
            Xd = np.zeros([1,100000,200,1])
            u = U[day]
            s = S[day]
            m = u.shape[0]
            s_ = np.tile(s, (m,1))
            u_ = np.hstack([u, s_]).reshape([1,-1,200,1])
            Xd[:, :m, :, :] = u_ 
            # create a feed dictionary for this batch
            feed_dict = {X: Xd,
                         # Y: Yd[idx,:],
                         is_training: True}
            # get batch size
            actual_batch_size = 1
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            if writer is not None:
                # import pdb; pdb.set_trace()
                loss, _, summary = session.run(variables,feed_dict=feed_dict)
                # import pdb; pdb.set_trace()
                writer.add_summary(summary, iter_cnt)
            else:
                loss, _ = session.run(variables,feed_dict=feed_dict)
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            
            # print every now and then
            if (iter_cnt % print_every) == 0:
                print("Iteration %r: with minibatch training loss = %r " % (iter_cnt,loss))
            iter_cnt += 1
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {1}, Overall loss = {0:.3g}"\
              .format(total_loss,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss

def l1_norm(X):
    # X = tf.sqrt(X**2)
    X = tf.abs(X)
    norm = tf.reduce_sum(X)
    return norm 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CNN translation for given arguments')
    parser.add_argument('epochs', type=int)
    # parser.add_argument('batch_size', type=int) 
    parser.add_argument('rate', type=float) 
    # parser.add_argument('lam', type=float) 
    parser.add_argument('GPU', type=int) 
    args = parser.parse_args()
    main(args)
