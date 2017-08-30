import tensorflow as tf
import PIL 
from PIL import Image
import scipy.io as sio
import glob 
import argparse

from model import * 
import autoencoder 

import numpy as np
from numpy import matlib
import math
import timeit

def main(args):
    X_train, Y_train = load_data(args.data)
    Y_train_ = np.stack([Y_train.squeeze()]*3,axis=3)
    # X_train_bw = (np.sum(X_train, axis=3)/(3)).reshape([-1,245,437,1])

    if args.GPU == 0:
        config = tf.ConfigProto(
                device_count = {'GPU': 0}
                )
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False

    tf.reset_default_graph()
    if args.data == 0:
        X = tf.placeholder(tf.float32, [None, 480, 640, 3])
        Y = tf.placeholder(tf.float32, [None, 480, 640, 1])
    elif args.data == 1:
        X = tf.placeholder(tf.float32, [None, 245, 437, 3])
        X_ = tf.placeholder(tf.float32, [None, 245, 437, 1])
        Y = tf.placeholder(tf.float32, [None, 245, 437, 1])
    is_training = tf.placeholder(tf.bool)
    
    with tf.variable_scope('Encoder') as enc: 
        latent_y = encoder(X, is_training, args.data)
    with tf.variable_scope('Decoder') as dec:
        output = decoder(latent_y, is_training, args.data)
    with tf.variable_scope('Loss_Encoder') as ln: 
        y_feats = autoencoder.encoder(output, is_training, args.data)
    with tf.variable_scope(ln, reuse=True):
        x_feats = autoencoder.encoder(Y, is_training, args.data)

    trans_loss = l1_norm(output-Y)
    reg_loss = TV_loss(output)
    feat_loss = gram_loss(y_feats[0], x_feats[0])
    feat_loss += gram_loss(y_feats[1], x_feats[1])
    feat_loss += gram_loss(y_feats[2], x_feats[2])
    feat_loss += gram_loss(y_feats[3], x_feats[3])

    # feat_loss = tf.nn.l2_loss(y_feats[0] - x_feats[0])
    # feat_loss += tf.nn.l2_loss(y_feats[1] - x_feats[1])
    # feat_loss += tf.nn.l2_loss(y_feats[2] - x_feats[2])
    # feat_loss += tf.nn.l2_loss(y_feats[3] - x_feats[3])

    # mean_loss = tf.reduce_mean(trans_loss + args.lam*feat_loss)
    mean_loss = tf.reduce_mean(trans_loss + 0.1*reg_loss + 10.0*feat_loss)
    tf.summary.scalar('loss', mean_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.rate)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    enc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Encoder')
    dec_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Decoder')
    loss_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Loss_Encoder')
    with tf.control_dependencies(extra_update_ops):
        train_full = optimizer.minimize(mean_loss,var_list=[enc_vars, dec_vars])
    
    sess = tf.Session(config=config)
    enc_saver = tf.train.Saver(var_list=enc_vars)
    dec_saver = tf.train.Saver(var_list=dec_vars)

    loss_saver = tf.train.Saver(var_list=loss_vars)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tb',sess.graph)

    sess.run(tf.global_variables_initializer())
    loss_saver.restore(sess, './loss_network/loss_network_enc')
    enc_saver.restore(sess, './trained_model/initial_model_enc')
    dec_saver.restore(sess, './trained_model/initial_model_dec')

    _ = run_model(sess, X, X_, Y, is_training, mean_loss, X_train, Y_train, Y_train, 
              epochs=args.epochs, batch_size=args.batch_size, 
              print_every=10, training=train_full, plot_losses=False,
              writer=writer, sum_vars=merged)

    model_name = './trained_model/final_model'
    # model_name += 'data_' + str(args.data)
    # model_name += '_epochs_' + str(args.epochs)
    # model_name += '_batchsize_' + str(args.batch_size)
    # model_name += '_rate_' + str(args.rate)
    enc_saver.save(sess, model_name+'_enc')
    dec_saver.save(sess, model_name+'_dec')

def load_data(data_idx, num=None):
    if data_idx == 0:
        mat_contents = sio.loadmat('./data/NYU_data.mat')
        X_train = mat_contents['images'].transpose([3,0,1,2])/255.0
        Y_train = mat_contents['depths'].transpose([2,0,1])
        Y_train = Y_train.reshape([1449,480,640,1])
        Y_train /= Y_train.max()
        Y_train -= 1
        Y_train *= -1.0
    elif data_idx == 1: 
        if num is not None: 
            TRAIN=num
        else:
            TRAIN=1963 

        X_train = np.zeros([TRAIN, 245, 437, 3])
        Y_train = np.zeros([TRAIN, 245, 437, 1])

        i = 0
        for filename in sorted(glob.glob('./data/AirSim/Scene/*.jpg')): 
            im=Image.open(filename)
            X_train[i,:,:,:] = np.array(im)[:,:,:]/255.0
            i += 1
            # import pdb; pdb.set_trace()
            if i == TRAIN: 
                break

        i = 0
        for filename in sorted(glob.glob('./data/AirSim/Depth/*.jpg')): 
            im=Image.open(filename)
            img_array = np.array(im)
            if img_array.ndim == 3:
                img_array = img_array[:,:,0]
            Y_train[i,:,:,0] = img_array[:,:]/255.0
            i += 1
            if i == TRAIN:
                break 

    return X_train, Y_train

def run_model(session, X, X_, Y, is_training, loss_val, Xd, Xd_, Yd, 
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False,writer=None, sum_vars=None):
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [loss_val, training]
    if writer is not None: 
        variables.append(sum_vars)

    # counter 
    iter_cnt = 0
    for e in range(epochs):
        losses = []
        
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         X_: Xd_[idx,:],
                         Y: Yd[idx,:],
                         is_training: True}
            # get batch size
            actual_batch_size = Yd[i:i+batch_size].shape[0]
            
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

def TV_loss(X):
    w = np.ones([3,3,1,1])*-1
    w[1,1,0,0] = 8
    # w /= 100.0
    # w = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])/50.0
    # w = w.reshape([3,3,1,1])
    W = tf.constant(w, dtype=tf.float32)
    edges = tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')
    loss = tf.reduce_sum(tf.abs(edges))
    return loss 

def gram_loss(X,Y):
    _, H_, W_, C_ = X.shape
    H = H_.value
    W = W_.value
    C = C_.value 
    psi_X = tf.reshape(X, [-1, H*W, C])
    gram_X = tf.matmul(tf.transpose(psi_X,[0,2,1]),psi_X)/(C*H*W)
    psi_Y = tf.reshape(Y, [-1, H*W, C])
    gram_Y = tf.matmul(tf.transpose(psi_Y,[0,2,1]),psi_Y)/(C*H*W)
    loss = tf.norm(gram_X-gram_Y)**2
    return loss 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CNN translation for given arguments')
    parser.add_argument('data', type=int) #0:NYU, 1:Airsim
    parser.add_argument('epochs', type=int)
    parser.add_argument('batch_size', type=int) 
    parser.add_argument('rate', type=float) 
    parser.add_argument('lam', type=float) 
    parser.add_argument('GPU', type=int) 
    args = parser.parse_args()
    main(args)
