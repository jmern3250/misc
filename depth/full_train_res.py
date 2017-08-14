import tensorflow as tf
import PIL 
from PIL import Image
import scipy.io as sio
import glob 
import argparse

import numpy as np
from numpy import matlib
import math
import timeit

def main(args):
    X_train, Y_train = load_data(args.data)
    # Y_train_ = np.stack([Y_train.squeeze()]*3,axis=3)
    # dev = tf.device('/cpu:0')
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
        # Y_ = tf.placeholder(tf.float32, [None, 245, 437, 3])
        Y = tf.placeholder(tf.float32, [None, 245, 437, 1])
    is_training = tf.placeholder(tf.bool)
    
    with tf.variable_scope('Encoder') as enc: 
        latent_y = encoder(X, is_training, args.data)
    with tf.variable_scope('Decoder') as dec:
        output = decoder(latent_y, is_training, args.data)
    # with tf.variable_scope(enc, reuse=True): 
    #     latent_x = encoder(X, is_training, args.data)

    trans_loss = tf.nn.l2_loss(output-Y)
    # feat_loss = tf.nn.l2_loss(latent_x-latent_y)
    # total_loss = trans_loss + feat_loss*args.lam    

    # mean_loss = tf.reduce_mean(total_loss)
    mean_loss = tf.reduce_mean(trans_loss)
    tf.summary.scalar('loss', mean_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.rate)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    enc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Encoder')
    dec_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Decoder')
    with tf.control_dependencies(extra_update_ops):
        train_full = optimizer.minimize(mean_loss)
        train_enc = optimizer.minimize(mean_loss, var_list=enc_vars)
    
    sess = tf.Session(config=config)
    enc_saver = tf.train.Saver(var_list=enc_vars)
    dec_saver = tf.train.Saver(var_list=dec_vars)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tb',sess.graph)

    sess.run(tf.global_variables_initializer())
    enc_saver.restore(sess, './E_Model/E_data_1_epochs_20_batchsize_10_rate_0.001_enc')
    dec_saver.restore(sess, './E_Model/E_data_1_epochs_20_batchsize_10_rate_0.001_dec')
    _ = run_model(sess, X, Y, is_training, mean_loss, X_train, Y_train, 
              epochs=args.epochs, batch_size=args.batch_size, 
              print_every=10, training=train_full, plot_losses=False,
              writer=writer, sum_vars=merged)
    model_name = './T_Model/T_'
    model_name += 'data_' + str(args.data)
    model_name += '_epochs_' + str(args.epochs)
    model_name += '_batchsize_' + str(args.batch_size)
    model_name += '_rate_' + str(args.rate)
    # model_name += '_lambda_' + str(args.lam)
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

def run_model(session, X, Y, is_training, loss_val, Xd, Yd, 
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


def bn_conv2d(X, is_training, filters, kernel_size, strides=1, padding='valid',activation='relu', name=None):
    if name is not None: 
        with tf.variable_scope(name):
            c1 = tf.layers.conv2d(
                                inputs=X, 
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                activation=None,
                                name='c1')
            bn1 = tf.layers.batch_normalization(
                            c1, training=is_training,
                            renorm=True,  
                            name='bn1')
            if activation == 'relu':
                h1 = tf.nn.relu(bn1, name='h1')
            elif activation =='tanh':
                h1 = tf.tanh(bn1, name='h1')
            else: 
                h1 = bn1
    else:
        c1 = tf.layers.conv2d(
                            inputs=X, 
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            activation=None,
                            name='c1')
        bn1 = tf.layers.batch_normalization(
                        c1, training=is_training,
                        renorm=True,  
                        name='bn1')
        if activation == 'relu':
            h1 = tf.nn.relu(bn1, name='h1')
        elif activation =='tanh':
            h1 = tf.tanh(bn1, name='h1')
        else: 
            h1 = bn1
    return h1

def bn_conv2d_transpose(X, is_training, filters, kernel_size, strides=1, padding='valid',activation='relu', name=None):
    if name is not None: 
        with tf.variable_scope(name):
            c1 = tf.layers.conv2d_transpose(
                                inputs=X, 
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                activation=None,
                                name='c1')
            bn1 = tf.layers.batch_normalization(
                            c1, training=is_training,
                            renorm=True,  
                            name='bn1')
            if activation == 'relu':
                h1 = tf.nn.relu(bn1, name='h1')
            elif activation =='tanh':
                h1 = tf.tanh(bn1, name='h1')
            else: 
                h1 = bn1
    else:
        c1 = tf.layers.conv2d_transpose(
                            inputs=X, 
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            activation=None,
                            name='c1')
        bn1 = tf.layers.batch_normalization(
                        c1, training=is_training,
                        renorm=True,  
                        name='bn1')
        if activation == 'relu':
            h1 = tf.nn.relu(bn1, name='h1')
        elif activation =='tanh':
            h1 = tf.tanh(bn1, name='h1')
        else: 
            h1 = bn1
    return h1

def res_conv2d(X, is_training, filters, kernel_size, strides=1, name=None):
    if name is not None: 
        with tf.variable_scope(name):
            c1 = tf.layers.conv2d(
                                inputs=X, 
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same',
                                activation=None,
                                name='c1')
            bn1 = tf.layers.batch_normalization(
                            c1, training=is_training,
                            renorm=True,  
                            name='bn1')
            h1 = tf.nn.relu(bn1, name='h1')
            c2 = tf.layers.conv2d(
                                inputs=h1, 
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same',
                                activation=None,
                                name='c2')
            bn2 = tf.layers.batch_normalization(
                            c2, training=is_training,
                            renorm=True,  
                            name='bn2')
            added = bn2 + X
            h2 = tf.nn.relu(added, name='h2')
    else: 
        c1 = tf.layers.conv2d(
                            inputs=X, 
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same',
                            activation=None,
                            name='c1')
        bn1 = tf.layers.batch_normalization(
                        c1, training=is_training,
                        renorm=True,  
                        name='bn1')
        h1 = tf.nn.relu(bn1, name='h1')
        c2 = tf.layers.conv2d(
                            inputs=h1, 
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same',
                            activation=None,
                            name='c2')
        bn2 = tf.layers.batch_normalization(
                        c2, training=is_training,
                        renorm=True,  
                        name='bn2')
        added = bn2 + X
        h2 = tf.nn.relu(added, name='h2')
    return h2 


def encoder(X, is_training, data):
    c0 = bn_conv2d(X, is_training, 32, [9,9], 
                    strides=2, padding='valid',
                    activation='relu', name='c0')
    c1 = bn_conv2d(c0, is_training, 64, [3,3], 
                    strides=2, padding='valid',
                    activation='relu', name='c1')
    c2 = bn_conv2d(c1, is_training, 128, [3,3], 
                    strides=2, padding='valid',
                    activation='relu', name='c2')
    r0 = res_conv2d(c2, is_training, 128, [3,3], strides=1, name='r0')
    r1 = res_conv2d(r0, is_training, 128, [3,3], strides=1, name='r1')
    r2 = res_conv2d(r1, is_training, 128, [3,3], strides=1, name='r2')
    
    return r2

def decoder(feats, is_training, data):
    t0 = bn_conv2d_transpose(feats, is_training, 64, [3,3], 
                            strides=2, padding='valid',
                            activation='relu', name='t0')
    t1 = bn_conv2d_transpose(t0, is_training, 32, [5,5], 
                            strides=2, padding='valid',
                            activation='relu', name='t1')
    t2 = bn_conv2d_transpose(t1, is_training, 16, [10,10], 
                            strides=4, padding='valid',
                            activation='relu', name='t2')
    c0 = tf.layers.conv2d(
                            inputs=t2, 
                            filters=1,
                            kernel_size=[2,2],
                            strides=2,
                            padding='valid',
                            activation=tf.tanh,
                            name='c0')
    output_ = c0 + 1.0
    output = output_ * 0.5
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CNN translation for given arguments')
    parser.add_argument('data', type=int) #0:NYU, 1:Airsim
    parser.add_argument('epochs', type=int) #0:NYU, 1:Airsim
    parser.add_argument('batch_size', type=int) #0:NYU, 1:Airsim
    parser.add_argument('rate', type=float) 
    # parser.add_argument('lam', type=float) 
    parser.add_argument('GPU', type=int) 
    args = parser.parse_args()
    main(args)
