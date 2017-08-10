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
    Y_train_ = np.stack([Y_train.squeeze()]*3,axis=3)
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
        Y = tf.placeholder(tf.float32, [None, 245, 437, 1])
    is_training = tf.placeholder(tf.bool)
    
    with tf.variable_scope('Encoder') as enc: 
        hc = encoder(X, is_training, args.data)
    with tf.variable_scope('Decoder') as dec:
        output = decoder(hc, is_training, args.data)

    # import pdb; pdb.set_trace+()
    loss = tf.nn.l2_loss(output-Y)
    mean_loss = tf.reduce_mean(loss)
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
    _ = run_model(sess, X, Y, is_training, mean_loss, Y_train_, Y_train, 
              epochs=args.epochs, batch_size=args.batch_size, 
              print_every=10, training=train_full, plot_losses=False,
              writer=writer, sum_vars=merged)
    model_name = './Models/PT_'
    model_name += 'data_' + str(args.data)
    model_name += '_epochs_' + str(args.epochs)
    model_name += '_batchsize_' + str(args.batch_size)
    model_name += '_rate_' + str(args.rate)
    model_name += '_decay_' + str(args.decay)
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
        for filename in glob.glob('./data/AirSim/Scene/*'): 
            im=Image.open(filename)
            X_train[i,:,:,:] = np.array(im)[:,:,:]/255.0
            i += 1
            # import pdb; pdb.set_trace()
            if i == TRAIN: 
                break

        i = 0
        for filename in glob.glob('./data/AirSim/Depth/*'): 
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


def conv_group(X, conv_params):
    ''' Convolution Parameters in Dictionary: 
        Kernel Size
        Number of Filters
        Stride 
        Padding (Valid/Same)
        Activation (Not yet )'''
    layers = conv_params['layers']
    filters = conv_params['filters']
    kernel_size = conv_params['size']
    stride = conv_params['stride']
    padding = conv_params['pad']
    conv_layers = {}
    for i in range(layers): 
        if i == 0:
            conv_layers[i] = tf.layers.conv2d(
                                inputs=X,
                                filters=filters,
                                kernel_size=kernel_size,
                                padding=padding,
                                activation=tf.nn.relu
                                )
        else:
            conv_layers[i] = tf.layers.conv2d(
                                inputs=conv_layers[i-1],
                                filters=filters,
                                kernel_size=kernel_size,
                                padding=padding,
                                activation=tf.nn.relu
                                )
    return conv_layers[layers-1]  

def encoder(X, is_training, data):

    c1 = tf.layers.conv2d(
                        inputs=X, 
                        filters=32,
                        kernel_size=[2,2],
                        strides=2,
                        padding='valid',
                        activation=None,
                        name='c1')
    bn1 = tf.layers.batch_normalization(
                    c1, training=is_training,
                    renorm=True,  
                    name='bn1')
    h1 = tf.nn.relu(bn1, name='h1')

    c2 = tf.layers.conv2d(
                        inputs=h1, 
                        filters=64,
                        kernel_size=[2,2],
                        strides=2,
                        padding='valid',
                        activation=None,
                        name='c2')
    bn2 = tf.layers.batch_normalization(
                    c2, training=is_training, 
                    renorm=True,
                    name='bn2')
    h2 = tf.nn.relu(bn2, name='h2')

    c3 = tf.layers.conv2d(
                        inputs=h2, 
                        filters=128,
                        kernel_size=3,
                        strides=2,
                        padding='valid',
                        activation=None,
                        name='c3')
    bn3 = tf.layers.batch_normalization(
                    c3, training=is_training, 
                    renorm=True,
                    name='bn3')
    h3 = tf.nn.relu(bn3, name='h3')

    c4 = tf.layers.conv2d(
                        inputs=h3, 
                        filters=256,
                        kernel_size=[1,1],
                        strides=2,
                        padding='valid',
                        activation=None,
                        name='c4')
    bn4 = tf.layers.batch_normalization(
                    c4, training=is_training, 
                    renorm=True,
                    name='bn4')
    h4 = tf.nn.relu(bn4, name='h4')
    
    if data == 0:
        tc5 = tf.layers.conv2d_transpose(
                inputs=h4,
                filters=256,
                kernel_size=[2,2],
                strides=4,
                activation=None,
                padding='valid',
                name='up1'
        )
        bn3_ = tf.layers.conv2d_transpose(
                inputs=h3,
                filters=256,
                kernel_size=[4,4],
                strides=2,
                activation=None,
                name='up2'
        )
        bn2_ = tf.layers.conv2d(
                inputs=h2,
                filters=256,
                kernel_size=[1,1],
                strides=1,
                padding='valid',
                activation=None, 
                name='c6')
    else:
        tc5 = tf.layers.conv2d_transpose(
                inputs=bn4,
                filters=256,
                kernel_size=[5,5],
                strides=4,
                activation=None,
                padding='valid',
                name='up1'
        )
        bn3_ = tf.layers.conv2d_transpose(
                inputs=bn3,
                filters=256,
                kernel_size=[3,3],
                strides=2,
                activation=None,
                name='up2'
        )
        bn2_ = tf.layers.conv2d(
                inputs=bn2,
                filters=256,
                kernel_size=[1,1],
                strides=1,
                padding='valid',
                activation=None,
                name='c6')

    # print(tc5.shape)
    # print(bn3_.shape)
    # print(bn2_.shape)

    composite = bn3_ + bn2_ + tc5
    # composite = tf.concat([bn3_, bn2_, tc5], axis=3)
    bnc = tf.layers.batch_normalization(
                    composite, training=is_training,
                    renorm=True, name = 'bnc')
    hc = tf.nn.relu(bnc, name='hc')
    return hc 

def decoder(hc, is_training, data):
    up4 = tf.layers.conv2d_transpose(
            inputs=hc,
            filters=256,
            kernel_size=[2,2],
            strides=2,
            activation=None,
            name='up4'
    )
    bnu4 = tf.layers.batch_normalization(
                    up4, training=is_training,
                    renorm=True, name = 'bnu4')
    hu4 = tf.nn.relu(bnu4, name='hu4')
    if data == 0:
        up5 = tf.layers.conv2d_transpose(
                inputs=hu4,
                filters=128,
                kernel_size=[2,2],
                strides=2,
                activation=None,
                name='up5'
        )
        bnu5 = tf.layers.batch_normalization(
                    up5, training=is_training,
                    renorm=True, name = 'bnu5')
        hu5 = tf.nn.relu(bnu5, name='hu5')

        up6 = tf.layers.conv2d_transpose(
                inputs=hu5,
                filters=64,
                kernel_size=[2,2],
                strides=2,
                activation=None,
                name='up6'
        )
        bnu6 = tf.layers.batch_normalization(
                    up6, training=is_training,
                    renorm=True, name = 'bnu6')
        hu6 = tf.tanh(bnu6, name='hu6')
    else: 
        up5 = tf.layers.conv2d_transpose(
                inputs=hu4,
                filters=128,
                kernel_size=[3,3],
                strides=2,
                activation=None,
                name='up5'
        )
        bnu5 = tf.layers.batch_normalization(
                    up5, training=is_training,
                    renorm=True, name = 'bnu5')
        hu5 = tf.nn.relu(bnu5, name='hu5')
        up6 = tf.layers.conv2d_transpose(
                inputs=hu5,
                filters=64,
                kernel_size=[3,3],
                strides=2,
                activation=None,
                name='up6'
        )
        bnu6 = tf.layers.batch_normalization(
                    up6, training=is_training,
                    renorm=True, name = 'bnu6')
        hu6 = tf.tanh(bnu6, name='hu6')

    output = tf.layers.conv2d(
            inputs=hu6,
            filters=1,
            kernel_size=[2,2],
            strides=2,
            activation=None, 
            name='c7'
    )
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CNN translation for given arguments')
    parser.add_argument('data', type=int) #0:NYU, 1:Airsim
    parser.add_argument('epochs', type=int) #0:NYU, 1:Airsim
    parser.add_argument('batch_size', type=int) #0:NYU, 1:Airsim
    parser.add_argument('rate', type=float) #0:NYU, 1:Airsim
    parser.add_argument('GPU', type=int) 
    args = parser.parse_args()
    main(args)
