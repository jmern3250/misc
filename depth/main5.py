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
    # dev = tf.device('/cpu:0')
    if args.GPU == 0:
        config = tf.ConfigProto(
                device_count = {'GPU': 0}
                )
    else:
        config = None

    tf.reset_default_graph()
    if args.data == 0:
        X = tf.placeholder(tf.float32, [None, 480, 640, 3])
        Y = tf.placeholder(tf.float32, [None, 480, 640, 1])
    elif args.data == 1:
        X = tf.placeholder(tf.float32, [None, 240, 420, 3])
        Y = tf.placeholder(tf.float32, [None, 240, 420, 1])
    LR = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    
    output = DACNet(X,args.data)
    
    loss = tf.nn.l2_loss(output-Y)
    mean_loss = tf.reduce_mean(loss)
    tf.summary.scalar('loss', mean_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.rate)
    train_step = optimizer.minimize(mean_loss)
    
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tb',sess.graph)
    # import pdb; pdb.set_trace()

    sess.run(tf.global_variables_initializer())
    # import pdb; pdb.set_trace()
    _ = run_model(sess, X, Y, is_training, mean_loss, X_train, Y_train, 
              epochs=args.epochs, batch_size=args.batch_size, 
              print_every=10, decay=args.decay,
              training=train_step, plot_losses=False, writer=writer, sum_vars=merged)
    model_name = './Models/'
    model_name += 'data_' + str(args.data)
    model_name += '_epochs_' + str(args.epochs)
    model_name += '_batchsize_' + str(args.batch_size)
    model_name += '_rate_' + str(args.rate)
    model_name += '_decay_' + str(args.decay)
    saver.save(sess, model_name)

def load_data(data_idx):
    if data_idx == 0:
        mat_contents = sio.loadmat('./data/NYU_data.mat')
        X_train = mat_contents['images'].transpose([3,0,1,2])/255.0
        Y_train = mat_contents['depths'].transpose([2,0,1])
        Y_train = Y_train.reshape([1449,480,640,1])
        Y_train /= 1.0 - Y_train.max()
    elif data_idx == 1: 
        TRAIN=2748

        X_train = np.zeros([TRAIN, 240, 420, 3])
        Y_train = np.zeros([TRAIN, 240, 420, 1])

        i = 0
        for filename in glob.glob('/home/jmern91/Python_add/misc/depth/data/AirSim/scene/train/*'): #assuming gif
            im=Image.open(filename)
            X_train[i,:,:,:] = np.array(im)[:240,:420,:]/255.0
            i += 1

        i = 0
        for filename in glob.glob('/home/jmern91/Python_add/misc/depth/data/AirSim/depth/train/*'): #assuming gif
            im=Image.open(filename)
            img_array = np.array(im)
            if img_array.ndim == 3:
                img_array = img_array[:,:,0]
            Y_train[i,:,:,0] = img_array[:240,:420]/255.0
            i += 1

    return X_train, Y_train

def run_model(session, X, Y, is_training, loss_val, X_train, Y_train, 
              epochs=1, batch_size=64, print_every=100, decay=None,
              training=None, plot_losses=False,writer=None,sum_vars=None):
    
    # shuffle indicies
    train_indicies = np.arange(X_train.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [loss_val]
    if training_now:
        variables.append(training)

    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses
        losses = []
        # Decay learning rate
        # if decay is not None: 
        #     L_rate *= decay
        
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(X_train.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%X_train.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: X_train[idx,...],
                         Y: Y_train[idx,...],
                         is_training: training_now}
            # import pdb; pdb.set_trace()
            # get batch size
            actual_batch_size = Y_train[i:i+batch_size].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            if training_now:
                if writer is not None:
                    loss_, summary = session.run([variables,sum_vars],feed_dict=feed_dict)
                    loss = loss_[0]
                    # import pdb; pdb.set_trace()
                    writer.add_summary(summary)
                else: 
                    loss = session.run([variables],feed_dict=feed_dict)
            else: 
                if writer is not None: 
                    loss, summary = session.run([variables,sum_vars],feed_dict=feed_dict)
                    writer.add_summary(summary)
                else:
                    loss = session.run(variables,feed_dict=feed_dict)
            # aggregate performance stats

            losses.append(loss*actual_batch_size)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration %r: with minibatch training loss = %r " % (iter_cnt,loss))
            iter_cnt += 1

        total_loss = np.sum(losses)/X_train.shape[0]
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

def DACNet(X,data):

    c1 = tf.layers.conv2d(
                        inputs=X, 
                        filters=64,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        name='c1')
    p1 = tf.nn.max_pool(c1, [1,2,2,1],[1,2,2,1],'VALID')

    c2 = tf.layers.conv2d(
                        inputs=p1, 
                        filters=128,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        name='c2')
    p2 = tf.nn.max_pool(c2, [1,2,2,1],[1,2,2,1],'VALID')

    c3 = tf.layers.conv2d(
                        inputs=p2, 
                        filters=256,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        name='c3')
    p3 = tf.nn.max_pool(c3, [1,2,2,1],[1,2,2,1],'VALID')

    c4 = tf.layers.conv2d(
                        inputs=p3, 
                        filters=512,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        name='c4')
    p4 = tf.nn.max_pool(c4, [1,2,2,1],[1,2,2,1],'VALID')
    c5 = tf.layers.conv2d(
                        inputs=p4, 
                        filters=512,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        name='c5')

    
    if data == 0:
        tc5 = tf.layers.conv2d_transpose(
                inputs=c5,
                filters=512,
                kernel_size=[2,2],
                strides=4,
                activation=tf.nn.relu,
                name='up1'
        )
        p3_ = tf.layers.conv2d_transpose(
                inputs=p3,
                filters=512,
                kernel_size=[2,2],
                strides=2,
                activation=tf.nn.relu,
                name='up2'
        )
        p2_ = tf.layers.conv2d(
                inputs=p2,
                filters=512,
                kernel_size=[1,1],
                strides=1,
                padding='valid',
                activation=tf.nn.relu, 
                name='c6')
    else:
        tc5 = tf.layers.conv2d_transpose(
                inputs=c5,
                filters=512,
                kernel_size=[2,5],
                strides=4,
                activation=tf.nn.relu,
                name='up1'
        )
        p3_ = tf.layers.conv2d_transpose(
                inputs=p3,
                filters=512,
                kernel_size=[2,3],
                strides=2,
                activation=tf.nn.relu,
                name='up2'
        )
        p2_ = tf.layers.conv2d(
                inputs=p2,
                filters=512,
                kernel_size=[1,1],
                strides=1,
                padding='valid',
                activation=tf.nn.relu,
                name='c6')

    # print(p3_.shape)
    # print(p2_.shape)
    # print(tc5.shape)
    # composite = p3_ + p2_ + tc5
    composite = tf.concat([p3_, p2_, tc5], axis=3)
    upscale = tf.layers.conv2d_transpose(
            inputs=composite,
            filters=500,
            kernel_size=[2,2],
            strides=8,
            activation=tf.nn.relu,
            name='up3'
    )
    output = tf.layers.conv2d(
            inputs=upscale,
            filters=1,
            kernel_size=[2,2],
            strides=2,
            activation=tf.nn.relu, 
            name='c7'
    )
    # print(X.shape)
    # print(output.shape)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CNN translation for given arguments')
    parser.add_argument('data', type=int) #0:NYU, 1:Airsim
    parser.add_argument('epochs', type=int) #0:NYU, 1:Airsim
    parser.add_argument('batch_size', type=int) #0:NYU, 1:Airsim
    parser.add_argument('rate', type=float) #0:NYU, 1:Airsim
    parser.add_argument('decay', type=float) 
    parser.add_argument('GPU', type=int) 
    args = parser.parse_args()
    main(args)