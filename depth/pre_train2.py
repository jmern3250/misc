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
    def run_model(session, loss_val, Xd, Yd, 
                  epochs=1, batch_size=64, print_every=100,
                  training=None, plot_losses=False):
        
        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = training is not None
        
        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [loss_val]
        if training_now:
            variables.append(training)
    #     pdb.set_trace()
        # counter 
        iter_cnt = 0
        for e in range(epochs):
            # keep track of losses
            losses = []
            
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
                # generate indicies for the batch
                start_idx = (i*batch_size)%Xd.shape[0]
                idx = train_indicies[start_idx:start_idx+batch_size]
                
                # create a feed dictionary for this batch
                feed_dict = {X: Xd[idx,:],
                             Y: Yd[idx,:],
                             is_training: training_now}
                # get batch size
                actual_batch_size = Yd[i:i+batch_size].shape[0]
                
                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                if training_now:
                    loss, _ = session.run(variables,feed_dict=feed_dict)
                else: 
                    loss = session.run(variables,feed_dict=feed_dict)
                # aggregate performance stats
    #             pdb.set_trace()
                losses.append(loss*actual_batch_size)
                
                # print every now and then
                if training_now and (iter_cnt % print_every) == 0:
                    print("Iteration %r: with minibatch training loss = %r " % (iter_cnt,loss))
                iter_cnt += 1
    #         pdb.set_trace()
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
    LR = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    
    output = DACNet(X, is_training, args.data)
    
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
    _ = run_model(sess, mean_loss, Y_train_, Y_train, 
              epochs=args.epochs, batch_size=args.batch_size, 
              print_every=100, training=train_step, plot_losses=False)
    model_name = './Models/PT_'
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
        Y_train /= Y_train.max()
        Y_train -= 1
        Y_train *= -1.0
    elif data_idx == 1: 
        TRAIN=2748

        X_train = np.zeros([TRAIN, 245, 437, 3])
        Y_train = np.zeros([TRAIN, 245, 437, 1])

        i = 0
        for filename in glob.glob('./data/AirSim/scene/train/*'): #assuming gif
            im=Image.open(filename)
            X_train[i,:,:,:] = np.array(im)[:,:,:]/255.0
            i += 1

        i = 0
        for filename in glob.glob('./data/AirSim/depth/train/*'): #assuming gif
            im=Image.open(filename)
            img_array = np.array(im)
            if img_array.ndim == 3:
                img_array = img_array[:,:,0]
            Y_train[i,:,:,0] = img_array[:,:]/255.0
            i += 1

    return X_train, Y_train


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

def DACNet(X, is_training, data):

    c1 = tf.layers.conv2d(
                        inputs=X, 
                        filters=32,
                        kernel_size=[2,2],
                        strides=2,
                        padding='valid',
                        name='c1')
    bn1 = tf.layers.batch_normalization(
                    c1, training=is_training, name='bn1')

    c2 = tf.layers.conv2d(
                        inputs=bn1, 
                        filters=64,
                        kernel_size=[2,2],
                        strides=2,
                        padding='valid',
                        name='c2')
    bn2 = tf.layers.batch_normalization(
                    c2, training=is_training, name='bn2')

    c3 = tf.layers.conv2d(
                        inputs=bn2, 
                        filters=128,
                        kernel_size=3,
                        strides=2,
                        padding='valid',
                        name='c3')
    bn3 = tf.layers.batch_normalization(
                    c3, training=is_training, name='bn3')

    c4 = tf.layers.conv2d(
                        inputs=bn3, 
                        filters=256,
                        kernel_size=[1,1],
                        strides=2,
                        padding='valid',
                        name='c4')
    bn4 = tf.layers.batch_normalization(
                    c4, training=is_training, name='bn4')
    
    if data == 0:
        tc5 = tf.layers.conv2d_transpose(
                inputs=bn4,
                filters=256,
                kernel_size=[2,2],
                strides=4,
                activation=tf.nn.relu,
                padding='valid',
                name='up1'
        )
        bn3_ = tf.layers.conv2d_transpose(
                inputs=bn3,
                filters=256,
                kernel_size=[4,4],
                strides=2,
                activation=tf.nn.relu,
                name='up2'
        )
        bn2_ = tf.layers.conv2d(
                inputs=bn2,
                filters=256,
                kernel_size=[1,1],
                strides=1,
                padding='valid',
                activation=tf.nn.relu, 
                name='c6')
    else:
        tc5 = tf.layers.conv2d_transpose(
                inputs=bn4,
                filters=256,
                kernel_size=[5,5],
                strides=4,
                activation=tf.nn.relu,
                padding='valid',
                name='up1'
        )
        bn3_ = tf.layers.conv2d_transpose(
                inputs=bn3,
                filters=256,
                kernel_size=[3,3],
                strides=2,
                activation=tf.nn.relu,
                name='up2'
        )
        bn2_ = tf.layers.conv2d(
                inputs=bn2,
                filters=256,
                kernel_size=[1,1],
                strides=1,
                padding='valid',
                activation=tf.nn.relu,
                name='c6')

    # print(tc5.shape)
    # print(bn3_.shape)
    # print(bn2_.shape)

    composite = bn3_ + bn2_ + tc5
    # composite = tf.concat([bn3_, bn2_, tc5], axis=3)

    up4 = tf.layers.conv2d_transpose(
            inputs=composite,
            filters=256,
            kernel_size=[2,2],
            strides=2,
            activation=tf.nn.relu,
            name='up4'
    )
    if data == 0:
        up5 = tf.layers.conv2d_transpose(
                inputs=up4,
                filters=128,
                kernel_size=[2,2],
                strides=2,
                activation=tf.nn.relu,
                name='up5'
        )
        up6 = tf.layers.conv2d_transpose(
                inputs=up5,
                filters=64,
                kernel_size=[2,2],
                strides=2,
                activation=tf.nn.relu,
                name='up6'
        )
    else: 
        up5 = tf.layers.conv2d_transpose(
                inputs=up4,
                filters=128,
                kernel_size=[3,3],
                strides=2,
                activation=tf.nn.relu,
                name='up5'
        )
        up6 = tf.layers.conv2d_transpose(
                inputs=up5,
                filters=64,
                kernel_size=[3,3],
                strides=2,
                activation=tf.nn.relu,
                name='up6'
        )

    output = tf.layers.conv2d(
            inputs=up6,
            filters=1,
            kernel_size=[2,2],
            strides=2,
            activation=tf.nn.relu, 
            name='c7'
    )
    print(X.shape)
    print(output.shape)
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
