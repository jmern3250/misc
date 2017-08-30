import numpy as np
from numpy import matlib
import tensorflow as tf


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
            # h2 = tf.nn.relu(added, name='h2')
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
    return added


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
    r3 = res_conv2d(r2, is_training, 128, [3,3], strides=1, name='r3')
    r4 = res_conv2d(r3, is_training, 128, [3,3], strides=1, name='r4')
    r5 = res_conv2d(r4, is_training, 128, [3,3], strides=1, name='r5')
    r6 = res_conv2d(r5, is_training, 128, [3,3], strides=1, name='r6')
    
    return [c0, c1, c2, r6]

def decoder(feats, is_training, data):
    c0, c1, c2, r4 = feats

    t0 = bn_conv2d_transpose(r4, is_training, 64, [5,5], 
                            strides=2, padding='valid',
                            activation='relu', name='t0')
    c2_ = bn_conv2d_transpose(c2, is_training, 64, [5,5], 
                            strides=2, padding='valid',
                            activation='relu', name='c2_') 
    t0_ = tf.concat([t0, c2_], axis=3, name='t0_')


    t1 = bn_conv2d_transpose(t0_, is_training, 32, [5,5], 
                            strides=2, padding='valid',
                            activation='relu', name='t1')
    c1_ = bn_conv2d_transpose(c1, is_training, 32, [9,9], 
                            strides=2, padding='valid',
                            activation='relu', name='c1_')
    t1_ = tf.concat([t1, c1_], axis=3, name='t1_')

    t2 = bn_conv2d_transpose(t1_, is_training, 16, [5,5], 
                            strides=2, padding='valid',
                            activation='relu', name='t2')
    c0_ = bn_conv2d_transpose(c0, is_training, 16, [17,17], 
                            strides=2, padding='valid',
                            activation='relu', name='c0_')
    t0_ = tf.concat([t2, c0_], axis=3, name='t2_')

    c_out = tf.layers.conv2d(
                            inputs=t0_, 
                            filters=1,
                            kernel_size=[9,9],
                            strides=1,
                            padding='valid',
                            activation=tf.tanh,
                            name='c0')
    return output