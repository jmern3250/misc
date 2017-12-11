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
        bn1 = instance_norm(c1, name='bn1')
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
    return added


def encoder(X, is_training):
    c0 = bn_conv2d(X, is_training, 1, [1000,101], 
                    strides=[100,1], padding='valid',
                    activation='relu', name='c0')
    c1 = bn_conv2d(c0, is_training, 16, [100,5], 
                    strides=[10,2], padding='valid',
                    activation='relu', name='c1')
    c2 = bn_conv2d(c1, is_training, 32, [5,5], 
                    strides=2, padding='valid',
                    activation='relu', name='c2')
    r = bn_conv2d(c2, is_training, 128, [43,22], 
                    strides=1, padding='valid',
                    activation='relu', name='r')
    # print('C0: %r' % c0.shape)
    # print('C1: %r' % c1.shape)
    # print('C2: %r' % c2.shape)
    # print('R: %r' % r.shape)
    return [c0, c1, c2, r]

def decoder(feats, is_training):
    c0, c1, c2, r = feats

    t0 = bn_conv2d_transpose(r, is_training, 32, [43,22], 
                            strides=1, padding='valid',
                            activation='relu', name='t0')
    t1 = bn_conv2d_transpose(t0, is_training, 16, [6,6], 
                            strides=2, padding='valid',
                            activation='relu', name='t1')
    t2 = bn_conv2d_transpose(t1, is_training, 1, [101,6], 
                            strides=[10,2], padding='valid',
                            activation='relu', name='t2')
    output = bn_conv2d_transpose(t2, is_training, 1, [1000,101], 
                            strides=[100,1], padding='valid',
                            activation='relu', name='output')
    # print('T0: %r' % t0.shape)
    # print('T1: %r' % t1.shape)
    # print('T2: %r' % t2.shape)
    # print('OUT: %r' % output.shape)
    return output
