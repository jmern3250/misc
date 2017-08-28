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
    return added


def discriminator(X, Y, is_training, data):
    inp = tf.concat([X,Y], axis=3)
    c0 = bn_conv2d(inp, is_training, 32, [9,9], 
                    strides=4, padding='valid',
                    activation='relu', name='c0')
    c1 = bn_conv2d(c0, is_training, 64, [9,9], 
                    strides=4, padding='valid',
                    activation='relu', name='c1')
    c2 = bn_conv2d(c1, is_training, 128, [9,9], 
                    strides=4, padding='valid',
                    activation='relu', name='c2')
    r0 = res_conv2d(c2, is_training, 128, [3,3], strides=1, name='r0')
    r1 = res_conv2d(r0, is_training, 128, [3,3], strides=1, name='r1')
    output = tf.layers.conv2d(r1, 1, [3,3], 
                    strides=1, padding='same',
                    activation=None, name='output')
    # output += 1.0
    # output *= 0.5
    return output
