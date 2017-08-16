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


def encoder(X, is_training, data):
    c0 = bn_conv2d(X, is_training, 8, [9,9], 
                    strides=2, padding='valid',
                    activation='relu', name='c0')
    c1 = bn_conv2d(c0, is_training, 16, [3,3], 
                    strides=2, padding='valid',
                    activation='relu', name='c1')
    c2 = bn_conv2d(c1, is_training, 32, [3,3], 
                    strides=2, padding='valid',
                    activation='relu', name='c2')
    c3 = bn_conv2d(c2, is_training, 64, [3,3], 
                    strides=2, padding='valid',
                    activation='relu', name='c3')
    return c3

def decoder(feats, is_training, data):
    t0 = bn_conv2d_transpose(feats, is_training, 64, [4,4], 
                            strides=2, padding='valid',
                            activation='relu', name='t0')
    t1 = bn_conv2d_transpose(t0, is_training, 32, [4,4], 
                            strides=2, padding='valid',
                            activation='relu', name='t1')
    t2 = bn_conv2d_transpose(t1, is_training, 16, [4,4], 
                            strides=2, padding='valid',
                            activation='relu', name='t2')
    t3 = bn_conv2d_transpose(t2, is_training, 16, [3,3], 
                            strides=4, padding='valid',
                            activation='relu', name='t3')
    c_out_ = tf.layers.conv2d(
                            inputs=t3, 
                            filters=1,
                            kernel_size=[9,9],
                            strides=2,
                            padding='valid',
                            activation=tf.nn.relu,
                            name='cout_')
    c_out = tf.layers.conv2d(
                            inputs=c_out_,
                            filters=1,
                            kernel_size=[4,4],
                            strides=1,
                            padding='valid',
                            activation=tf.tanh,
                            name='cout')
    output_ = c_out + 1.0
    output = output_ * 0.5
    return output
