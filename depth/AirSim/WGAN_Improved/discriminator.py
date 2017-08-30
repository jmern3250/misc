import numpy as np
from numpy import matlib
import tensorflow as tf






def d_res_conv2d(X, filters, kernel_size, strides=1, name=None):
    if name is not None: 
        with tf.variable_scope(name):
            c1 = tf.layers.conv2d(
                                inputs=X, 
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same',
                                activation=tf.nn.relu,
                                name='c1')
            c2 = tf.layers.conv2d(
                                inputs=c1, 
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same',
                                activation=None,
                                name='c2')
            added = c2 + X

    else: 
        c1 = tf.layers.conv2d(
                            inputs=X, 
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same',
                            activation=tf.nn.relu,
                            name='c1')
        c2 = tf.layers.conv2d(
                            inputs=c1, 
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same',
                            activation=None,
                            name='c2')
        added = c2 + X
    return added


def discriminator(X, Y):
    inp = tf.concat([X,Y], axis=3)
    c0 = tf.layers.conv2d(
                        inputs=inp, 
                        filters=32, 
                        kernel_size=[9,9],
                        strides=4, 
                        padding='valid',
                        activation=tf.nn.relu, 
                        name='c0')
    c1 = tf.layers.conv2d(
                        inputs=c0, 
                        filters=64, 
                        kernel_size=[9,9],
                        strides=4, 
                        padding='valid',
                        activation=tf.nn.relu, 
                        name='c1')
    c2 = tf.layers.conv2d(
                        inputs=c1, 
                        filters=128, 
                        kernel_size=[9,9],
                        strides=4, 
                        padding='valid',
                        activation=tf.nn.relu, 
                        name='c2')
    r0 = d_res_conv2d(c2, 128, [3,3], strides=1, name='r0')
    r1 = d_res_conv2d(r0, 128, [3,3], strides=1, name='r1')
    output = tf.layers.conv2d(r1, 1, [3,3], 
                    strides=1, padding='same',
                    activation=None, name='output')
    return output
