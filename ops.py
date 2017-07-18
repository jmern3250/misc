import tensorflow as tf

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(x, leak*x)

def bn(x, phase, center=True, scale=True, name='batch_norm'):
    return tf.contrib.layers.batch_norm(inputs=x, center=center, scale=scale,
                                        is_training=phase, scope=name, data_format='NHWC')

def linear(input, output_dim, noise_amplitude, stddev=0.02, bias_start=0.0, name="linear"):
    shape = input.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable("W", [shape[1], output_dim], tf.float32, tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", [output_dim], tf.float32, initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input, w) + b