import tensorflow as tf
import numpy as np
import pdb
from ops import *


class Signal():
    def __init__(self, amplitude, frequency, noise_weight):
        self.A = amplitude
        self.w = frequency
        self.C = noise_weight

    def sample(self, t):
        return np.dot(self.A.T , np.sin(self.w * t)) + self.C * (np.random.ranf() - 0.5)

    def digital(self, t, resolution, max_v):
        true_sig = np.dot(self.A.T , np.sin(self.w * t))
        mod_sig = true_sig * 2**resolution / max_v
        raw_bits = np.unpackbits(np.array(mod_sig, dtype=np.uint8))
        bit_vector = np.zeros((1,resolution))
        offset = resolution - raw_bits.shape[0]
        bit_vector[0, offset:offset+raw_bits.shape[0]] = raw_bits
        return bit_vector


class ADC():
    def __init__(self, layer_depths = [64,32],
                 resolution = 4,
                 learning_rate = 1e-3):
        self.layer_depths = layer_depths
        self.resolution = resolution
        self.learning_rate = learning_rate
        #TODO Add in all model attributes


    def build_model(self):
        with tf.variable_scope("adc"):
            self.signal_input = tf.placeholder(tf.float32, [1,1])
            self.signal_labels = tf.placeholder(tf.float32, [1,self.resolution])

            current_input = self.signal_input
            for i, output_depth in enumerate(self.layer_depths + [self.resolution]):
                self.result = tf.nn.relu(linear(current_input, output_dim=output_depth, noise_amplitude=0.0, name="linear_" + str(i)),
                                    name = "relu_" + str(i))
                current_input = self.result

            self.output = tf.sigmoid(self.result,name="sig")
            self.output_layer = self.result
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.signal_labels)
            )

        #summaries
        self.loss_sum = tf.summary.scalar("loss", self.loss)
        self.input_sum = tf.summary.scalar("input", self.signal_input[0][0])
        self.labels_sum = tf.summary.histogram("labels", self.signal_labels)
        self.logits_sum = tf.summary.histogram("logits", self.output)


    def train(self):

        optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


        sig_gen = Signal(amplitude=np.asarray([1.0]),
                         frequency=np.asarray([100]),
                         noise_weight=0.0)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            self.sum = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("./summaries/run1", sess.graph)

            for iter in range(10000):
                e_loss, sum_str = sess.run([self.loss, self.sum], feed_dict={
                    self.signal_input: np.asarray([[sig_gen.sample(iter*0.1)]]),
                    self.signal_labels: sig_gen.digital(iter*0.1, self.resolution, 3.0)
                })
                self.writer.add_summary(sum_str, iter)

                if iter % 1000 == 0:
                    print("Loss: ", e_loss)


