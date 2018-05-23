#!/usr/bin/env python3
import argparse
import numpy as np 
import tensorflow as tf 
import gym 
# from mpi4py import MPI
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi

class MLP(object):
	def __init__(self, n_layers, layer_sizes, activations, scope='mlp'):
		self.scope = scope
		self.n_layers = n_layers
		if not isinstance(layer_sizes, list):
			self.layer_sizes = [layer_sizes]*self.n_layers
		else:
			self.layer_sizes = layer_sizes
		if not isinstance(activations, list):
			self.activations = [activations]*self.n_layers
		self.build_graph()
		self.initialized = False 
		self.session = tf.Session()

	def build_graph(self):
		self.X = tf.placeholder(tf.float32, [None, 4])
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
			for i in range(self.n_layers):
				name = 'h' + str(i)
				h = tf.layers.dense(
						    h,
						    self.layer_sizes[i],
						    activation=self.activations[i],
						    use_bias=True,
						    kernel_initializer=tf.contrib.layers.xavier_initializer(),
						    bias_initializer=tf.zeros_initializer(),
						    name=name,
						)
			self.Y_ = tf.layers.dense(
						    h,
						    2,
						    activation=None,
						    use_bias=True,
						    kernel_initializer=tf.contrib.layers.xavier_initializer(),
						    bias_initializer=tf.zeros_initializer(),
						    name='Y_',
						)
		var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		self.saver = tf.train.Saver(var_list=var_list)

def train(env_id, num_timesteps, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    # rank = MPI.COMM_WORLD.Get_rank()
    # if rank == 0:
    #     logger.configure()
    # else:
    logger.configure(dir='./log/', format_strs=['stdout', 'csv', 'tensorboard'])
    # logger.set_level(logger.DISABLED)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = gym.make(env_id)
    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=600, max_kl=0.02, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    graph_vars = U.tf.trainable_variables(scope='pi/pol')
    saver = tf.train.Saver(var_list=graph_vars)
    saver.save(sess, './models/sgd_model')
    env.close()
    return graph_vars, sess

def extract_weights(var_list, sess):
	graph = sess.run(var_list, feed_dict={})
	return graph 


def main(args):
	env_id = 'CartPole-v0'
	seed = 0
	var_list, sess = train(env_id, args.num_timesteps, seed)
	graph = extract_weights(var_list, sess)
	import pdb; pdb.set_trace()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train and fine tune MLP')
	parser.add_argument('num_timesteps', type=int)
	args = parser.parse_args()
	main(args)