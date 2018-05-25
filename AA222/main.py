#!/usr/bin/env python3
import argparse
import numpy as np 
import tensorflow as tf 
import gym 
import pickle 
from mpi4py import MPI
from baselines import logger
from mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
import genetic_optimizer
import baselines.common.tf_util as U

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
		self.X = tf.placeholder(tf.float32, [None, 4], name='X')
		h = self.X
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
			self.Y = tf.layers.dense(
						    h,
						    2,
						    activation=None,
						    use_bias=True,
						    kernel_initializer=tf.contrib.layers.xavier_initializer(),
						    bias_initializer=tf.zeros_initializer(),
						    name='Y',
						)
		var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		self.saver = tf.train.Saver(var_list=var_list)

def train(env, num_timesteps, seed):
    
    sess = U.single_threaded_session()
    sess.__enter__()

    logger.configure(dir='./log/', format_strs=['stdout', 'csv', 'tensorboard'])
    # logger.set_level(logger.DISABLED)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=600, max_kl=0.02, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    graph_vars = U.tf.trainable_variables(scope='pi/pol')
    saver = tf.train.Saver(var_list=graph_vars)
    saver.save(sess, './models/sgd_model')
    return graph_vars, sess

def extract_weights(var_list, sess):
	graph_vars = sess.run(var_list, feed_dict={})
	return graph_vars 

def extract_stats(sess):
	run_sum = sess.graph.get_tensor_by_name('pi/obfilter/runningsum:0')
	run_sumsq = sess.graph.get_tensor_by_name('pi/obfilter/runningsumsq:0')
	run_count = sess.graph.get_tensor_by_name('pi/obfilter/count:0')
	ob_sum = sess.run(run_sum, feed_dict={})
	ob_sumsq = sess.run(run_sumsq, feed_dict={})
	ob_count = sess.run(run_count, feed_dict={})	
	mean = ob_sum/ob_count
	std = ob_sumsq/ob_count
	return mean, std 

def build_graph(graph_vars):
	graph = tf.Graph()
	sess = tf.InteractiveSession(graph=graph)
	with graph.as_default():
		mlp = MLP(2, 64, tf.nn.tanh, scope='mlp')
		var_list = tf.trainable_variables()
		update_ops = []
		for i, var in enumerate(var_list):
			update_ops.append(var.assign(graph_vars[i]))	
		sess.run(tf.global_variables_initializer())
		sess.run(update_ops)

	return graph, mlp.X, mlp.Y, update_ops, sess

def evaluation(var_list, env, graph, X, Y, sess, n_episodes=10):
	update_ops = []
	with graph.as_default():
		graph_vars = tf.trainable_variables()
		for i, var in enumerate(graph_vars):
			update_ops.append(var.assign(var_list[i]))
		sess.run(update_ops, feed_dict={})
	np.random.seed(0)
	score = 0.
	for _ in range(n_episodes):
		observation = env.reset()
		ep_score = 0.
		done = False
		while not done: 
			observation = observation.reshape([1, -1])
			with graph.as_default():
				logits = sess.run(Y, 
					feed_dict={X:observation})
			action = np.argmax(logits)
			observation, reward, done, _ = env.step(action)
			ep_score += reward

		score += ep_score
	return score 

def main(args):
	#### Train policy via TRPO ####
	env_id = 'CartPole-v0'
	env = gym.make(env_id)
	# seed = 0
	# graph_vars, sess = train(env, args.num_timesteps, seed)
	# vars_list = extract_weights(graph_vars, sess)
	# with open('./models/SGD/mlp0.p', 'wb') as f: 
	# 	pickle.dump(vars_list, f)
	# sess.close()
	
	with open('./models/SGD/mlp0.p', 'rb') as f: 
		vars_list = pickle.load(f)
	graph, X, Y, _, sess = build_graph(vars_list)

	#### GA Fine Tuning #### 
	eval_dict = {'env':env, 'n_episodes':10, 'graph':graph, 
				 'X':X, 'Y':Y, 'sess':sess}
	results = genetic_optimizer.main([vars_list], evaluation, 
		eval_fcn_arg_dict=eval_dict,
		n_itrs=10, population_size=50, n_survivors=5, 
		p_crossover=0.5, mutation_std=0.1, noise_decay=0.99)
	with open('./models/GA/results0.p', 'wb') as f: 
		pickle.dump(results, f)

	print(score)
	# import pdb; pdb.set_trace()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train and fine tune MLP')
	parser.add_argument('num_timesteps', type=int)
	args = parser.parse_args()
	main(args)