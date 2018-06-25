#!/usr/bin/env python3
import argparse
import numpy as np 
import tensorflow as tf 
import gym 
import pickle 
from mountain_car_stochastic import MountainCarEnvStochastic as MCS
from mpi4py import MPI
from baselines import logger
from mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
import genetic_optimizer
import baselines.common.tf_util as U
import matplotlib.pyplot as plt 

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
		self.X = tf.placeholder(tf.float32, [None, 2], name='X')
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
							3,
							activation=None,
							use_bias=True,
							kernel_initializer=tf.contrib.layers.xavier_initializer(),
							bias_initializer=tf.zeros_initializer(),
							name='Y',
						)
		var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		self.saver = tf.train.Saver(var_list=var_list)

def tf_discount_rewards(tf_r): 
	gamma = 0.99
	discount_f = lambda a, v: a*gamma + v;
	tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
	tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
	return tf_discounted_r

def VPG(env, graph, sess, X, Y, n_steps):
	P = tf.nn.softmax(Y, name='P')
	A = tf.placeholder(dtype=tf.float32, shape=[None,3], name='A')
	EPR = tf.placeholder(dtype=tf.float32, shape=[None,1], name="EPR")

	dEPR = tf_discount_rewards(EPR)
	EPRmean, EPRvar= tf.nn.moments(dEPR, [0], shift=None, name="reward_moments")
	dEPR -= EPRmean
	dEPR /= tf.sqrt(EPRvar + 1e-6)

	loss = tf.nn.l2_loss(A-P) 
	with tf.variable_scope('train') as tr:
		optimizer = tf.train.RMSPropOptimizer(1e-4, decay=0.99)
		tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=dEPR)
		train_op = optimizer.apply_gradients(tf_grads)
	train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train')
	sess.run(tf.variables_initializer(train_vars,name='init'))
	n = 0
	i = 0
	while n < n_steps:
		idx = i%10
		env.seed(idx)
		observation = env.reset()
		done = False
		observations = []
		actions = []
		rewards = []
		while not done:
			observation = observation.reshape([1, -1])
			p = sess.run(P, 
				feed_dict={X:observation})
			action = np.random.choice([0,1, 2], p=p[0])
			observation, reward, done, _ = env.step(action)
			observations.append(observation)
			actions.append(action)
			rewards.append(reward)
			n +=1
			if done:
				print('Episode %r done with score: %r' % (i, np.sum(rewards)))
				m = len(actions)
				acts = np.zeros([m,3])
				acts[range(m),actions] = 1
				rewards = np.array(rewards).reshape([-1,1])
				feed_dict = {X:observations,
							 A:acts,
							 EPR:rewards}
				sess.run(train_op, feed_dict=feed_dict)
				observations = []
				actions = []
				rewards = []
				i += 1

def train(env, num_timesteps, seed):
	
	sess = U.single_threaded_session()
	sess.__enter__()

	logger.configure(dir='./log/', format_strs=['stdout', 'csv', 'tensorboard'])
	# logger.set_level(logger.DISABLED)
	def policy_fn(name, ob_space, ac_space):
		return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
			hid_size=64, num_hid_layers=2)
	trpo_mpi.learn(env, policy_fn, timesteps_per_batch=10000, max_kl=0.02, cg_iters=10, cg_damping=0.1,
		max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
	graph_vars = U.tf.trainable_variables(scope='pi/pol')
	saver = tf.train.Saver(var_list=graph_vars)
	saver.save(sess, './models/mountaincar/sgd_model')
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
	for i in range(n_episodes):
		env.seed(i)
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
	# ## Train policy via TRPO ####
	# env = MCS()
	# seed = 0
	# graph_vars, sess = train(env, args.num_timesteps, seed)
	# vars_list = extract_weights(graph_vars, sess)
	# with open('./models/mountaincar/SGD/mlp0.p', 'wb') as f: 
	# 	pickle.dump(vars_list, f)
	# sess.close()

	# #### GA Fine Tuning #### 	
	# with open('./models/mountaincar/SGD/mlp0.p', 'rb') as f: 
	# 	vars_list = pickle.load(f)

	# env = MCS()
	# graph, X, Y, _, sess = build_graph(vars_list)

	# eval_dict = {'env':env, 'n_episodes':10, 'graph':graph, 
	# 			 'X':X, 'Y':Y, 'sess':sess}
	# results = genetic_optimizer.main([vars_list], evaluation, 
	# 	eval_fcn_arg_dict=eval_dict,
	# 	n_itrs=10, population_size=30, n_survivors=15, 
	# 	p_crossover=0.25, mutation_std=0.1, noise_decay=0.9)
	# with open('./models/mountaincar/GA/results0.p', 'wb') as f: 
	# 	pickle.dump(results, f)

	# #### SGD Fine Tuning ####
	# with open('./models/mountaincar/SGD/mlp0.p', 'rb') as f: 
	# 	vars_list = pickle.load(f)
	# graph, X, Y, _, sess = build_graph(vars_list)

	# env = MCS()
	# VPG(env, graph, sess, X, Y, 10000)

	# vars_list = extract_weights(tf.trainable_variables(), sess)
	# with open('./models/mountaincar/SGD/mlp1.p', 'wb') as f: 
	# 	pickle.dump(vars_list, f)

	# #### Eval GA ####
	# with open('./models/mountaincar/GA/results0.p', 'rb') as f: 
	# 	results = pickle.load(f)

	# elite_idx = np.argmax(results['elite_scores'])
	# elite_vars = results['elite_pop'][elite_idx]
	# graph, X, Y, _, sess = build_graph(elite_vars)

	# env = MCS()
	# scores = []
	# for i in range(500):
	# 	env.seed(i)
	# 	observation = env.reset()
	# 	ep_score = 0.
	# 	done = False
	# 	while not done: 
	# 		observation = observation.reshape([1, -1])
	# 		with graph.as_default():
	# 			logits = sess.run(Y, 
	# 				feed_dict={X:observation})
	# 		action = np.argmax(logits)
	# 		observation, reward, done, _ = env.step(action)
	# 		ep_score += reward
	# 	scores.append(ep_score)

	# print('Mean score: %r' % np.mean(scores))
	# print('STD score: %r' % np.std(scores))
	# print('SE score: %r' % (np.std(scores)/np.sqrt(500)))
	
	#### Eval SGD ####
	with open('./models/mountaincar/SGD/mlp0.p', 'rb') as f: 
		vars_list = pickle.load(f)
	graph, X, Y, _, sess = build_graph(vars_list)

	env = MCS()
	scores = []
	for i in range(500):
		env.seed(i)
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
			if not done: 
				done = ep_score >= 200.
		scores.append(ep_score)
	print('Mean score: %r' % np.mean(scores))
	print('STD score: %r' % np.std(scores))
	print('SE score: %r' % (np.std(scores)/np.sqrt(500)))




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train and fine tune MLP')
	parser.add_argument('num_timesteps', type=int)
	args = parser.parse_args()
	main(args)