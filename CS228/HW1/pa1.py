"""
CS 228: Probabilistic Graphical Models
Winter 2018
Programming Assignment 1: Bayesian Networks

Author: Aditya Grover
"""

import numpy as np 
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.io import loadmat
from scipy.special import logsumexp

def plot_histogram(data, title='histogram', xlabel='value', ylabel='frequency', savefile='hist'):
	'''
	Plots a histogram.
	'''

	plt.figure()
	plt.hist(data)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig(savefile, bbox_inches='tight')
	plt.show()
	plt.close()

	return

def get_p_z1(z1_val):
	'''
	Helper. Computes the prior probability for variable z1 to take value z1_val.
	P(Z1=z1_val)
	'''

	return bayes_net['prior_z1'][z1_val]

def get_p_z2(z2_val):
	'''
	Helper. Computes the prior probability for variable z2 to take value z2_val.
	P(Z2=z2_val)
	'''

	return bayes_net['prior_z2'][z2_val]

def get_p_xk_cond_z1_z2(z1_val, z2_val, k):
	'''
	Note: k ranges from 1 to 784.
	Helper. Computes the conditional probability that variable xk assumes value 1 
	given that z1 assumes value z1_val and z2 assumes value z2_val
	P(Xk = 1 | Z1=z1_val , Z2=z2_val)
	'''

	return bayes_net['cond_likelihood'][(z1_val, z2_val)][0, k-1]

def get_p_x_cond_z1_z2(z1_val, z2_val):
	'''
	Computes the conditional probability of the entire vector x,
	given that z1 assumes value z1_val and z2 assumes value z2_val
	TODO
	'''
	return bayes_net['cond_likelihood'][(z1_val, z2_val)]

def get_pixels_sampled_from_p_x_joint_z1_z2():
	'''
	This function should sample from the joint probability distribution specified by the model, 
	and return the sampled values of all the pixel variables (x). 
	Note that this function should return the sampled values of ONLY the pixel variables (x),
	discarding the z part.
	TODO. 
	'''
	z1_vals = [key for key in bayes_net['prior_z1'].keys()]
	z1_probs = [p for p in bayes_net['prior_z1'].values()]
	z1 = np.random.choice(z1_vals, size=5, replace=True, p=z1_probs)

	z2_vals = [key for key in bayes_net['prior_z2'].keys()]
	z2_probs = [p for p in bayes_net['prior_z2'].values()]
	z2 = np.random.choice(z2_vals, size=5, replace=True, p=z2_probs)
	z_samples = [s for s in zip(z1, z2)]
	dists = list(map(lambda x: bayes_net['cond_likelihood'][x], z_samples))
	x_samples = list(map(lambda x: np.random.binomial(1, x), dists))
	return x_samples

def p_x_given_z(x, z_vals):
	dists = bayes_net['cond_likelihood'][z_vals]
	log_p_z1 = np.log(bayes_net['prior_z1'][z_vals[0]])
	log_p_z2 = np.log(bayes_net['prior_z2'][z_vals[1]])
	probs = np.multiply(dists, x) + np.multiply((1. - np.array(dists)), (1 - np.array(x)))
	log_prob = np.sum(np.log(probs)) + log_p_z1 + log_p_z2
	return log_prob

def log_p_x(x):
	probs = list(map(lambda z_vals: p_x_given_z(x, z_vals), z_vals_list))
	return logsumexp(probs)

def log_p_x_list(x):
	probs = list(map(lambda z_vals: p_x_given_z(x, z_vals), z_vals_list))
	return probs

def get_conditional_expectation(data):
	'''
	TODO
	'''
	import itertools

	global z_vals_list 
	z_vals_list = list(itertools.product(disc_z1, disc_z2))
	z1_vals_list = np.array([z[0] for z in z_vals_list])
	z2_vals_list = np.array([z[1] for z in z_vals_list])
	z_vals_probs = np.array([bayes_net['prior_z1'][z[0]]*bayes_net['prior_z2'][z[1]] for z in z_vals_list])
	z_log_probs = np.log(z_vals_probs)
	import multiprocessing
	pool = multiprocessing.Pool()
	x_log_probs = np.array(pool.map(log_p_x_list, data.tolist()))
	pool.close()
	x_marginal_probs = logsumexp(x_log_probs, axis=1)
	probs = np.exp(x_log_probs + z_log_probs) #- np.expand_dims(x_marginal_probs, axis=1)
	etas = 1./np.sum(probs, axis=1, keepdims=True)
	probs *= etas
	e_z1 = np.sum(probs*z1_vals_list, axis=1)
	e_z2 = np.sum(probs*z2_vals_list, axis=1)
	return e_z1, e_z2

def q4():
	'''
	Plots the pixel variables sampled from the joint distribution as 28 x 28 images.
	Your job is to implement get_pixels_sampled_from_p_x_joint_z1_z2
	'''

	plt.figure()
	images = get_pixels_sampled_from_p_x_joint_z1_z2()
	for i in range(5):
	    plt.subplot(1, 5, i+1)
	    plt.imshow(images[i].reshape(28, 28), cmap='gray')
	    plt.title('Sample: ' + str(i+1))
	plt.tight_layout()
	plt.savefig('a4', bbox_inches='tight')
	plt.show()
	plt.close()

	return

def q5():
	'''
	Plots the expected images for each latent configuration on a 2D grid.
	Your job is to implement get_p_x_cond_z1_z2
	'''

	canvas = np.empty((28*len(disc_z1), 28*len(disc_z2)))
	for i, z1_val in enumerate(disc_z1):
	    for j, z2_val in enumerate(disc_z2):
	        canvas[(len(disc_z1)-i-1)*28:(len(disc_z2)-i)*28, j*28:(j+1)*28] = \
	        get_p_x_cond_z1_z2(z1_val, z2_val).reshape(28, 28)

	plt.figure()        
	plt.imshow(canvas, cmap='gray')
	plt.tight_layout()
	plt.savefig('a5', bbox_inches='tight')
	plt.show()

	plt.close()

	return

def q6():
	'''
	Loads the data and plots the histograms. Rest is TODO.
	'''

	mat = loadmat('q6.mat')
	val_data = mat['val_x']#[:1000]
	test_data = mat['test_x']#[:1000]


	'''
	TODO
	'''
	import itertools

	global z_vals_list 
	z_vals_list = list(itertools.product(disc_z1, disc_z2))
	
	import multiprocessing
	pool = multiprocessing.Pool()
	val_log_probs = pool.map(log_p_x, val_data.tolist())
	test_log_probs = pool.map(log_p_x, test_data.tolist())
	pool.close()

	mean_log_prob = np.mean(val_log_probs)
	std_log_prob = np.std(val_log_probs)

	z_score = np.abs(test_log_probs - mean_log_prob)/std_log_prob
	corrupt_idxs = np.argwhere(z_score >= 3).flatten()
	true_idxs = np.argwhere(z_score <= 3).flatten()

	real_marginal_log_likelihood = [test_log_probs[i] for i in true_idxs]
	corrupt_marginal_log_likelihood = [test_log_probs[i] for i in corrupt_idxs]
	plot_histogram(real_marginal_log_likelihood, title='Histogram of marginal log-likelihood for real data',
			 xlabel='marginal log-likelihood', savefile='a6_hist_real')

	plot_histogram(corrupt_marginal_log_likelihood, title='Histogram of marginal log-likelihood for corrupted data',
		xlabel='marginal log-likelihood', savefile='a6_hist_corrupt')

	return

def q7():
	'''
	Loads the data and plots a color coded clustering of the conditional expectations. Rest is TODO.
	'''

	mat = loadmat('q7.mat')
	data = mat['x']#[:100]
	labels = mat['y']#[:100]

	mean_z1, mean_z2 = get_conditional_expectation(data)

	# import pdb; pdb.set_trace()
	plt.figure() 
	plt.scatter(mean_z1, mean_z2, c=labels)
	plt.colorbar()
	plt.grid()
	plt.savefig('a7', bbox_inches='tight')
	plt.show()
	plt.close()

	return

def load_model(model_file):
	'''
	Loads a default Bayesian network with latent variables (in this case, a variational autoencoder)
	'''

	with open('trained_mnist_model', 'rb') as infile:
		cpts = pkl.load(infile,encoding='latin1')

	model = {}
	model['prior_z1'] = cpts[0]
	model['prior_z2'] = cpts[1]
	model['cond_likelihood'] = cpts[2]

	return model

def main():

	global disc_z1, disc_z2
	n_disc_z = 25
	disc_z1 = np.linspace(-3, 3, n_disc_z)
	disc_z2 = np.linspace(-3, 3, n_disc_z)

	global bayes_net
	bayes_net = load_model('trained_mnist_model')

	'''
	TODO: Using the above Bayesian Network model, complete the following parts.
	'''
	q4()
	q5()
	q6()
	q7()

	return

if __name__== '__main__':

	main()
