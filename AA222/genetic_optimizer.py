#!/usr/bin/env python3
import argparse
import numpy as np

def eval_pop(population, eval_fcn, eval_fcn_arg_dict=None):
	''' Generates list of scores using provided evaluation function'''
	scores = []
	for member in population: 
		scores.append(eval_fcn(member, **eval_fcn_arg_dict))
	return scores

def selection(population, scores, n_survivors):
	''' Implements performance weighted selection '''
	weights = scores/np.sum(scores)
	survivors = numpy.random.choice(population, size=n_survivors, replace=True, p=weights)
	return survivors

def gen_population(survivors, n_population, p_crossover=0.5, mutation_std=0.1):
	''' Applies crossover and mutation to generate new population'''
	n_survivors = len(population)
	if n_survivors >= n_population: 
		raise ValueError('Population size must exceed number of survivors!')
	new_pop = survivors 
	for _ in range(n_population - n_survivors):
		if np.random.uniform() >= p_crossover:
			parents = np.random.choice(survivors, size=2)
			child = crossover(*parents)
		else:
			parent = np.random.choice(survivors, size=1)
			child = mutate(parent)
		new_pop.append(child)
	return new_pop

def mutate(parent, noise_std=0.1):
	''' Mutates a parent using zero-mean Gaussian noise to produce a child '''
	child = []
	for layer in parent: 
		n_layer = layer + np.random.normal(loc=0.0, scale=noise_std, size=layer.shape)
		child.append(n_layer)
	return child

def crossover(parent_a, parent_b):
	''' Cross over on two parents to produce child ''' 
	child = []
	for i, layer_a in enumerate(paernt_a):
		if np.random.uniform(0.0, 1.0) >= 0.5:
			child.append(layer_a)
		else:
			child.append(parent_b[i])
	return child 

def main(init_population, eval_fcn, eval_fcn_arg_dict=None,
		n_itrs=500, population_size=100, n_survivors=10, 
		p_crossover=0.5, mutation_std=0.1, noise_decay=0.99):
	''' Runs the genetic algorithm ''' 
	mean_scores = []
	std_scores = []
	max_scores = []
	for i in range(n_itrs):
		population= gen_population(survivors, n_population=population_size,
					p_crossover=p_crossover, mutation_std=mutation_std)
		mutation_std *= noise_decay
		
		scores = eval_pop(population, eval_fcn, eval_fcn_arg_dict)
		mean_scores.append(np.mean(scores))
		std_scores.append(np.std(scores))
		max_scores.append(np.amax(scores))

		survivors = selection(population, scores, n_survivors)
		print('%r iterations of %r complete' % ((i+1), n_itrs))
		print('Current best score: %r' % np.amax(scores))

	elite_scores = eval_pop(survivors, eval_fcn, eval_fcn_arg_dict)
	data = {'elite_pop':survivors,
			'elite_scores':elite_scores,
			'mean_scores':mean_scores,
			'std_scores':std_scores,
			'max_scores':max_scores}
	return data
