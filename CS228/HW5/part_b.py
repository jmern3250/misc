"""
CS 228: Probabilistic Graphical Models
Winter 2019 (instructor: Stefano Ermon)
Starter Code for Part B
Author: Shengjia Zhao (sjzhao@stanford.edu)
"""

from utils import *
import numpy as np
from scipy.stats import multivariate_normal
import math


def estimate_phi_lambda(Z):
    """Perform MLE estimation of phi and lambda as described in B(i)
    Assumes that Y variables have been estimated using heuristic proposed in the question.
    Input:
        Z: A numpy array of size (N, M), where Z[i, j] = 0 or 1 indicating the party preference
    Output:
        MLE_phi: a real number, estimate of phi
        MLE_lambda: a real number, estimate of lambda

    This function will be autograded.
    """
    m, n = Z.shape
    Y = np.round(np.sum(Z,axis=1,keepdims=True)/n)
    MLE_phi = np.sum(Y)/m
    MLE_lambda = np.sum(Y == Z)/(m*n)
    return {'phi': MLE_phi, 'lambda': MLE_lambda}

def p_x_z(x, z, params):
    ''' log P(x|z)'''
    if z == 0: 
        return multivariate_normal.logpdf(x, mean=params['mu0'], cov=params['sigma0'])
    elif z == 1:
        return multivariate_normal.logpdf(x, mean=params['mu1'], cov=params['sigma1'])

def p_x_y(x, y, params):
    ''' log P(x|y) '''
    p_x_0 = p_x_z(x, 0, params) + p_z_y(0, y, params)
    p_x_1 = p_x_z(x, 1, params) + p_z_y(1, y, params)
    return log_sum_exp(p_x_0, p_x_1) 

def p_z_y(z, y, params):
    ''' log P(z|y) '''
    if z == y: 
        return np.log(params['lambda'])
    else:
        return np.log(1. - params['lambda'])

def p_y(y, params):
    if y == 1.:
        return np.log(params['phi'])
    else:
        return np.log(1. - params['phi'])

def p_z(z, params):
    p_z_0 = p_z_y(z, 0, params) + p_y(0, params)
    p_z_1 = p_z_y(z, 1, params) + p_y(1, params)
    return log_sum_exp(p_z_0, p_z_1)

def p_x(x, params):
    ''' log P(x) '''
    p_x_0 = p_x_z(x, 0, params) + p_z(0, params) 
    p_x_1 = p_x_z(x, 1, params) + p_z(1, params) 
    return log_sum_exp(p_x_0, p_x_1)

def z_y_fun(z, y, X, i, j, params):
    _, n, _ = X.shape
    value = 0. 
    for k in range(n):
        if k != j:
            x = X[i,k]
            z_0 = p_x_z(x, 0, params) + p_z_y(0, y, params)
            z_1 = p_x_z(x, 1, params) + p_z_y(1, y, params)
            value += log_sum_exp(z_0, z_1)
    value += p_y(y, params) + p_x_z(X[i,j], z, params) + p_z_y(z, y, params)
    return value

def compute_yz_marginal(X, params):
    """Evaluate log p(y_i=1|X) and log p(z_{ij}=1|X)

    Input:
        X: A numpy array of size (N, M, 2), where X[i, j] is the 2-dimensional vector
            representing the voter's properties
        params: A dictionary with the current parameters theta, elements include:
            phi: (float), as stated in the question
            lambda: (float), as stated in the question
            mu0: size (2,) numpy array econding the estimate of the mean of class 0
            mu1: size (2,) numpy array econding the estimate of the mean of class 1
            sigma0: size (2,2) numpy array econding the estimate of the covariance of class 0
            sigma1: size (2,2) numpy array econding the estimate of the covariance of class 1
    Output:
        y_prob: An numpy array of size X.shape[0:1]; y_prob[i] store the value of log p(y_i=1|X, theta)
        z_prob: An numpy array of size X.shape[0:2]; z_prob[i, j] store the value of log p(z_{ij}=1|X, theta)

    You should use the log-sum-exp trick to avoid numerical overflow issues (Helper functions in utils.py)
    This function will be autograded.
    """
    m, n, c = X.shape

    y_prob = np.zeros([m,])
    for i in range(m):
        y_1 = 0.
        y_0 = 0.
        for j in range(n):
            x = X[i,j]
            y_1 += p_x_y(x, 1, params)
            y_0 += p_x_y(x, 0, params)
        y_1 += p_y(1, params)
        y_0 += p_y(0, params)
        y_prob[i] =  y_1 - log_sum_exp(y_1, y_0)

    z_prob = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            x = X[i,j]
            z_1_y_1 = z_y_fun(1, 1, X, i, j, params)
            z_1_y_0 = z_y_fun(1, 0, X, i, j, params)
            z_0_y_1 = z_y_fun(0, 1, X, i, j, params)
            z_0_y_0 = z_y_fun(0, 0, X, i, j, params)
            z_1 = log_sum_exp(z_1_y_1, z_1_y_0)
            z_0 = log_sum_exp(z_0_y_1, z_0_y_0)
            z_prob[i,j] = z_1 - log_sum_exp(z_0, z_1) 
       
    return y_prob, z_prob


def compute_yz_joint(X, params):
    """ Compute the joint probability of log p(y_i, z_{ij}|X, params)
    Input:
        X: As usual
        params: A dictionary containing the old parameters, refer to compute compute_yz_marginal
    Output:
        yz_prob: A array of shape (X.shape[0], X.shape[1], 2, 2);
            yz_prob[i, j, u, v] should store the value of log p(y_i=u, z_{ij}=v|X, params)
            Don't forget to normalize your (conditional) probability

    Note: To avoid numerical overflow, you should use log_sum_exp trick (Helper functions in utils.py)

    This function will be autograded.
    """
    m, n, _ = X.shape
    yz_prob = np.zeros((m,n,2,2))
    for i in range(m):
        for j in range(n):
            # i, j, y, z
            x = X[i, j]
            yz_prob[i,j,0,0] += p_x_z(x, 0, params) + p_z_y(0, 0, params) + p_y(0, params)
            yz_prob[i,j,1,0] += p_x_z(x, 0, params) + p_z_y(0, 1, params) + p_y(1, params)
            yz_prob[i,j,0,1] += p_x_z(x, 1, params) + p_z_y(1, 0, params) + p_y(0, params)
            yz_prob[i,j,1,1] += p_x_z(x, 1, params) + p_z_y(1, 1, params) + p_y(1, params)
            for k in range(n):
                if k != j:
                    x_ = X[i,k]
                    yz_prob[i,j,0,0] += p_x_y(x_, 0, params)
                    yz_prob[i,j,1,0] += p_x_y(x_, 1, params)
                    yz_prob[i,j,0,1] += p_x_y(x_, 0, params)
                    yz_prob[i,j,1,1] += p_x_y(x_, 1, params)
    eta = log_sum_exp_np(yz_prob, axis=2)
    eta = log_sum_exp_np(eta, axis=3)
    yz_prob -= eta
    # test = log_sum_exp_np(yz_prob, axis=2)
    # test = log_sum_exp_np(test, axis=3)
    # import pdb; pdb.set_trace()
    # print()
    return yz_prob


def em_step(X, params):
    """ Make one EM update according to question B(iii)
    Input:
        X: As usual
        params: A dictionary containing the old parameters, refer to compute compute_yz_marginal
    Output:
        new_params: A dictionary containing the new parameters

    This function will be autograded.
    """
    m, n, _ = X.shape
    yz_prob = np.exp(compute_yz_joint(X, params))
    y_prob, z_prob = compute_yz_marginal(X, params)
    phi = np.sum(np.exp(y_prob))/m
    print('phi')
    lam = np.sum(yz_prob[...,0,0]) + np.sum(yz_prob[...,1,1])
    lam /= (m*n)
    print('lambda')
    ##########
    z_prob = np.exp(np.expand_dims(z_prob, axis=2))
    X0 = X*(1. - z_prob)
    X1 = X*z_prob
    X_list = []
    X0_list = []
    X1_list = []
    z_list = []
    for i in range(m):
        for j in range(n):
            X_list.append(X[i,j,:])
            X0_list.append(X0[i,j,:])
            X1_list.append(X1[i,j,:])
            z_list.append(z_prob[i,j,0])
    X = np.array(X_list)    
    X0 = np.array(X0_list)
    X1 = np.array(X1_list)
    pi = np.mean(z_prob)
    mu0 = np.sum(X0, axis=0)/np.sum(1. - z_prob)
    mu1 = np.sum(X1, axis=0)/np.sum(z_prob)
    print('mu')
    sigma0 = 0.
    sigma1 = 0.
    total0 = 0.
    total1 = 0.
    for i in range(len(z_list)):
        d0 = X[i,:] - mu0
        sigma0 += (1. - z_list[i])*np.outer(d0,d0)
        d1 = X[i,:] - mu1
        sigma1 += z_list[i]*np.outer(d1,d1)
        total0 += 1 - z_list[i]
        total1 += z_list[i] 
    sigma0 /= total0
    sigma1 /= total1
    print('sigma')
    new_params = {'phi':phi, 'lambda':lam, 'mu0':mu0,
                  'mu1':mu1, 'sigma0':sigma0, 'sigma1':sigma1}
    return new_params


def compute_log_likelihood(X, params):
    """ Compute the log likelihood log p(X) under current parameters.
    To compute this you can first call the function compute_yz_joint

    Input:
        X: As usual
        params: As in the description for compute_yz_joint
    Output: A real number representing log p(X)

    This function will be autograded
    """
    m, n, _ = X.shape
    likelihood = 0.
    for i in range(m):
        p_y_0 = p_y(0, params)
        p_y_1 = p_y(1, params)
        for j in range(n):
            x = X[i,j]
            p_y_0 += log_sum_exp(p_x_z(x,0,params) + p_z_y(0,0,params), p_x_z(x,1,params) + p_z_y(1,0,params))
            p_y_1 += log_sum_exp(p_x_z(x,0,params) + p_z_y(0,1,params), p_x_z(x,1,params) + p_z_y(1,1,params))
        likelihood += log_sum_exp(p_y_0, p_y_1)

    return likelihood


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # Read data
    X_labeled, Z_labeled = read_labeled_matrix()
    X_unlabeled = read_unlabeled_matrix()

    # # Question B(i)
    # from part_a import estimate_params
    # params = estimate_params(X_labeled, Z_labeled)
    # params.update(estimate_phi_lambda(Z_labeled))

    # colorprint("MLE estimates for PA part b.i:", "teal")
    # colorprint("\tMLE phi: %s\n\tMLE lambda: %s\n"%(params['phi'], params['lambda']), 'red')

    # # Question B(ii)
    # y_prob, z_prob = compute_yz_marginal(X_unlabeled, params)   # Get the log probability of y and z conditioned on x
    # colorprint("Your predicted party preference:", "teal")
    # colorprint(str((y_prob > np.log(0.5)).astype(np.int)), 'red')

    # plt.scatter(X_unlabeled[:, :, 0].flatten(), X_unlabeled[:, :, 1].flatten(),
    #             c=np.array(['red', 'blue'])[(z_prob > np.log(0.5)).astype(np.int).flatten()], marker='+')
    # plt.show()

    # Question B(iii)
    params = get_random_params()
    likelihoods = []
    for i in range(10):
        likelihoods.append(compute_log_likelihood(X_unlabeled, params))
        params = em_step(X_unlabeled, params)
    colorprint("MLE estimates for PA part b.iv:", "teal")
    colorprint("\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s\n\tphi: %s\n\tlambda: %s\n"
               % (params['mu0'], params['mu1'], params['sigma0'], params['sigma1'], params['phi'], params['lambda']), "red")
    plt.plot(likelihoods)
    plt.show()

    # Question B(iv)
    y_prob, z_prob = compute_yz_marginal(X_unlabeled, params)
    colorprint("Your predicted party preference:", "teal")
    colorprint(str((y_prob > np.log(0.5)).astype(np.int)), 'red')
    plt.scatter(X_unlabeled[:, :, 0].flatten(), X_unlabeled[:, :, 1].flatten(),
                c=np.array(['red', 'blue'])[(z_prob > np.log(0.5)).astype(np.int).flatten()], marker='+')
    plt.show()

