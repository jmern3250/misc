"""
CS 228: Probabilistic Graphical Models
Winter 2019 (instructor: Stefano Ermon)
Starter Code for Part A
"""

from utils import *
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


def estimate_params(X, Z):
    """Perform MLE estimation of model 1 parameters.

    Input:
        X: A numpy array of size (N, M, 2), where X[i, j] is the 2-dimensional vector
            representing the voter's properties
        Z: A numpy array of size (N, M), where Z[i, j] = 0 or 1 indicating the party preference

    Output: A dictionary with the following keys/values
        pi: (float), estimate of party proportions
        mu0: size (2,) numpy array econding the estimate of the mean of class 0
        mu1: size (2,) numpy array econding the estimate of the mean of class 1
        sigma0: size (2,2) numpy array econding the estimate of the covariance of class 0
        sigma1: size (2,2) numpy array econding the estimate of the covariance of class 1

    This function will be autograded.

    Note: your algorithm should work for any value of N and M
    """
    pi = np.sum(Z)/Z.size
    idxs0 = np.argwhere(Z==0)
    idxs1 = np.argwhere(Z==1)
    X0 = np.array([X[idx[0], idx[1], :] for idx in idxs0.tolist()])
    X1 = np.array([X[idx[0], idx[1], :] for idx in idxs1.tolist()])
    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)
    sigma0 = np.cov(X0.transpose())
    sigma1 = np.cov(X1.transpose())
    return {'pi': pi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}


def em_update(X, params):
    """ Perform one EM update based on unlabeled data X
    Input:
        X: A numpy array of size (N, M, 2), where X[i, j] is the 2-dimensional vector
            representing the voter's properties
        params: A dictionary, the previous parameters, see the description in estimate_params
    Output: The updated parameter. The output format is identical to estimate_params

    This function will be autograded.

    Note: You will most likely need to use the function estimate_z_prob_given_x
    """
    z_prob = estimate_z_prob_given_x(X, params)
    Z_estimated = np.round(z_prob)
    params = estimate_params(X, Z_estimated)
    return params


def estimate_z_prob_given_x(X, params):
    """ Estimate p(z_{ij}|x_{ij}, theta)
    Input:
        X: Identical to the function em_update
        params: Identical to the function em_update
    Output: A 2D numpy array z_prob with the same size as X.shape[0:2],
            z_prob[i, j] should store the value of p(z_{ij}|x_{ij}, theta)
            Note: it should be a normalized probability

    This function will be autograded.
    """
    numerator = multivariate_normal.pdf(X, mean=params['mu1'], cov=params['sigma1'])*params['pi']
    denominator = multivariate_normal.pdf(X, mean=params['mu1'], cov=params['sigma1'])+multivariate_normal.pdf(X, mean=params['mu0'], cov=params['sigma0'])
    z_prob = numerator/denominator
    return z_prob


def compute_log_likelihood(X, params):
    """ Estimate the log-likelihood of the entire data log p(X|theta)
    Input:
        X: Identical to the function em_update
        params: Identical to the function em_update
    Output A real number representing the log likelihood

    This function will be autograded.

    Note: You will most likely need to use the function estimate_z_prob_given_x
    """
    likelihood = 0.0
    m, n = X.shape[:2]
    # import pdb; pdb.set_trace()
    # print()
    for i in range(m):
        for j in range(n):
            prob = multivariate_normal.pdf(X[i,j], mean=params['mu1'], cov=params['sigma1'])*params['pi']
            prob += multivariate_normal.pdf(X[i,j], mean=params['mu0'], cov=params['sigma0'])*(1-params['pi'])
            likelihood += np.log(prob)
    return likelihood




if __name__ == '__main__':
    #===============================================================================
    # This runs the functions that you have defined to produce the answers to the
    # assignment problems
    #===============================================================================

    # Read data
    X_labeled, Z_labeled = read_labeled_matrix()
    X_unlabeled = read_unlabeled_matrix()

    # pt a.i
    params = estimate_params(X_labeled, Z_labeled)

    colorprint("MLE estimates for PA part a.i:", "teal")
    colorprint("\tpi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
        %(params['pi'], params['mu0'], params['mu1'], params['sigma0'], params['sigma1']), "red")

    # pt a.ii

    # params = estimate_params(X_labeled, Z_labeled)  # Initialize
    likelihoods = []
    while True:
        likelihoods.append(compute_log_likelihood(X_unlabeled, params))
        if len(likelihoods) > 2 and likelihoods[-1] - likelihoods[-2] < 0.01:
            break
        params = em_update(X_unlabeled, params)

    colorprint("MLE estimates for PA part a.ii:", "teal")
    colorprint("\tpi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
        %(params['pi'], params['mu0'], params['mu1'], params['sigma0'], params['sigma1']), "red")

    plt.plot(likelihoods)
    plt.show()
