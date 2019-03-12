"""
CS 228: Probabilistic Graphical Models
Winter 2019 (instructor: Stefano Ermon)
Starter Code for Part B
Author: Shengjia Zhao (sjzhao@stanford.edu)
"""

from utils import *
import numpy as np
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
    MLE_phi = 0.0
    MLE_lambda = 0.0
    return {'phi': MLE_phi, 'lambda': MLE_lambda}


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
    y_prob = 0.0
    z_prob = 0.0
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
    yz_prob = 0.0
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
    new_params = {}
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

    likelihood = 0.0
    return likelihood


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # Read data
    X_labeled, Z_labeled = read_labeled_matrix()
    X_unlabeled = read_unlabeled_matrix()

    # Question B(i)
    from part_a import estimate_params
    params = estimate_params(X_labeled, Z_labeled)
    params.update(estimate_phi_lambda(Z_labeled))

    colorprint("MLE estimates for PA part b.i:", "teal")
    colorprint("\tMLE phi: %s\n\tMLE lambda: %s\n"%(params['phi'], params['lambda']), 'red')

    # Question B(ii)
    params = get_random_params()
    y_prob, z_prob = compute_yz_marginal(X_unlabeled, params)   # Get the log probability of y and z conditioned on x
    colorprint("Your predicted party preference:", "teal")
    colorprint(str((y_prob > np.log(0.5)).astype(np.int)), 'red')

    plt.scatter(X_unlabeled[:, :, 0].flatten(), X_unlabeled[:, :, 1].flatten(),
                c=np.array(['red', 'blue'])[(z_prob > np.log(0.5)).astype(np.int).flatten()], marker='+')
    plt.show()

    # Question B(iii)
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

