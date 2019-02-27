###############################################################################
# Finishes PA 3
# author: Ya Le, Billy Jun, Xiaocheng Li
# date: Jan 25, 2018


## Edited by Zhangyuan Wang, 01/2019
###############################################################################

## Utility code for PA3
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import itertools
from factor_graph import *
from factors import *


def loadLDPC(name):
    """
    :param - name: the name of the file containing LDPC matrices

    return values:
    G: generator matrix
    H: parity check matrix

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    """
    A = sio.loadmat(name)
    G = A['G']
    H = A['H']
    return G, H


def loadImage(fname, iname):
    '''
    :param - fname: the file name containing the image
    :param - iname: the name of the image
    (We will provide the code using this function, so you don't need to worry too much about it)

    return: image data in matrix form
    '''
    img = sio.loadmat(fname)
    return img[iname]


def applyChannelNoise(y, epsilon):
    '''
    :param y - codeword with 2N entries
    :param epsilon - the probability that each bit is flipped to its complement

    return corrupt message yTilde
    yTilde_i is obtained by flipping y_i with probability epsilon

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    ###############################################################################
    # TODO: Your code here!

    noise = np.random.choice(2, size=y.size, p=(1.-epsilon, epsilon)).reshape(y.shape)
    yTilde = (y + noise)%2

    ###############################################################################
    assert y.shape == yTilde.shape
    return yTilde


def encodeMessage(x, G):
    '''
    :param - x orginal message
    :param[in] G generator matrix
    :return codeword y=Gx mod 2

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    return np.mod(np.dot(G, x), 2)


def constructFactorGraph(yTilde, H, epsilon):
    '''
    Args
    - yTilde: np.array, shape [2N, 1], observed codeword containing 0's and 1's
    - H: np.array, shape [N, 2N], parity check matrix
    - epsilon: float, probability that each bit is flipped to its complement

    Returns: FactorGraph

    You should consider two kinds of factors:
    - M unary factors
    - N each parity check factors

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    N = H.shape[0]
    M = H.shape[1]
    G = FactorGraph(numVar=M, numFactor=N+M)
    G.var = list(range(M))
    ##############################################################
    # To do: your code starts here
    # Add unary factors
    idx = 0
    for i, var in enumerate(G.var):
        val = np.array([epsilon, 1. - epsilon] if bool(yTilde[i][0])
                        else [1. - epsilon, epsilon])
        factor = Factor(scope=[var], card=[2], val=val, name='Unary' + str(var))
        G.factors.append(factor)
        G.varToFactor[var].append(i)
        G.factorToVar[i].append(var)     
        idx += 1
    # Add parity factors
    # You may find the function itertools.product useful
    # (https://docs.python.org/2/library/itertools.html#itertools.product)
    factors = []
    for i in range(N):
        row = H[i,:]
        scope = np.argwhere(row).squeeze()
        card = [2 for _ in scope]
        val = np.zeros(card)
        for prod in itertools.product([0, 1], repeat=len(scope)):
            val[prod] = 1 if np.sum(prod)%2 == 0 else 0
        factors.append(Factor(scope=scope, card=card, val=val, name='Parity' + str(i)))
        G.factorToVar[idx] = scope
        for var in scope: 
            G.varToFactor[var].append(idx)
        idx += 1
    G.factors += factors
    ##############################################################
    return G


def do_part_a():
    yTilde = np.array([1, 1, 1, 1, 1, 1]).reshape(6, 1)
    print("yTilde.shape", yTilde.shape)
    H = np.array([
        [0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 1]])
    epsilon = 0.05
    G = constructFactorGraph(yTilde, H, epsilon)
    ##############################################################
    # To do: your code starts here
    # Design two invalid codewords ytest1, ytest2 and one valid codewords
    # ytest3. Report their weights respectively.
    ytest1 = np.array([0, 1, 1, 0, 1, 0])
    ytest2 = np.array([0, 1, 0, 1, 1, 0])
    ytest3 = np.array([1, 1, 1, 1, 0, 0])

    ##############################################################
    print(G.evaluateWeight(ytest1),
          G.evaluateWeight(ytest2),
          G.evaluateWeight(ytest3))


def sanity_check_noise():
    '''
    Sanity check applyChannelNoise to make sure bits are flipped at
    a reasonable proportion.
    '''
    N = 256
    epsilon = 0.05
    err_percent = 0
    num_trials = 1000
    x = np.zeros((N, 1), dtype='int32')
    for _ in range(num_trials):
        x_noise = applyChannelNoise(x, epsilon)
        err_percent += np.sum(x_noise)/N
    err_percent /= num_trials
    assert abs(err_percent-epsilon) < 0.005


def do_part_b(fixed=False, npy_file=None):
    '''
    We provide you an all-zero initialization of message x. If fixed=True and
    `npy_file` is not given, you should apply noise on y to get yTilde.
    Otherwise, load in the npy_file to get yTilde. Then do loopy BP to obtain
    the marginal probabilities of the unobserved y_i's.

    Args
    - fixed: bool, False if using random noise, True if loading from given npy file
    - npy_file: str, path to npy file, must be specified when fixed=True
    '''
    G, H = loadLDPC('ldpc36-128.mat')

    print((H.shape))
    epsilon = 0.05
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)
    if not fixed:
        yTilde = applyChannelNoise(y, epsilon)
        print("Applying random noise at eps={}".format(epsilon))
        print(np.sum(yTilde)) #17
    else:
        assert npy_file is not None
        yTilde = np.load(npy_file)
        print("Loading yTilde from {}".format(npy_file))
    ##########################################################################################
    # To do: your code starts here
    graph = constructFactorGraph(yTilde, H, epsilon)
    for var, factors in enumerate(graph.varToFactor):
        for factor in factors:
            graph.getInMessage(var, factor, 'varToFactor')
    for factor, varbs in enumerate(graph.factorToVar):
        if len(varbs) == 1:
            graph.messagesFactorToVar[(factor, varbs[0])]  = graph.factors[factor]
        else:
            for i, var in enumerate(varbs):
                graph.getInMessage(factor, var, 'factorToVar')
                

    graph.runParallelLoopyBP(50)
    marginals = []
    correct = 0
    for i, val in enumerate(y.squeeze()): 
        marginal = graph.estimateMarginalProbability(i)
        correct += np.argmax(marginal) == val
        marginals.append(marginal[1])

    plt.figure()
    plt.plot(range(i+1), marginals, '.k')
    plt.title('Posterior probability bit=1')
    plt.xlabel('Bit Index')
    plt.ylabel('Posterior Probability')
    plt.savefig('./P5bii.png')
    plt.show()
    print('Percent Recovery: ', correct/y.size)
    ##############################################################


def do_part_cd(numTrials, error, iterations=50):
    '''
    param - numTrials: how many trials we repreat the experiments
    param - error: the transmission error probability
    param - iterations: number of Loopy BP iterations we run for each trial
    '''
    G, H = loadLDPC('ldpc36-128.mat')
    ##############################################################
    # To do: your code starts here
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)
    plt.figure()
    for _ in range(numTrials):
        yTilde = applyChannelNoise(y, error)
        graph = constructFactorGraph(yTilde, H, error)
        for var, factors in enumerate(graph.varToFactor):
            for factor in factors:
                graph.getInMessage(var, factor, 'varToFactor')
        for factor, varbs in enumerate(graph.factorToVar):
            if len(varbs) == 1:
                graph.messagesFactorToVar[(factor, varbs[0])]  = graph.factors[factor]
            else:
                for i, var in enumerate(varbs):
                    graph.getInMessage(factor, var, 'factorToVar')
        values = []
        for itr in range(iterations):
            graph.runParallelLoopyBP(1)
            if ((itr+1)%10) == 0:
                print('Iteration %i of %i complete'%(itr+1, iterations))
            marginals = graph.getMarginalMAP()
            distance = np.sum(marginals)
            values.append(distance)

        plt.plot(values)

    plt.title('Hamming Distance over LBP Iterations')
    plt.ylabel('Hamming Distance')
    plt.xlabel('LBP Iteration')
    plt.savefig('./P5cd' + str(int(100*error)) + '.png')
    plt.show()
    ##############################################################


def do_part_ef(error):
    '''
    param - error: the transmission error probability
    '''
    G, H = loadLDPC('ldpc36-1600.mat')
    img = loadImage('images.mat', 'cs242')
    ##############################################################
    # To do: your code starts here
    # You should flattern img first and treat it as the message x in the previous parts.
    img_shape = img.shape
    N = G.shape[1]
    x = img.reshape((N, 1))
    y = encodeMessage(x, G)
    yTilde = applyChannelNoise(y, error)
    graph = constructFactorGraph(yTilde, H, error)
    for var, factors in enumerate(graph.varToFactor):
        for factor in factors:
            graph.getInMessage(var, factor, 'varToFactor')
    for factor, varbs in enumerate(graph.factorToVar):
        if len(varbs) == 1:
            graph.messagesFactorToVar[(factor, varbs[0])]  = graph.factors[factor]
        else:
            for i, var in enumerate(varbs):
                graph.getInMessage(factor, var, 'factorToVar')
    plot_iterations = [0, 1, 2, 3, 5, 10, 20, 30]
    plt.figure()
    for i in range(30):
        if i in plot_iterations:
            marginals = graph.getMarginalMAP()
            img_out = marginals[:N].reshape(img_shape)
            idx = plot_iterations.index(i)
            plt.subplot(2, 4, idx + 1)
            plt.imshow(img_out)
            plt.title("Iteration: " + str(i))
        graph.runParallelLoopyBP(1)
        print('Iteration %i complete'%(i+1))
    marginals = graph.getMarginalMAP()
    img_out = marginals[:N].reshape(img_shape)
    idx = plot_iterations.index(i+1)
    plt.subplot(2, 4, idx + 1)
    plt.imshow(img_out)
    plt.title("Iteration: " + str(i+1))
    plt.savefig('./P5ef' + str(int(100*error)) +'.png')
    plt.show()
    plt.close()


    ################################################################


if __name__ == '__main__':
    print('Doing part (a): Should see 0.0, 0.0, >0.0')
    do_part_a()
    
    print("Doing sanity check applyChannelNoise")
    sanity_check_noise()
    print('Doing part (b) fixed')
    # do_part_b(fixed=True, npy_file='part_b_test_1.npy')    # This should perfectly recover original code
    # do_part_b(fixed=True, npy_file='part_b_test_2.npy')    # This may not recover at perfect probability
    print('Doing part (b) random')
    # do_part_b(fixed=False)
    print('Doing part (c)')
    do_part_cd(10, 0.06)
    print('Doing part (d)')
    do_part_cd(10, 0.08)
    do_part_cd(10, 0.10)
    print('Doing part (e)')
    # do_part_ef(0.06)
    print('Doing part (f)')
    do_part_ef(0.10)
