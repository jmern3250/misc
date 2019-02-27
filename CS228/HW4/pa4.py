# Gibbs sampling algorithm to denoise an image
# Author : Gunaa AV, Isaac Caswell
# Edits : Bo Wang, Kratarth Goel, Aditya Grover, Stephanie Wang
# Date : 2/10/2019

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import copy 


def markov_blanket(i, j, Y, X):
    '''Gets the values of the Markov blanket of Y_ij.

    Args
    - i: int, row index of Y
    - j: int, col index of Y
    - Y: np.array, shape [H, W], note that read_txt_file() pads the image with
            0's to help you take care of corner cases
    - X: np.array, shape [H, W]

    Returns
    - blanket: list, values of the Markov blanket of Y_ij

    Example: if i = j = 1, the function should return
        [Y[0,1], Y[1,0], Y[1,2], Y[2,1], X[1,1]]

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    blanket = []
    ########
    blanket = [Y[i-1,j],Y[i,j-1],Y[i,j+1],Y[i+1,j],X[i,j]]
    ########
    return blanket


def sampling_prob(markov_blanket):
    '''Computes P(X=1 | MarkovBlanket(X)).

    Args
    - markov_blanket: list, values of a variable's Markov blanket, e.g. [1,1,-1,1,-1]
        Because beta = eta in this case, the order doesn't matter. See part (a)

    Returns
    - prob: float, the probability of the variable being 1 given its Markov blanket

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    ########
    numerator = np.exp(np.sum(markov_blanket))
    denominator = np.exp(np.sum(markov_blanket)) + np.exp(-np.sum(markov_blanket))
    prob = numerator/denominator
    ########
    return prob


def sample(i, j, Y, X, dumb_sample=False):
    '''Samples a value for Y_ij. It should be sampled by:
    - if dumb_sample=True: the probability conditioned on all other variables
    - if dumb_sample=False: the consensus of Markov blanket

    Args
    - i: int, row index of Y
    - j: int, col index of Y
    - Y: np.array, shape [H, W], note that read_txt_file() pads the image with
            0's to help you take care of corner cases
    - X: np.array, shape [H, W]

    Returns: -1 or +1

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    blanket = markov_blanket(i, j, Y, X)

    if not dumb_sample:
        prob = sampling_prob(blanket)
        return np.random.choice([+1, -1], p=[prob, 1 - prob])
    else:
        return 1 if sum(blanket) > 0 else -1


def compute_energy(Y, X):
    '''Computes the energy E(Y, X) of the current assignment for the image.

    Args
    - Y: np.array, shape [H, W], note that read_txt_file() pads the image with
            0's to help you take care of corner cases
    - X: np.array, shape [H, W]

    Returns: float

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.

    This function can be efficiently implemented in one line with numpy parallel operations.
    You can give it a try to speed up your own experiments. This is not required.
    '''
    energy = 0.0
    ########
    if np.all(Y.shape == X.shape):
        Y_padded = copy.deepcopy(Y)
        h, w = Y.shape
        h -= 2
        w -= 2
        Y = copy.deepcopy(Y[1:h+1, 1:w+1])
    else:
        h, w = Y.shape
        Y_padded = np.zeros_like(X)
        Y_padded[1:h+1, 1:w+1] = Y
    blanket = np.zeros_like(Y)
    blanket = np.stack([blanket]*5, axis=2)
    blanket[...,0] = Y_padded[:h,1:w+1]
    blanket[...,1] = Y_padded[1:h+1,:w]
    blanket[...,2] = Y_padded[2:h+2,1:w+1]
    blanket[...,3] = Y_padded[1:h+1,2:w+2]
    blanket[...,4] = X[1:h+1,1:w+1]
    energy = np.multiply(np.expand_dims(Y, axis=2), blanket)
    weight = np.ones([1,1,5])*0.5
    weight[...,4] = 1.
    energy = np.multiply(energy, weight)
    energy = -np.sum(energy)
    # energy = np.multiply(np.expand_dims(Y, axis=2), blanket)
    # import pdb; pdb.set_trace()
    ########
    return energy


def get_posterior_by_sampling(filename, max_burns, max_samples,
                              initialization='same', logfile=None,
                              dumb_sample=False):
    '''Performs Gibbs sampling and computes the energy of each  assignment for
    the image specified in filename.
    - If dumb_sample=False: runs max_burns iterations of burn in and then
        max_samples iterations for collecting samples
    - If dumb_sample=True: runs max_samples iterations and returns final image

    Args
    - filename: str, file name of image in text format, ends in '.txt'
    - max_burns: int, number of iterations of burn in
    - max_samples: int, number of iterations of collecting samples
    - initialization: str, one of ['same', 'neg', 'rand']
    - logfile: str, file name for storing the energy log (to be used for
        plotting later), see plot_energy()
    - dumb_sample: bool, True to use the trivial reconstruction in part (e)

    Returns
    - posterior: np.array, shape [H, W], type float64, value of each entry is
        the probability of it being 1 (estimated by the Gibbs sampler)
    - Y: np.array, shape [H, W], type np.int32,
        the final image (for dumb_sample=True, in part (e))
    - frequencyZ: dict, keys: count of the number of 1's in the Z region,
        values: frequency of such count

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    print ('Reading file:', filename)
    X = read_txt_file(filename)
    ########
    h, w = X.shape
    if initialization == 'same':
        Y = X[1:h-1, 1:w-1]
    elif initialization == 'neg':
        Y = -X[1:h-1, 1:w-1]
    elif initialization == 'rand':
        Y = np.random.choice([-1,1],size=(h-2,w-2))
    if logfile is not None:
        f = open(logfile, 'w+')
    if dumb_sample:
        itr = 0
        for _ in range(max_burns):
            Y = batch_sample(X,Y,True,True)
            energy = compute_energy(Y, X)
            if logfile is not None: 
                f.write('%i\t%f\tB\r\n'%(itr,energy))
            itr += 1
        if logfile is not None: 
            f.close()
        posterior = None
        frequencyZ = None
    else:
        itr = 0
        for _ in range(max_burns):
            Y = batch_sample(X,Y,True)
            Y = batch_sample(X,Y,False)
            energy = compute_energy(Y, X)
            if logfile is not None: 
                f.write('%i\t%f\tB\r\n'%(itr,energy))
            itr += 1
        samples = []
        energies = []
        Z_counts = []
        for _ in range(max_samples):
            Y = batch_sample(X,Y,True)
            Y = batch_sample(X,Y,False)
            Z = Y[125:162,143:174]
            Z_counts.append(np.sum(Z==1))
            samples.append(Y)
            energy = compute_energy(Y, X)
            if logfile is not None: 
                f.write('%i\t%f\tS\r\n'%(itr,energy))
            itr += 1
        if logfile is not None: 
            f.close()

        posterior = np.sum((np.array(samples)+1.)/2., axis=0)/max_samples
        vals, keys = np.histogram(Z_counts, bins=np.arange(650,725), range=None, normed=None, weights=None, density=None)
        frequencyZ = dict(zip(keys,vals))
    ########
    return posterior, Y, frequencyZ


def denoise_image(filename, max_burns, max_samples, initialization='rand',
                  logfile=None, dumb_sample=False):
    '''Performs Gibbs sampling on the image.

    Args:
    - filename: str, file name of image in text format, ends in '.txt'
    - max_burns: int, number of iterations of burn in
    - max_samples: int, number of iterations of collecting samples
    - initialization: str, one of ['same', 'neg', 'rand']
    - logfile: str, file name for storing the energy log (to be used for
        plotting later), see plot_energy()
    - dumb_sample: bool, True to use the trivial reconstruction in part (e)

    Returns
    - denoised: np.array, shape [H, W], type float64,
        denoised image scaled to [0, 1], the zero padding is also removed
    - frequencyZ: dict, keys: count of the number of 1's in the Z region,
        values: frequency of such count

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    posterior, Y, frequencyZ = get_posterior_by_sampling(
        filename, max_burns, max_samples, initialization, logfile=logfile,
        dumb_sample=dumb_sample)

    if dumb_sample:
        denoised = 0.5 * (Y + 1.0)  # change Y scale from [-1, +1] to [0, 1]
    else:
        denoised = np.zeros(posterior.shape, dtype=np.float64)
        denoised[posterior > 0.5] = 1
    return denoised, frequencyZ


# ===========================================
# Helper functions for plotting etc
# ===========================================
def plot_energy(filename):
    '''Plots the energy as a function of the iteration number.

    Args
    - filename: str, path to energy log file, each row has three terms
        separated by a '\t'
        - iteration: iteration number
        - energy: the energy at this iteration
        - 'S' or 'B': indicates whether it's burning in or a sample

    e.g.
        1   -202086.0   B
        2   -210446.0   S
        ...
    '''
    x = np.genfromtxt(filename, dtype=None, encoding='utf8')
    its, energies, phases = zip(*x)
    its = np.asarray(its)
    energies = np.asarray(energies)
    phases = np.asarray(phases)

    burn_mask = (phases == 'B')
    samp_mask = (phases == 'S')
    assert np.sum(burn_mask) + np.sum(samp_mask) == len(x), 'Found bad phase'

    its_burn, energies_burn = its[burn_mask], energies[burn_mask]
    its_samp, energies_samp = its[samp_mask], energies[samp_mask]

    p1, = plt.plot(its_burn, energies_burn, 'r')
    p2, = plt.plot(its_samp, energies_samp, 'b')
    plt.title(filename)
    plt.xlabel('iteration number')
    plt.ylabel('energy')
    plt.legend([p1, p2], ['burn in', 'sampling'])
    plt.savefig('%s.png' % filename)
    plt.close()


def read_txt_file(filename):
    '''Reads in image from .txt file and adds a padding of 0's.

    Args
    - filename: str, image filename, ends in '.txt'

    Returns
    - Y: np.array, shape [H, W], type int32, padded with a border of 0's to
        take care of edge cases in computing the Markov blanket
    '''
    f = open(filename, 'r')
    lines = f.readlines()
    height = int(lines[0].split()[1].split('=')[1])
    width = int(lines[0].split()[2].split('=')[1])
    Y = np.zeros([height+2, width+2], dtype=np.int32)
    for line in lines[2:]:
        i, j, val = [int(entry) for entry in line.split()]
        Y[i+1, j+1] = val
    return Y


def convert_to_png(denoised_image, title):
    '''Saves an array as a PNG file with the given title.

    Args
    - denoised_image: np.array, shape [H, W]
    - title: str, title and filename for figure
    '''
    plt.imshow(denoised_image, cmap='gray_r')
    plt.title(title)
    plt.savefig(title + '.png')


def get_error(img_a, img_b):
    '''Computes the fraction of all pixels that differ between two images.

    Args
    - img_a: np.array, shape [H, W]
    - img_b: np.array, shape [H, W]

    Returns: float
    '''
    assert img_a.shape == img_b.shape
    N = img_a.shape[0] * img_a.shape[1]  # number of pixels in an image
    return np.sum(np.abs(img_a - img_b) > 1e-5) / float(N)


#==================================
# doing part (c), (d), (e), (f)
#==================================

def batch_sample(X,Y,even=True,dumb_sample=False):
    h, w = Y.shape
    mask = np.zeros_like(Y) 
    mask[::2, ::2] = 1
    mask[1::2, 1::2] = 1
    if not even:
        mask = (mask==0).astype(np.int32)
    Y_padded = np.zeros_like(X)
    Y_padded[1:h+1, 1:w+1] = Y
    blanket = np.zeros_like(Y)
    blanket = np.stack([blanket]*5, axis=2)
    blanket[...,0] = Y_padded[:h,1:w+1]
    blanket[...,1] = Y_padded[1:h+1,:w]
    blanket[...,2] = Y_padded[2:h+2,1:w+1]
    blanket[...,3] = Y_padded[1:h+1,2:w+2]
    blanket[...,4] = X[1:h+1,1:w+1]
    sums = np.sum(blanket, axis=2)
    if dumb_sample:
        Y = np.sign(sums)
    else:
        probs = np.exp(sums)/(np.exp(sums) + np.exp(-sums))
        logits = np.random.random(size=probs.shape)
        sample_1 = (logits <= probs).astype(np.int32)*2 - 1
        Y = Y*(mask==0).astype(np.int32) + sample_1*mask
    return Y 

def perform_part_c():
    '''
    Run denoise_image() with different initializations and plot out the energy
    functions.
    '''
    ########
    denoise_image('./data/noisy_20.txt', 100, 1000, initialization='rand',
                  logfile='log_rand', dumb_sample=False)
    denoise_image('./data/noisy_20.txt', 100, 1000, initialization='neg',
                  logfile='log_neg', dumb_sample=False)
    denoise_image('./data/noisy_20.txt', 100, 1000, initialization='same',
                  logfile='log_same', dumb_sample=False)
    ########

    #### plot out the energy functions
    plot_energy('log_rand')
    plot_energy('log_neg')
    plot_energy('log_same')


def perform_part_d():
    '''
    Run denoise_image() with different noise levels of 10% and 20%, and report
    the errors between denoised images and original image. Strip the 0-padding
    before computing the errors. Also, don't forget that denoise_image() strips
    the zero padding and scales them into [0, 1].
    '''
    ########

    orig_img = read_txt_file('./data/orig.txt')
    orig_img = orig_img + 1
    orig_img = orig_img/2

    denoised_10, _ = denoise_image('./data/noisy_10.txt', 100, 1000, initialization='rand',
                  logfile=None, dumb_sample=False)
    denoised_20, _  = denoise_image('./data/noisy_20.txt', 100, 1000, initialization='rand',
                  logfile=None, dumb_sample=False)
    
    diff_10 = np.sum((orig_img[1:-1,1:-1] - denoised_10) != 0)
    diff_20 = np.sum((orig_img[1:-1,1:-1] - denoised_20) != 0)
    print()
    print('Denoised 10 error pixels: ', int(diff_10)) 
    print('Denoised 20 error pixels: ', int(diff_20)) 
    print()
    # Denoised 10 error pixels:  510
    # Denoised 20 error pixels:  899
    ########

    #### save denoised images and original image as PNG
    convert_to_png(denoised_10, 'denoised_10')
    convert_to_png(denoised_20, 'denoised_20')
    convert_to_png(orig_img, 'orig_img')


def perform_part_e():
    '''
    Run denoise_image() using dumb sampling with different noise levels of 10%
    and 20%.
    '''
    ########
    orig_img = read_txt_file('./data/orig.txt')
    orig_img = orig_img + 1
    orig_img = orig_img/2

    denoised_dumb_10, _ = denoise_image('./data/noisy_10.txt', 1, 30, initialization='same',
                  logfile=None, dumb_sample=True)
    denoised_dumb_20, _  = denoise_image('./data/noisy_20.txt', 100, 30, initialization='same',
                  logfile=None, dumb_sample=True)

    diff_10 = np.sum((orig_img[1:-1,1:-1] - denoised_dumb_10) != 0)
    diff_20 = np.sum((orig_img[1:-1,1:-1] - denoised_dumb_20) != 0)
    print()
    print('Denoised 10 error pixels: ', int(diff_10)) 
    print('Denoised 20 error pixels: ', int(diff_20)) 
    print()
    ########

    #### save denoised images as PNG
    convert_to_png(denoised_dumb_10, 'denoised_dumb_10')
    convert_to_png(denoised_dumb_20, 'denoised_dumb_20')


def perform_part_f():
    '''
    Run Z square analysis
    '''
    MAX_BURNS = 100
    MAX_SAMPLES = 1000

    _, f = denoise_image('./data/noisy_10.txt', MAX_BURNS, MAX_SAMPLES, initialization='same')
    width = 1.0
    plt.clf()
    plt.bar(list(f.keys()), list(f.values()), width, color='b')
    plt.show()
    _, f = denoise_image('./data/noisy_20.txt', MAX_BURNS, MAX_SAMPLES, initialization='same')
    plt.clf()
    plt.bar(list(f.keys()), list(f.values()), width, color='b')
    plt.show()


if __name__ == '__main__':
    # perform_part_c()
    # perform_part_d()
    perform_part_e()
    # perform_part_f()
