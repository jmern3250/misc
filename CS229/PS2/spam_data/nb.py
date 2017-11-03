import numpy as np
import pdb

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    state = {}
    m, k = matrix.shape
    ###################
    den1 = np.sum(category)
    den0 = m - den1 
    mask1 = category
    mask0 = 1 - category 
    vals = np.unique(matrix)
    min_val = 0
    max_val = 300
    keys = np.arange(min_val, max_val)
    num1 = np.zeros([0,k])
    num0 = np.zeros([0,k])
    for key in keys: 
        mat1 = matrix*mask1.reshape([-1,1])
        mat0 = matrix*mask0.reshape([-1,1])
        mat1 = mat1 == key
        mat0 = mat0 == key
        num1 = np.vstack([num1, np.sum(mat1, axis=0)])
        num0 = np.vstack([num0, np.sum(mat0, axis=0)])

    phi_1 = (num1 + 1.0)/(den1 + k) 
    phi_0 = (num0 + 1.0)/(den0 + k) 
    phi_y = np.mean(category)
    state['phi_1'] = phi_1
    state['phi_0'] = phi_0
    state['phi_y'] = phi_y
    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    m, k = matrix.shape
    ###################
    phi_1 = state['phi_1']
    phi_0 = state['phi_0'] 
    phi_y = state['phi_y']


    P1 = np.zeros([0,1])
    P0 = np.zeros([0,1])
    for i in range(m):
        p1 = 1.0
        p0 = 1.0
        X = matrix[i,:] 
        for j in range(k):
            x = int(X[j])
            p1 *= phi_1[x, j]
            p0 *= phi_0[x, j]
        p1 *= phi_y
        p0 *= (1.0 - phi_y)
        P1 = np.vstack([P1, p1]) 
        P0 = np.vstack([P0, p0]) 
    den = P1[:] + P0[:]
    P1 /= den
    P0 /= den
    P = np.hstack([P0, P1])
    output = np.argmax(P, axis=1)
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print 'Error: %1.4f' % error

def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')
    # pdb.set_trace()
    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
    return

if __name__ == '__main__':
    main()
