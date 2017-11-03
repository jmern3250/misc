import numpy as np
import matplotlib.pyplot as plt
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
    pdb.set_trace()
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
    # P1 /= den
    # P0 /= den
    P = np.hstack([P0, P1])
    output = np.argmax(P, axis=1)
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    # print 'Error: %1.4f' % error
    return error

def top_tokens(state, tokenlist, n_top):
    n = len(tokenlist)
    # Scores = np.array([n,])
    phi_1 = state['phi_1']
    phi_0 = state['phi_0']
    p1 = np.sum(phi_1, axis=0)
    p0 = np.sum(phi_0, axis=0) 
    score = np.log(p1) - np.log(p0)
    top_list = []
    for _ in range(n_top):
        top_idx = np.argmax(score)
        top_token = tokenlist[top_idx]
        top_list.append(top_token)
        score[top_idx] = -np.inf
    return top_list

def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    # testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')
    # # pdb.set_trace()
    # state = nb_train(trainMatrix, trainCategory)
    # output = nb_test(testMatrix, state)

    # evaluate(output, testCategory)

    # top_list = top_tokens(state, tokenlist, 5)
    # print(top_list)
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')
    train_files = ['MATRIX.TRAIN.50','MATRIX.TRAIN.100','MATRIX.TRAIN.200','MATRIX.TRAIN.400','MATRIX.TRAIN.800','MATRIX.TRAIN.1400']
    Size = [50, 100, 200, 400, 800, 1400]
    Error = []
    for f in train_files:
        trainMatrix, _, trainCategory = readMatrix(f)
        state = nb_train(trainMatrix, trainCategory)
        output = nb_test(testMatrix, state)
        error = evaluate(output, testCategory)
        Error.append(error)

    pdb.set_trace()
    plt.figure()
    plt.plot(Size, Error)
    plt.title('Test Error vs Training Set Size')
    plt.xlabel('Set Size')
    plt.ylabel('Test Error')
    plt.show()
    return

if __name__ == '__main__':
    main()
