import numpy as np
# import matplotlib.pyplot as plt

import pdb

def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def softmax(x):
    """
    Compute softmax function for input. 
    Use tricks from previous assignment to avoid overflow
    """
	### YOUR CODE HERE
    m = np.amax(x)
    ex = np.exp(x - m)

    logs = np.log(ex) - np.log(np.sum(ex, axis=1,keepdims=True))
    s = np.exp(logs)
	### END YOUR CODE
    return s

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    ### YOUR CODE HERE
    # s = (1 + np.exp(-x))**(-1)

    maskp = x >= 0
    maskn = x < 0
    xp = np.multiply(x, maskp)
    xn = np.multiply(x, maskn)
    sp = (1 + np.exp(-xp))**(-1)
    sn = np.divide(np.exp(xn), 1 + np.exp(xn)) 
    s = np.multiply(sp, maskp) + np.multiply(sn, maskn)

    ### END YOUR CODE
    return s

def cxentropy(x,y):
    v = np.multiply(np.log(x), y)
    ce = -np.sum(v,axis=1)
    return ce

def forward_prop(data, labels, params):
    """
    return hidder layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    # pdb.set_trace()
    y1 = data.dot(W1) + b1
    h = sigmoid(y1)
    y2 = h.dot(W2) + b2 
    y = softmax(y2)
    loss = cxentropy(y, labels)
    ### END YOUR CODE
    return h, y, loss

def backward_prop(data, labels, params):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    h, output, loss = forward_prop(data, labels, params)
    m, n = data.shape
    dLdy2 = (output - labels)/m 
    dy2dh = W2
    dLdh = dLdy2.dot(dy2dh.T)
    dhdy1 = np.multiply((1.0 - h),dot(h))
    dLdy1 = np.multiply(dLdh,dhdy1)

    gradW2 = h.T.dot(dLdy2) 
    gradb2 = np.sum(dLdy2, axis=0)
    gradW1 = data.T.dot(dLdy1)
    gradb1 = np.sum(dLdy1, axis=0)

    ### END YOUR CODE

    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2
    # pdb.set_trace()
    return grad

def nn_train(trainData, trainLabels, devData, devLabels):
    (m, n) = trainData.shape
    num_hidden = 300
    learning_rate = 5
    params = {}

    ### YOUR CODE HERE
    B = 1000
    E = 30
    I = int(m/B)

    # Initialize Network 
    W1 = np.random.normal(loc=0.0, scale=1.0,size=[n,num_hidden])
    W2 = np.random.normal(loc=0.0, scale=1.0,size=[num_hidden,10])
    b1 = np.zeros([1, num_hidden])
    b2 = np.zeros([1, 10])
    params['W1'] = W1
    params['W2'] = W2
    params['b1'] = b1
    params['b2'] = b2

    # Train Network 

    TLoss = []
    DLoss = []
    for e in xrange(E):
        for i in xrange(I):
            startidx = i*B
            endidx = B*(i+1)
            data_batch = trainData[startidx:endidx, :]
            label_batch = trainLabels[startidx:endidx, :]
            grad = backward_prop(data_batch, label_batch, params)  
            # pdb.set_trace()
            for key, value in grad.items(): 
                # pdb.set_trace()
                params[key] =  params[key] - learning_rate*value #np.mean(value,axis=0, keepdims=True)
            # pdb.set_trace()
        _, _, train_loss = forward_prop(trainData, trainLabels, params)
        _, _, dev_loss = forward_prop(devData, devLabels, params)
        train_loss = np.mean(train_loss)
        dev_loss = np.mean(dev_loss)
        print('Epoch %r done with training loss: %r and development loss: %r' % (e, train_loss, dev_loss))
        pdb.set_trace()
        TLoss.append(train_loss)
        DLoss.append(dev_loss)
    # pdb.set_trace()
    ### END YOUR CODE

    return params

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def main():
    np.random.seed(100)
    trainData, trainLabels = readData('images_train.csv', 'labels_train.csv')
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)
    trainData = trainData[p,:]
    trainLabels = trainLabels[p,:]

    devData = trainData[0:10000,:]
    devLabels = trainLabels[0:10000,:]
    trainData = trainData[10000:,:]
    trainLabels = trainLabels[10000:,:]

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testData, testLabels = readData('images_test.csv', 'labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std
	
    params = nn_train(trainData, trainLabels, devData, devLabels)
    accuracy = nn_test(trainData, trainLabels, params)
    print accuracy 
    readyForTesting = False
    if readyForTesting:
        accuracy = nn_test(testData, testLabels, params)
	print 'Test accuracy: %f' % accuracy

if __name__ == '__main__':
    main()
