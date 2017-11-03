import numpy as np 
import matplotlib.pyplot as plt

from lr_debug import * 

import pdb



def logistic_regression(X, Y, ints):
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    I = []
    Grad = []
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta  - learning_rate * (grad)
        if i % 100 == 0:
        	Grad.append(grad)
        	I.append(i)
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
        if i >= ints:
        	print('Did not converge')
        	break
    return Grad, I

Xa, Ya = load_data('data_a.txt')
Xb, Yb = load_data('data_b.txt')

Xa_p = Xa[Ya>0, :]
Xa_n = Xa[Ya<0, :]

Xb_p = Xb[Yb>0, :]
Xb_n = Xb[Yb<0, :] 

gradA, IA = logistic_regression(Xa, Ya, 50000)

gradB, IB = logistic_regression(Xb, Yb, 50000)

gradA = np.array(gradA)
plt.figure()
plt.plot(IA[200:], gradA[200:,0])
plt.plot(IA[200:], gradA[200:,1])
plt.plot(IA[200:], gradA[200:,2])
plt.title('Dataset A Gradient Decay')
plt.xlabel('Iterations')
plt.ylabel('Gradient Value')
plt.show()

gradB = np.array(gradB)
plt.figure()
plt.plot(IB[200:], gradB[200:,0])
plt.plot(IB[200:], gradB[200:,1])
plt.plot(IB[200:], gradB[200:,2])
plt.title('Dataset B Gradient Decay')
plt.xlabel('Iterations')
plt.ylabel('Gradient Value')
plt.show()

plt.figure()
plt.plot(Xa_p[:,1], Xa_p[:,2], 'x')
plt.plot(Xa_n[:,1], Xa_n[:,2], 'o')
plt.title('Dataset A')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

plt.figure()
plt.plot(Xb_p[:,1], Xb_p[:,2], 'x')
plt.plot(Xb_n[:,1], Xb_n[:,2], 'o')
plt.title('Dataset B')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

Xap_ = []
Yap_ = []

for i in range(Xa_p.shape[0]):
	if 1 - Xa_p[i,1] <= Xa_p[i,2]:
		Xap_.append(Xa_p[i,:])
		Yap_.append(1)

Xap_ = np.array(Xap_)
Yap_ = np.array(Yap_)

Xan_ = []
Yan_ = []

for i in range(Xa_n.shape[0]):
	if 1 - Xa_n[i,1] >= Xa_n[i,2]:
		Xan_.append(Xa_n[i,:])
		Yan_.append(-1)

Xan_ = np.array(Xan_)
Yan_ = np.array(Yan_)
plt.figure()
plt.plot(Xap_[:,1], Xap_[:,2], 'x')
plt.plot(Xan_[:,1], Xan_[:,2], 'o')
plt.title('Truncated Dataset A')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

Xa_ = np.vstack([Xap_, Xan_])
Ya_ = np.vstack([Yap_.reshape([-1,1]), Yan_.reshape([-1,1])]).squeeze()
grad_, I_ = logistic_regression(Xa_, Ya_, 500000)