import numpy as np 
import pickle 
# import scipy 
# from scipy import sparse 
# from scipy.sparse import linalg 
import matplotlib.pyplot as plt 

import pdb
import time

with open('./Pickles/S_example.p', 'rb') as f: 
	S = pickle.load(f)

S = S[::-1].reshape([-1,1])
S_ = S[10:]
lS = np.log(S_)
lX = np.log(np.arange(90)+ 11.0).reshape([-1,1]) 
lX_ = np.hstack([lX, np.ones([90,1])])
# pdb.set_trace()
A, _, _, _ = np.linalg.lstsq(lX_, lS)
k = -A[0]
a = np.exp(A[1])

# St = np.sum((np.arange(100)+1.0)**(-k))
# S_ = np.sum((np.arange(20000)+1.0)**(-k))
# print(St/S_)

# pdb.set_trace()

plt.figure()
plt.plot(np.arange(100)+1, S, 's')
plt.plot(np.arange(100)+1, (np.arange(100)+1.0)**(-k)*a)
plt.yscale('log')
plt.xscale('log')
plt.title('Eigen-value/Power-Law fit')
plt.xlabel('Eigen-value Index')
plt.ylabel('Eigen-Value')
plt.legend(['True', 'Predicted'])
plt.grid(True)
plt.savefig('./Plots/PowerLaw.png')
plt.close()

plt.figure()
plt.plot(np.arange(100)+1, S, 's')
plt.plot(np.arange(15000)+1, (np.arange(15000)+1.0)**(-k)*a)
plt.title('Eigen-value Decay')
plt.xlabel('Eigen-value Index')
plt.ylabel('Eigen-Value')
plt.legend(['True', 'Predicted'])
plt.savefig('./Plots/EigDecay.png')
plt.show()
