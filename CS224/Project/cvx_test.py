import numpy as np 
# import cvxpy as cvx 
import scipy 
from scipy import optimize 

import pdb

N = 100
M = 10

A = np.random.uniform(size=[N,N])
# U, s, V = np.linalg.svd(A)
# S = np.diag(s[0:M])

# B = cvx.Variable(M,M)
# V = cvx.Variable(M,N)

# A_ = cvx.quad_form(V,S)
# obj = cvx.Minimize(cvx.norm(A - A_))
# constraints = []
# prob = cvx.Problem(obj, constraints)

# prob.solve(verbose=True)

# import pdb; pdb.set_trace()

def lossX(X,S,A):
	X_ = X.reshape([10,100])
	A_ = (X_.T.dot(S)).dot(X_)
	loss = np.linalg.norm(A - A_)
	# pdb.set_trace()
	return loss 

def lossS(S,X,A):
	S_ = S.reshape([10,10])
	A_ = (X.T.dot(S_)).dot(X)
	loss = np.linalg.norm(A - A_)
	# pdb.set_trace()
	return loss 

X0 = np.random.uniform(size=[M,N])
S0 = np.random.uniform(size=[M,M])

done = False
while not done: 
	res = scipy.optimize.minimize(lossX, X0, (S,A))
	
import pdb; pdb.set_trace()