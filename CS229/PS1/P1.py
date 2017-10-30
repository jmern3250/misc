import numpy as np 
import csv 

import pdb

Xlist = []
with open('logistic_x.txt') as f: 
	Xcsv = csv.reader(f, delimiter=' ', skipinitialspace=True)
	for x in Xcsv:
		Xlist.append(x)

Ylist = []
with open('logistic_y.txt') as f: 
	Ycsv = csv.reader(f, delimiter=' ', skipinitialspace=True)
	for y in Ycsv:
		Ylist.append(y)

m = len(Xlist) - 1
X = np.ones([m, 3])
Y = np.zeros([m, 1])

for i in range(m):
	X[i,:2] = Xlist[i]
	Y[i,:] = Ylist[i]

theta = np.zeros([3,])


def jacobian(theta, x, y):
	m = x.shape[0]
	jac = np.zeros([1,theta.size])
	for i in range(m):
		jac += np.divide(x[i], theta.T.dot(x[i]))
	jac /= -m
	return jac 

jac = jacobian(theta, X, Y)

pdb.set_trace()