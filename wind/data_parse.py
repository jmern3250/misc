import numpy as np 
import scipy.io
import glob 
import pickle

X = []
M = []
Y = []

# n=0
for filename in glob.glob('./data/FAST/v2/*.mat'):
	data = scipy.io.loadmat(filename)['avg'].squeeze()
	xlist = []
	mlist = []
	ylist = []
	for dat in data:
		# import pdb; pdb.set_trace()
		x = np.zeros([7,])
		m = dat[4].squeeze()
		y = np.zeros([5,])
		x[0] = dat[5]
		x[1] = dat[6]
		x[2] = dat[7]
		x[3] = dat[8]
		x[4] = dat[9]
		x[5] = dat[10]
		x[6] = dat[11]
		y[0] = dat[14]
		y[1] = dat[15]
		y[2] = dat[16]
		y[3] = dat[17]
		y[4] = dat[18]
		xlist.append(x)
		mlist.append(m)
		ylist.append(y)
	xarray = np.stack(xlist)
	marray = np.stack(mlist)
	yarray = np.stack(ylist)
	X.append(xarray)
	M.append(marray)
	Y.append(yarray)

# print('%r datapoints parsed' % n)
Xmlp = np.concatenate(X)
Mmlp = np.concatenate(M)
Ymlp = np.concatenate(Y)

with open('./data/X2.p', 'wb') as f: 
	pickle.dump(Xmlp, f)

with open('./data/M2.p', 'wb') as f: 
	pickle.dump(Mmlp, f)

with open('./data/Y2.p', 'wb') as f: 
	pickle.dump(Ymlp, f)

