import numpy as np 
import scipy.io
import glob 
import pickle
from data_smooth import exp_smooth
X = []
Y = []

# n=0
for filename in glob.glob('./data/FAST/v2/*.mat'):
	data = scipy.io.loadmat(filename)['avg'].squeeze()
	xlist = []
	ylist = []
	for dat in data:
		x = np.zeros([5,])
		y = np.zeros([5,])
		x[0] = dat[7]
		x[1] = dat[8]
		x[2] = dat[9]
		x[3] = dat[10]
		x[4] = dat[11]
		y[0] = dat[14]
		y[1] = dat[15]
		y[2] = dat[16]
		y[3] = dat[17]
		y[4] = dat[18]
		xlist.append(x)
		ylist.append(y)
	xarray = np.stack(xlist)
	yarray = np.stack(ylist)
	Xsmooth = exp_smooth(xarray, 0.25, 3)
	Ysmooth = exp_smooth(yarray, 0.25, 3)
	X.append(Xsmooth)
	Y.append(Ysmooth)

Xmlp = np.concatenate(X)
Ymlp = np.concatenate(Y)

with open('./data/Xsmooth.p', 'wb') as f: 
	pickle.dump(Xmlp, f)

with open('./data/Ysmooth.p', 'wb') as f: 
	pickle.dump(Ymlp, f)

