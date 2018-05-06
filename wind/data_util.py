import numpy as np 
import scipy.io
import glob 
import pickle

X = []
Y = []

n=0
for filename in glob.glob('./FAST/*.mat'):
	data = scipy.io.loadmat(filename)['avg'].squeeze()
	xlist = []
	ylist = []
	for dat in data:
		x = np.zeros([4,])
		x[0] = dat[5]
		x[1] = dat[6]
		x[2] = dat[7]
		x[3] = dat[8]
		y = np.array(dat[9][0])
		xlist.append(x)
		ylist.append(y)
	xarray = np.stack(xlist[19:])
	yarray = np.stack(ylist[19:])
	n += yarray.shape[0]
	if n >= 600 and n<=800:
		print('%r files parsed' % n)
		print('Filename: %r' % filename)
	X.append(xarray)
	Y.append(yarray)
print('%r datapoints parsed' % n)
Xmlp = np.concatenate(X)
Ymlp = np.concatenate(Y)
with open('./data/Xmlp.p', 'wb') as f: 
	pickle.dump(Xmlp, f)

with open('./data/Ymlp.p', 'wb') as f: 
	pickle.dump(Ymlp, f)

