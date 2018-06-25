import numpy as np 
import scipy.io
import glob 
import pickle

X = []
M = []
Y = []
# Y0 = []

'''	
    Unused: idxS idxE tS tE 
    M: avgRootM 
    X: PitchAngleB1 tPitchAngleB1 GenTorque tGenTorque GenSpeed tGenSpeed GenPower tGenPower
    Y0: Veq Seq u v dir veer 
    Y: K1 Keq dVL dVU K2
    '''
for filename in glob.glob('./data/DS2/FAST/*.mat'):
	data = scipy.io.loadmat(filename)['avg'].squeeze()
	xlist = []
	mlist = []
	ylist = []
	# y0list = []
	for dat in data:
		x = np.zeros([8,])
		m = dat[4].squeeze()
		y0 = np.zeros([6, ])
		y = np.zeros([5,])
		x[0] = dat[5]
		x[1] = dat[6]
		x[2] = dat[7]
		x[3] = dat[8]
		x[4] = dat[9]
		x[5] = dat[10]
		x[6] = dat[11]
		x[7] = dat[12]
		# y0[0] = dat[13]
		# y0[1] = dat[14]
		# y0[2] = dat[15]
		# y0[3] = dat[16]
		# y0[4] = dat[17]
		# y0[5] = dat[18]
		y[0] = dat[19]
		y[1] = dat[20]
		y[2] = dat[21]
		y[3] = dat[22]
		y[4] = dat[23]
		xlist.append(x)
		mlist.append(m)
		ylist.append(y)
		# y0list.append(y0)
	xarray = np.stack(xlist)
	marray = np.stack(mlist)
	yarray = np.stack(ylist)
	# y0array = np.stack(y0list)
	X.append(xarray)
	M.append(marray)
	Y.append(yarray)
	# Y0.append(y0array)
# print('%r datapoints parsed' % n)
Xmlp = np.concatenate(X)
Mmlp = np.concatenate(M)
Ymlp = np.concatenate(Y)
# Y0mlp = np.concatenate(Y0)


with open('./data/X3.p', 'wb') as f: 
	pickle.dump(Xmlp, f)

with open('./data/M3.p', 'wb') as f: 
	pickle.dump(Mmlp, f)

with open('./data/Y3.p', 'wb') as f: 
	pickle.dump(Ymlp, f)

# with open('./data/Y03.p', 'wb') as f: 
# 	pickle.dump(Y0mlp, f)