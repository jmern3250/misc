import numpy as np 
import pickle 

def exp_smooth(Xraw, alpha, n=1):
	X_ = Xraw
	for _ in range(n):
		Xsmooth = np.zeros_like(X_)
		Xsmooth[0,:] = X_[0,:]
		for i in range(Xsmooth.shape[0]-1):
			Xsmooth[i+1,:] = X_[i+1,:]*alpha + Xsmooth[i,:]*(1.-alpha)
		X_ = Xsmooth
	return Xsmooth 


if __name__ == '__main__':
	import matplotlib.pyplot as plt 

	ALPHA = 0.25

	with open('./data/Xtest.p', 'rb') as f: 
		Xraw = pickle.load(f)

	with open('./data/Ytest.p', 'rb') as f: 
		Yraw = pickle.load(f)

	# Xsmooth = exp_smooth(Xraw, ALPHA)
	# plt.figure()
	# plt.plot(Xraw[:,2], '.')
	# plt.plot(Xsmooth[:,2], '-')
	# plt.show()
	Ysmooth = exp_smooth(Yraw, ALPHA, 3)
	plt.figure()
	plt.plot(Yraw[:,0], '.')
	plt.plot(Ysmooth[:,0], '-')
	plt.show()