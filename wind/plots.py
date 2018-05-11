import numpy as np 
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

from mlp import * 

with open('./data/X2.p', 'rb') as f: 
	Xd = pickle.load(f)
with open('./data/M2.p', 'rb') as f: 
	Md = pickle.load(f)
with open('./data/Y2.p', 'rb') as f: 
	Yd = pickle.load(f)
Xmean = np.mean(Xd, axis=0)
Xstd = np.std(Xd, axis=0)
Mmean = np.mean(Md, axis=0)
Mstd = np.std(Md, axis=0)
Ymean = np.mean(Yd, axis=0)
Ystd = np.std(Yd, axis=0)

Xd_ = Xd - Xmean
Xd_ /= Xstd

Md_ = Md - Mmean
Md_ /= Mstd

Yd_ = Yd - Ymean
Yd_ /= Ystd

n_samples = Xd.shape[0]

Xtrain = Xd_[:(n_samples-100), ...]
Mtrain = Md_[:(n_samples-100), ...]
Ytrain = Yd_[:(n_samples-100), ...]
Xtest = Xd_[(n_samples-100):, ...]
Mtest = Md_[(n_samples-100):, ...]
Ytest = Yd[(n_samples-100):, ...]

mlp = MLP(5, 256, lrelu, scope='mlp')
mlp.restore_graph('./models_v2/model1')


names = ['K1', 'K2', 'Keq', 'dVL', 'dVU']

##### Training Predictions #####
y_ = mlp.predict(Xtrain, Mtrain)
y_ *= Ystd
y_ += Ymean
Ytrain *= Ystd
Ytrain += Ymean

### Plot Ypred vs Y ### 

for i in range(5):
	min_val = np.amin(y_[:,i])
	max_val = np.amax(y_[:,i])
	fname = './results/mlp/train/' + 'ys_' + names[i] + '.png'
	plt.figure()
	plt.plot(Ytrain[:,i].squeeze(), y_[:,i].squeeze(), '.')
	plt.plot([min_val,max_val], [min_val, max_val], '-k')
	plt.grid(True)
	plt.xlabel(names[i] + ' True')
	plt.ylabel(names[i] + ' Predicted')
	plt.title(names[i] + ' True vs. Predicted')
	plt.savefig(fname)
	plt.close()	

### Plot Ypred and Y for samples ### 

for i in range(5):
	fname = './results/mlp/train/' + 'pred_' + names[i] + '.png'
	plt.figure()
	plt.plot(Ytrain[:2500,i].squeeze(), '-')
	plt.plot(y_[:2500,i].squeeze(), '.', ms=2)
	plt.grid(True)
	plt.xlabel('Sample Number')
	plt.ylabel(names[i])
	plt.title(names[i] + ' Predictions')
	plt.legend(['Truth Data', 'Predictions'])
	plt.savefig(fname)
	plt.close()	

### Plot residuals ### 

residuals = y_ - Ytrain

for i in range(5):
	fname = './results/mlp/train/' + 'res_' + names[i] + '.png'
	plt.figure()
	plt.plot(residuals[:,i].squeeze(), '.')
	plt.plot([0,n_samples], [0,0], '-k')
	plt.grid(True)
	plt.xlabel('Sample Number')
	plt.ylabel('Residual Value')
	plt.title(names[i] + ' Residuals')
	plt.savefig(fname)
	plt.close()	

MSE = np.mean(residuals**2, axis=0)
RMSE = np.sqrt(MSE)
print('Training MSE:', MSE)
print('Training RMSE:', RMSE)

##### Test Predictions #####
y_ = mlp.predict(Xtest, Mtest)
y_ *= Ystd
y_ += Ymean

### Plot Ypred vs Y ### 

for i in range(5):
	min_val = np.amin(y_[:,i])
	max_val = np.amax(y_[:,i])
	fname = './results/mlp/test/' + 'ys_' + names[i] + '.png'
	plt.figure()
	plt.plot(Ytest[:,i].squeeze(), y_[:,i].squeeze(), '.')
	plt.plot([min_val,max_val], [min_val, max_val], '-k')
	plt.grid(True)
	plt.xlabel(names[i] + ' True')
	plt.ylabel(names[i] + ' Predicted')
	plt.title(names[i] + ' True vs. Predicted')
	plt.savefig(fname)
	plt.close()	

### Plot Ypred and Y for samples ### 

for i in range(5):
	fname = './results/mlp/test/' + 'pred_' + names[i] + '.png'
	plt.figure()
	plt.plot(Ytest[:,i].squeeze(), '-')
	plt.plot(y_[:,i].squeeze(), '.', ms=2)
	plt.grid(True)
	plt.xlabel('Sample Number')
	plt.ylabel(names[i])
	plt.title(names[i] + ' Predictions')
	plt.legend(['Truth Data', 'Predictions'])
	plt.savefig(fname)
	plt.close()	

### Plot residuals ### 

residuals = y_ - Ytest

for i in range(5):
	fname = './results/mlp/test/' + 'res_' + names[i] + '.png'
	plt.figure()
	plt.plot(residuals[:,i].squeeze(), '.')
	plt.plot([0,100], [0,0], '-k')
	plt.grid(True)
	plt.xlabel('Sample Number')
	plt.ylabel('Residual Value')
	plt.title(names[i] + ' Residuals')
	plt.savefig(fname)
	plt.close()

MSE = np.mean(residuals**2, axis=0)
RMSE = np.sqrt(MSE)
print('Testing MSE:', MSE)
print('Testing RMSE:', RMSE)