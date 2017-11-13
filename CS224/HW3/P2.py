import numpy as np
import snap 
import csv 
import matplotlib.pyplot as plt 

import pdb 

def sample_powerlaw(alpha, xmin, N):
	u = np.random.uniform(low=0.0, high=1.0, size=N)
	# pdb.set_trace()
	x = xmin*(1-u)**(1.0/(1.0-alpha))
	return x 

def powerlaw(X, alpha, xmin):
	coef = (alpha - 1.0)/xmin
	val = (x/xmin)**(-alpha)
	return coef*val

X_ = sample_powerlaw(2.0, 1.0, 100000)
X = np.round(X_).astype(np.int64)

x = np.arange(1.0,101.0)
p = powerlaw(x, 2.0, 1.0)

hist, bins = np.histogram(X, bins=100, range=(1,100))

# plt.figure()
# plt.loglog(bins[1:], hist.astype(np.float64)/100000.0, '.')
# plt.loglog(x, p, '-')
# plt.title('Empirical vs True Power-law Distribution')
# plt.xlabel('X=x value')
# plt.ylabel('P(X=x)')
# plt.legend(['Empirical', 'True'])
# plt.show()

#P 2.3
xmin = 1.0
P = hist.astype(np.float64)/100000.0
X = bins[1:]
X_ = np.log(X/xmin)
bias = np.ones([X_.shape[0], 1])
Xtilde = np.hstack([X_.reshape([-1,1]), bias])
lp = np.log(P)
A, res, _, _ = np.linalg.lstsq(Xtilde, lp)
alpha_ls = -A[0]
xmin_ls = (alpha_ls - 1)/np.exp(A[1])
mean_err = np.mean(res)
print('Original LS estimate alpha: %r, xmin: %r, with mean error: %r' % (alpha_ls, xmin_ls, mean_err))

xmin = 1.0
P = hist.astype(np.float64)/100000.0
X = bins[40:91]
X_ = np.log(X/xmin)
bias = np.ones([X_.shape[0], 1])
Xtilde = np.hstack([X_.reshape([-1,1]), bias])
lp = np.log(P[40:91])
A, res, _, _ = np.linalg.lstsq(Xtilde, lp)
alpha_ls_ = -A[0]
xmin_ls_ = (alpha_ls_ - 1)/np.exp(A[1])
mean_err = np.mean(res)
print('Improved LS estimate alpha: %r, xmin: %r, with mean error: %r' % (alpha_ls_, xmin_ls_, mean_err))

p_ls = powerlaw(x, alpha_ls, xmin_ls)
p_ls_ = powerlaw(x, alpha_ls_, xmin_ls_)

# plt.figure()
# plt.loglog(bins[1:], hist.astype(np.float64)/100000.0, '.')
# plt.loglog(x, p, '-')
# plt.loglog(x, p_ls, '-.')
# plt.loglog(x, p_ls_, '--')
# plt.title('Empirical vs True vs Estimated Power-law Distribution')
# plt.xlabel('X=x value')
# plt.ylabel('P(X=x)')
# plt.legend(['Empirical', 'True', 'Original LS', 'Improved LS'])
# plt.show()

# P2.4
xmin = 1.0
X_ = sample_powerlaw(2.0, 1.0, 100000)
X = np.round(X_).astype(np.int64)
hist, bins = np.histogram(X, bins=100, range=(1,100))


X_ = np.log(X/xmin)
A = np.sum(X_, axis=0)
alpha_mle = X_.shape[0]/A + 1

print('MLE estimate: %r' % alpha_mle)

p_mle = powerlaw(x, alpha_mle, xmin)

# plt.figure()
# plt.loglog(bins[1:], hist.astype(np.float64)/100000.0, '.')
# plt.loglog(x, p, '-')
# plt.loglog(x, p_mle, '-.')
# plt.title('Empirical vs True vs Estimated Power-law Distribution')
# plt.xlabel('X=x value')
# plt.ylabel('P(X=x)')
# plt.legend(['Empirical', 'True', 'MLE'])
# plt.show()


# P2.5
xmin = 1.0
Alpha_ls = []
Alpha_ls_ = []
Alpha_mle = []
for i in range(100):
	X_ = sample_powerlaw(2.0, 1.0, 100000)
	X = np.round(X_).astype(np.int64)
	hist, bins = np.histogram(X, bins=100, range=(1,100))
	P = hist.astype(np.float64)/100000.0
	X_ = np.log(X/xmin)
	A = np.sum(X_, axis=0)
	Alpha_mle.append(X_.shape[0]/A + 1)
	
	X = bins[1:]
	X_ = np.log(X/xmin)
	bias = np.ones([X_.shape[0], 1])
	Xtilde = np.hstack([X_.reshape([-1,1]), bias])
	lp = np.log(P)
	A, _, _, _ = np.linalg.lstsq(Xtilde, lp)
	Alpha_ls.append(-A[0])

	X = bins[40:91]
	X_ = np.log(X/xmin)
	bias = np.ones([X_.shape[0], 1])
	Xtilde = np.hstack([X_.reshape([-1,1]), bias])
	lp = np.log(P[40:91])
	A, _, _, _ = np.linalg.lstsq(Xtilde, lp)
	Alpha_ls_.append(-A[0])

mean_ls = np.mean(Alpha_ls)
mean_ls_ = np.mean(Alpha_ls_)
mean_mle = np.mean(Alpha_mle)

std_ls = np.std(Alpha_ls)
std_ls_ = np.std(Alpha_ls_)
std_mle = np.std(Alpha_mle)

print('Mean LS: %r, Mean LS improved: %r, Mean MLE: %r' % (mean_ls, mean_ls_, mean_mle))
print('STD LS: %r, STD LS improved: %r, STD MLE: %r' % (std_ls, std_ls_, std_mle))
