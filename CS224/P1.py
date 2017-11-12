import numpy as np
import snap 
import csv 
import matplotlib.pyplot as plt 

import pdb 

with open('thresholds.txt', 'rb') as f: 
	data_ = csv.reader(f, delimiter=' ')

	data = []
	for dat in data_: 
		data.append(int(dat[0]))


m = len(data)
Ncum = np.zeros([m,])
n = 0
final = None 
for i, dat in enumerate(data):
	n += dat
	Ncum[i] = n
	if n <= i and final is None: 
		final = n 

print 'A total of %r people will join the riot' % final
N = np.arange(m)

plt.figure()
plt.plot(N, Ncum,'.')
plt.plot(N,N)
plt.grid(True, which='both')
plt.show()


pdb.set_trace()