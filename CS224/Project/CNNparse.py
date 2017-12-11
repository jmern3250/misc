import numpy as np 
import pickle 
import os 
import csv 
from collections import Counter 
import scipy 
from scipy import sparse 
from scipy.sparse import linalg 
import matplotlib.pyplot as plt 

import pdb
import time

''' 
This file imports the edge list created from Bitcoin transaction ledger through 2013.12.28 
and parses the data into a series of daily snapshots represented as the 500 X 500 SVD-PCA reduction 
of the adjacency matrix. 
'''

START = 1293840000 # 1 January, 2011 
ACTIVE = 10 #Number of days of inactivity before node is removed from graph 

if os.path.exists('./Pickles/rcvrs.p'):
    with open('./Pickles/rcvrs.p', 'rb') as f:
        rcvrs_dict = np.load(f)
    with open('./Pickles/sndrs.p', 'rb') as f:
        sndrs_dict = np.load(f)
    with open('./Pickles/idx.p', 'rb') as f:
        idx = np.load(f)
    with open('./Pickles/days.p', 'rb') as f:
        days = np.load(f)
else:
    with open('./Data/txedge.txt', 'rt') as f: 
        datareader = csv.reader(f, delimiter='\t')
        day0 = None
        sndrs = []
        rcvrs = []
        days = []
        idx = []
        i = 0
        day_ = 0
        sndrs_dict = {day_:[]}
        rcvrs_dict = {day_:[]}
        for row in datareader: 
            # pdb.set_trace()
            seconds = int(row[3])
            day = int((seconds-START)/(60*60*24))
            # if day0 is None: 
            #     day0 = day
            # day -= day0
            if day < 0:
                pass 
            else: 
                if day != day_:
                    sndrs_dict[day] = []
                    rcvrs_dict[day] = []
                    day_ = day
                days.append(day)
                # sndrs.append(int(row[1]))
                # rcvrs.append(int(row[]))
                idx.append(int(row[1]+row[2]))
                sndrs_dict[day].append(int(row[1]))
                rcvrs_dict[day].append(int(row[2]))

            i += 1
            if i%1000000 == 0:
                print('Row %r parsed' % i)

    # ids = rcvrs + sndrs
    idx = np.unique(idx)
    del rcvrs, sndrs 

    with open('./Pickles/rcvrs.p', 'wb') as f: 
        pickle.dump(rcvrs_dict, f)

    with open('./Pickles/sndrs.p', 'wb') as f: 
        pickle.dump(sndrs_dict, f)

    with open('./Pickles/idx.p', 'wb') as f: 
        pickle.dump(idx, f)

    with open('./Pickles/days.p', 'wb') as f: 
        pickle.dump(days, f)

n = max(days) + 1
R = 100 # Target "image" size

# data_set = np.zeros([n, R, R]) # day, sender, recipeient 
# data_set = {}
U_set = {}
S_set = {}
A = set() # Active nodes
I = set() # Inactive nodes
E = set() # Edges
C = Counter() 

year = 0 
for day, sndrs in sndrs_dict.items():
    A_ = set() #Nodes active on current day 
    rcvrs = rcvrs_dict[day]
    m = len(sndrs)

    for i in range(m): 
        e = []
        sndr = sndrs[i]
        rcvr = rcvrs[i]
        A_.add(sndr)
        e.append(sndr) 
        A_.add(rcvr)
        e.append(rcvr)
        E.add(tuple(e))

    nA = A - A_ #Previously active nodes not active today 

    C.update(nA)
    I_ = set() #Nodes that become inactive today 
    for key, value in C.items():
        if value >= ACTIVE: 
            # pdb.set_trace()
            I_.add(key)

    for node in A_: 
        C[node] = 0
    A = A.union(A_)
    A = A - I_ 
    I = I.union(I_)
    # I = I - A_ 

    E_ = set() # Edges to be removed 
    for edge in E: 
        if edge[0] in I or edge[1] in I:
            E_.add(edge)

    E = E - E_ 
    N = sorted(list(A)) # All nodes in graph 
    m = len(N)
    idx = {}
    for i, node in enumerate(N): 
        idx[node] = i 

    vals = [1]*len(E)
    rows = [idx[edge[0]] for edge in E]
    cols = [idx[edge[1]] for edge in E]
    # M = np.zeros([m,m])
    # M[rows,cols] = 1.0

    # pdb.set_trace()
    dat = (vals, (rows, cols))

    M = scipy.sparse.coo_matrix(dat, shape=[m,m],dtype='float64')
    U, S, V = scipy.sparse.linalg.svds(M, k=R, tol=1e-2)
    # pdb.set_trace()
    U_set[day] = U
    S_set[day] = S
    # if day%20 == 0:
    #     pdb.set_trace()

    print('Day %r done with %r nodes in graph' % (day, m))
    if (day+1)%365 == 0:
        U_name = './Pickles/Udata' + str(year) +'.p'
        with open(U_name, 'wb') as f: 
            pickle.dump(U_set, f)
        S_name = './Pickles/Sdata' + str(year) +'.p'
        with open(S_name, 'wb') as f: 
            pickle.dump(S_set, f)
        print('Year %r saved' % year)
        year += 1 
        U_set = {}
        S_set = {}


with open('./Pickles/CNNdata_All.p', 'wb') as f: 
    pickle.dump(data_set, f)

pdb.set_trace()


# for day in days
