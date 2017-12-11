import numpy as np 
import pickle 
import snap 
import os
from collections import Counter 
import matplotlib.pyplot as plt 

import pdb 

def fitdeg(X,Y):
    m = len(X)
    lY = np.log(Y)
    lX = np.log(X).reshape([-1,1]) 
    lX_ = np.hstack([lX, np.ones([m,1])])
    A, _, _, _ = np.linalg.lstsq(lX_, lY)
    k = -A[0]
    a = np.exp(A[1])
    return k, a

START = 1293840000 # 1 January, 2011 
ACTIVE = 10 #Number of days of inactivity before node is removed from graph 

if os.path.exists('./Pickles/rcvrs.p'):
    with open('./Pickles/rcvrs2.p', 'rb') as f:
        rcvrs_dict = pickle.load(f)
    with open('./Pickles/sndrs2.p', 'rb') as f:
        sndrs_dict = pickle.load(f)
    # with open('./Pickles/idx2.p', 'rb') as f:
    #     idx = pickle.load(f)
    # with open('./Pickles/days2.p', 'rb') as f:
    #     days = pickle.load(f)
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


A = set() # Active nodes
I = set() # Inactive nodes
E = set() # Edges
C = Counter() 

Edge_count = []
Node_count = []
Diameter = []
Clustering = []
PowerCoef = []
PowerExp = []

total_days = len(sndrs_dict)

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

    

    print('Day %r of %r analyzed' % (day+1, total_days))
    if (day+10)%10 == 0:
        nn = len(A)
        ne = len(E)
        nn_dia = int(nn*0.1) # Number of nodes for diameter BFS 

        Graph = snap.TNGraph.New(nn,ne)
        for node in A: 
            Graph.AddNode(node)
        for edge in E: 
            src = edge[0]
            dst = edge[1]
            Graph.AddEdge(src, dst)

        snap.DelZeroDegNodes(Graph) 

        dia = snap.GetBfsEffDiam(Graph, nn_dia, True)
        cluster = snap.GetClustCf(Graph)

        
        Clustering.append(cluster)
        Node_count.append(nn)
        Edge_count.append(ne)
        Diameter.append(dia)
        DegToCntV = snap.TIntPrV()
        snap.GetDegCnt(Graph, DegToCntV)
        ndeg = len(DegToCntV)
        deg_count = []
        deg_bin = []
        for i in range(ndeg):
            deg_bin.append(DegToCntV[i].GetVal1())
            deg_count.append(DegToCntV[i].GetVal2()) 
        ndeg = int(ndeg*0.9)
        k, a = fitdeg(deg_bin[:ndeg],deg_count[:ndeg]) 
        PowerCoef.append(a)
        PowerExp.append(k)

    if day >= 900:
        break 

pdb.set_trace()
Data = {}
Data['Edge_count'] = Edge_count
Data['Node_count'] = Node_count
Data['Diameter'] = Diameter
Data['Clustering'] = Clustering
Data['PowerCoef'] = PowerCoef 
Data['PowerExp'] = PowerExp

with open('./Pickles/GraphData.p', 'wb') as f: 
    pickle.dump(Data, f)

GOut = snap.TFOut('Year1.graph')
Graph.Save(GOut)
GOut.Flush()