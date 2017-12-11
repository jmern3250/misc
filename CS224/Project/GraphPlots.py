import numpy as np 
import pickle 
import snap 
import os
from collections import Counter 
# import matplotlib.pyplot as plt 

import pdb 

# with open('./Pickles/GraphData.p', 'rb') as f: 
#     Data = pickle.load(f)

# Edge_count = Data['Edge_count']
# Node_count = Data['Node_count']
# Diameter = Data['Diameter']
# Clustering = Data['Clustering']
# PowerCoef = Data['PowerCoef'] 
# PowerExp = Data['PowerExp']

# nveA, _, _, _ = np.linalg.lstsq(np.array(Node_count).reshape([-1,1]), np.array(Edge_count).reshape([-1,1]))
# nveA = nveA[0][0]

# plt.figure()
# plt.plot(Node_count, Edge_count, '.')
# plt.plot(Node_count, np.array(Node_count)*nveA)
# plt.title('Edge vs Node Growth')
# plt.xlabel('Number of Nodes')
# plt.ylabel('Number of Edges')
# plt.legend(['Data', 'Line-Fit (slope=1.43)'])
# plt.savefig('./Plots/NvE.png')
# plt.close()

# plt.figure()
# plt.plot(Diameter, '.-')
# plt.title('Diameter Evolution')
# plt.xlabel('Days index')
# plt.ylabel('Approximate Graph Diameter')
# plt.savefig('./Plots/Dia.png')
# plt.close()

# plt.figure()
# plt.plot(Clustering, '.')
# plt.title('Clustering Coefficient Evolution')
# plt.xlabel('Days index')
# plt.ylabel('Average Clustering Coefficient')
# plt.savefig('./Plots/Clst.png')
# plt.close()

# plt.figure()
# plt.plot(PowerExp, '.')
# plt.title('Power Law Exponent Evolution')
# plt.xlabel('Days index')
# plt.ylabel('Alpha Value')
# plt.savefig('./Plots/Alpha.png')
# plt.close()

# plt.figure()
# plt.plot(PowerCoef, '.')
# plt.title('Power Law Coefficient Evolution')
# plt.xlabel('Days index')
# plt.ylabel('Gamma Value')
# plt.savefig('./Plots/Gamma.png')
# plt.close()


GIn = snap.TFIn('./Year1.graph')
Graph = snap.TNGraph.Load(GIn)
OutV = snap.TIntPrV()
snap.GetOutDegCnt(Graph, OutV)
InV = snap.TIntPrV()
snap.GetInDegCnt(Graph, InV)
nOut = len(OutV)
nIn = len(InV)
out_count = []
out_bin = []
in_count = []
in_bin = []
for i in range(nIn):
    in_bin.append(InV[i].GetVal1())
    in_count.append(InV[i].GetVal2()) 
for i in range(nOut):
    out_bin.append(OutV[i].GetVal1())
    out_count.append(OutV[i].GetVal2())

print('Maximum SCC Size: %r' % snap.GetMxSccSz(Graph))

plt.figure()
plt.plot(out_bin, out_count, label='Out Degree')
plt.plot(in_bin, in_count, label='In Degree')
plt.title('Final Graph Degree Distribution')
plt.xlabel('Node Degree')
plt.ylabel('Node Count')
plt.legend()
plt.savefig('./Plots/InNOut.png')
plt.close()
import pdb; pdb.set_trace()
