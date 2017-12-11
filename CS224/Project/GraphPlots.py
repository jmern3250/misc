import numpy as np 
import pickle 
import snap 
import os
from collections import Counter 
import matplotlib.pyplot as plt 

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

in_size = []
out_size = []
max_node = Graph.GetNodes()-1
for i in range(1000):
	idx = Graph.GetRndNId()
	tree_out = snap.GetBfsTree(Graph, idx, True, False)
	tree_in = snap.GetBfsTree(Graph, idx, False, True)
	out_size.append(tree_out.GetNodes())
	in_size.append(tree_in.GetNodes())
in_array = np.sort(in_size)
out_array = np.sort(out_size)
idx_array = np.arange(0,1,0.001)

print('Maximum SCC Size: %r' % snap.GetMxSccSz(Graph))

plt.figure()
plt.plot(idx_array[100:], in_array[100:], '--', label='In-link Reachability')
plt.plot(idx_array[100:], out_array[100:], '-.', label='Out-link Reachability')
plt.title('Final Graph Reachability')
plt.xlabel('Fraction of Starting Nodes')
plt.ylabel('Number of Reachable Nodes')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.savefig('./Plots/Reach.png')
plt.close()


import pdb; pdb.set_trace()
