import numpy as np 
import pickle 
import snap 
import os
from collections import Counter 
import matplotlib.pyplot as plt 

import pdb 

with open('./Pickles/GraphData.p', 'rb') as f: 
    Data = pickle.load(f)

Edge_count = Data['Edge_count']
Node_count = Data['Node_count']
Diameter = Data['Diameter']
Clustering = Data['Clustering']
PowerCoef = Data['PowerCoef'] 
PowerExp = Data['PowerExp']

# pdb.set_trace()
nveA, _, _, _ = np.linalg.lstsq(np.array(Node_count).reshape([-1,1]), np.array(Edge_count).reshape([-1,1]))
nveA = nveA[0][0]

plt.figure()
plt.plot(Node_count, Edge_count, '.')
plt.plot(Node_count, np.array(Node_count)*nveA)
plt.title('Edge vs Node Growth')
plt.xlabel('Number of Nodes')
plt.ylabel('Number of Edges')
plt.legend(['Data', 'Line-Fit (slope=1.43)'])
plt.savefig('./Plots/NvE.png')
plt.close()

plt.figure()
plt.plot(Diameter, '.-')
plt.title('Diameter Evolution')
plt.xlabel('Days index')
plt.ylabel('Approximate Graph Diameter')
plt.savefig('./Plots/Dia.png')
plt.close()

plt.figure()
plt.plot(Clustering, '.')
plt.title('Clustering Coefficient Evolution')
plt.xlabel('Days index')
plt.ylabel('Average Clustering Coefficient')
plt.savefig('./Plots/Clst.png')
plt.close()

plt.figure()
plt.plot(PowerExp, '.')
plt.title('Power Law Exponent Evolution')
plt.xlabel('Days index')
plt.ylabel('Alpha Value')
plt.savefig('./Plots/Alpha.png')
plt.close()

plt.figure()
plt.plot(PowerCoef, '.')
plt.title('Power Law Coefficient Evolution')
plt.xlabel('Days index')
plt.ylabel('Gamma Value')
plt.savefig('./Plots/Gamma.png')
plt.close()


# GOut = snap.TFOut('Year1.graph')
# Graph.Save(GOut)
# GOut.Flush()