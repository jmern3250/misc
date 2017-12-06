import numpy as np
import snap 
import csv 
import matplotlib.pyplot as plt 
import scipy.stats
import pickle 

import pdb 

IMDB = snap.LoadEdgeList(snap.PUNGraph, './imdb_actor_edges.tsv')
pdb.set_trace()
n_IMDB = IMDB.GetNodes()
IMDB_nodes = {}
for node in IMDB.Nodes(): 
	nid = node.GetId()
	IMDB_nodes[nid] = 0

BETA = 0.05 
DELTA = 0.5

IMDB_infected = []
for sim in range(100):
	IMDB_nodes = dict.fromkeys(IMDB_nodes.keys(), 0)
	patient0 = np.random.choice(IMDB_nodes.keys())
	IMDB_nodes[patient0] = 1
	done = False
	while not done:
		IMDB_nodes_ = IMDB_nodes.copy()
		for node, state in IMDB_nodes.items():
			if state == 1: 
				state_ = np.random.choice([-1,1])
				IMDB_nodes_[node] = state_
		for node in IMDB.Nodes(): 
			nid = node.GetId()
			deg = node.GetDeg()
			state = IMDB_nodes[nid]
			if state == 1:
				for i in range(deg):
					nbr = node.GetNbrNId(i)
					nbr_state = IMDB_nodes[nbr]
					if nbr_state == 0:
						infect = int(np.random.uniform(0.0,1.0) <= BETA)
						IMDB_nodes_[nbr] = infect

		IMDB_nodes = IMDB_nodes_.copy()
		inf_array = np.array([k for k in IMDB_nodes.values()])
		infected = np.sum(inf_array == 1)
		if infected == 0:
			recovered = np.sum(inf_array == -1)
			done = True 
			IMDB_infected.append(float(recovered)/float(n_IMDB))
	print('IMDB Simulation %r done' % (sim+1))

# # # Erdos Renyi Graph
ER = snap.LoadEdgeList(snap.PUNGraph, './SIR_erdos_renyi.txt')
n_ER = ER.GetNodes()
# ER_nodes = {}
# for node in ER.Nodes(): 
# 	nid = node.GetId()
# 	ER_nodes[nid] = 0

# BETA = 0.05 
# DELTA = 0.5

# ER_infected = []
# for sim in range(100):
# 	ER_nodes = d = dict.fromkeys(ER_nodes.keys(), 0)
# 	patient0 = np.random.choice(ER_nodes.keys())
# 	ER_nodes[patient0] = 1
# 	done = False
# 	while not done:
# 		ER_nodes_ = ER_nodes.copy()
# 		for node, state in ER_nodes.items():
# 			if state == 1: 
# 				state_ = np.random.choice([-1,1])
# 				ER_nodes_[node] = state_
# 		for node in ER.Nodes(): 
# 			nid = node.GetId()
# 			deg = node.GetDeg()
# 			state = ER_nodes[nid]
# 			if state == 1:
# 				for i in range(deg):
# 					nbr = node.GetNbrNId(i)
# 					nbr_state = ER_nodes[nbr]
# 					if nbr_state == 0:
# 						infect = int(np.random.uniform(0.0,1.0) <= BETA)
# 						ER_nodes_[nbr] = infect
						
# 		ER_nodes = ER_nodes_.copy()
# 		inf_array = np.array([k for k in ER_nodes.values()])
# 		infected = np.sum(inf_array == 1)
# 		if infected == 0:
# 			recovered = np.sum(inf_array == -1)
# 			done = True 
# 			ER_infected.append(float(recovered)/float(n_ER))
# 	print('ER Simulation %r done' % (sim+1))

# # #Preferential Attachment 

PA = snap.LoadEdgeList(snap.PUNGraph, './SIR_preferential_attachment.txt')
n_PA = PA.GetNodes()
# PA_nodes = {}
# for node in PA.Nodes(): 
# 	nid = node.GetId()
# 	PA_nodes[nid] = 0

# BETA = 0.05 
# DELTA = 0.5

# PA_infected = []
# for sim in range(100):
# 	PA_nodes = d = dict.fromkeys(PA_nodes.keys(), 0)
# 	patient0 = np.random.choice(PA_nodes.keys())
# 	PA_nodes[patient0] = 1
# 	done = False
# 	while not done:
# 		PA_nodes_ = PA_nodes.copy()
# 		for node, state in PA_nodes.items():
# 			if state == 1: 
# 				state_ = np.random.choice([-1,1])
# 				PA_nodes_[node] = state_
# 		for node in PA.Nodes(): 
# 			nid = node.GetId()
# 			deg = node.GetDeg()
# 			state = PA_nodes[nid]
# 			if state == 1:
# 				for i in range(deg):
# 					nbr = node.GetNbrNId(i)
# 					nbr_state = PA_nodes[nbr]
# 					if nbr_state == 0:
# 						infect = int(np.random.uniform(0.0,1.0) <= BETA)
# 						PA_nodes_[nbr] = infect
						
# 		PA_nodes = PA_nodes_.copy()
# 		inf_array = np.array([k for k in PA_nodes.values()])
# 		infected = np.sum(inf_array == 1)
# 		if infected == 0:
# 			recovered = np.sum(inf_array == -1)
# 			done = True 
# 			PA_infected.append(float(recovered)/float(n_PA))
# 	print('PA Simulation %r done' % (sim+1))

# with open('./P3_1.p', 'wb') as f: 
# 		pickle.dump({'IMDB':IMDB_infected, 'ER':ER_infected, 'PA':PA_infected},f)
# print('Sim Results Saved')

with open('./P3_1.p', 'rb') as f: 
	data = pickle.load(f)

IMDB_infected = data['IMDB']
ER_infected = data['ER']
PA_infected = data['PA']
epidemic_IMDB = np.array(IMDB_infected) >= 0.5
epidemic_ER = np.array(ER_infected) >= 0.5
epidemic_PA = np.array(PA_infected) >= 0.5
pdb.set_trace()
n_epi_IMDB = np.sum(epidemic_IMDB)
n_epi_ER = np.sum(epidemic_ER)
n_epi_PA = np.sum(epidemic_PA)

print('Proportion of simulations causing epidemic IMDB: %r' %(n_epi_IMDB/100.0))
print('Proportion of simulations causing epidemic Erdos Renyi: %r' %(n_epi_ER/100.0))
print('Proportion of simulations causing epidemic Preferential Attachment: %r' %(n_epi_PA/100.0))
# pdb.set_trace()
#IMDB vs ER
chi1, p1, _, _ = scipy.stats.chi2_contingency([[n_epi_IMDB, 100-n_epi_IMDB],[n_epi_ER, 100-n_epi_ER]])
#IMDB vs PA
chi2, p2, _, _ = scipy.stats.chi2_contingency([[n_epi_IMDB, 100-n_epi_IMDB],[n_epi_PA, 100-n_epi_PA]])
#PA vs ER
chi3, p3, _, _ = scipy.stats.chi2_contingency([[n_epi_PA, 100-n_epi_PA],[n_epi_ER, 100-n_epi_ER]])
pdb.set_trace()