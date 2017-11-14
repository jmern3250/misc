import numpy as np
import snap 
import csv 
import matplotlib.pyplot as plt 
import scipy.stats
import pickle 

import pdb 

IMDB = snap.LoadEdgeList(snap.PUNGraph, './imdb_actor_edges.tsv')
n_IMDB = IMDB.GetNodes()
# IMDB_nodes = {}
# for node in IMDB.Nodes(): 
# 	nid = node.GetId()
# 	IMDB_nodes[nid] = 0
# IMDBDegs = snap.TIntPrV()
# snap.GetNodeInDegV(IMDB, IMDBDegs)
# IMDBDegs.Sort()
# deg_list = []
# for i in range(n_IMDB):
# 	deg_list.append(IMDBDegs[i].GetVal2())

# patient0 = []
# for i in range(10):
# 	nidx = np.argmax(deg_list)
# 	nid = IMDBDegs[nidx].GetVal1()
# 	patient0.append(nid)
# 	deg_list[nidx] = 0.0
# BETA = 0.05 
# DELTA = 0.5
# IMDB_infected = []
# for sim in range(100):
# 	IMDB_nodes = d = dict.fromkeys(IMDB_nodes.keys(), 0)
# 	for patient in patient0:
# 		IMDB_nodes[patient] = 1
# 	done = False
# 	while not done:
# 		IMDB_nodes_ = IMDB_nodes.copy()
# 		for node, state in IMDB_nodes.items():
# 			if state == 1: 
# 				state_ = np.random.choice([-1,1])
# 				IMDB_nodes_[node] = state_
# 		for node in IMDB.Nodes(): 
# 			nid = node.GetId()
# 			deg = node.GetDeg()
# 			state = IMDB_nodes[nid]
# 			if state == 1:
# 				for i in range(deg):
# 					nbr = node.GetNbrNId(i)
# 					nbr_state = IMDB_nodes[nbr]
# 					if nbr_state == 0:
# 						infect = int(np.random.uniform(0.0,1.0) <= BETA)
# 						IMDB_nodes_[nbr] = infect

# 		IMDB_nodes = IMDB_nodes_.copy()
# 		inf_array = np.array([k for k in IMDB_nodes.values()])
# 		infected = np.sum(inf_array == 1)
# 		if infected == 0:
# 			recovered = np.sum(inf_array == -1)
# 			done = True 
# 			IMDB_infected.append(float(recovered)/float(n_IMDB))
# 	print('IMDB Simulation %r done' % (sim+1))

# # Erdos Renyi Graph
ER = snap.LoadEdgeList(snap.PUNGraph, './SIR_erdos_renyi.txt')
n_ER = ER.GetNodes()
# ER_nodes = {}
# for node in ER.Nodes(): 
# 	nid = node.GetId()
# 	ER_nodes[nid] = 0

# BETA = 0.05 
# DELTA = 0.5
# ERDegs = snap.TIntPrV()
# snap.GetNodeInDegV(ER, ERDegs)
# ERDegs.Sort()
# deg_list = []
# for i in range(n_ER):
# 	deg_list.append(ERDegs[i].GetVal2())

# patient0 = []
# for i in range(10):
# 	nidx = np.argmax(deg_list)
# 	nid = ERDegs[nidx].GetVal1()
# 	patient0.append(nid)
# 	deg_list[nidx] = 0.0
# ER_infected = []
# for sim in range(100):
# 	ER_nodes = d = dict.fromkeys(ER_nodes.keys(), 0)
# 	for patient in patient0:
# 		ER_nodes[patient] = 1
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

# # Preferential Attachment 

PA = snap.LoadEdgeList(snap.PUNGraph, './SIR_preferential_attachment.txt')
n_PA = PA.GetNodes()
# PA_nodes = {}
# for node in PA.Nodes(): 
# 	nid = node.GetId()
# 	PA_nodes[nid] = 0

# BETA = 0.05 
# DELTA = 0.5
# PADegs = snap.TIntPrV()
# snap.GetNodeInDegV(PA, PADegs)
# PADegs.Sort()
# deg_list = []
# for i in range(n_PA):
# 	deg_list.append(PADegs[i].GetVal2())

# patient0 = []
# for i in range(10):
# 	nidx = np.argmax(deg_list)
# 	nid = PADegs[nidx].GetVal1()
# 	patient0.append(nid)
# 	deg_list[nidx] = 0.0
# PA_infected = []
# for sim in range(100):
# 	PA_nodes = d = dict.fromkeys(PA_nodes.keys(), 0)
# 	for patient in patient0:
# 		PA_nodes[patient] = 1
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

# # IMDB Random
# BETA = 0.05 
# DELTA = 0.5

# IMDB_infected_rnd = []
# for sim in range(100):
# 	IMDB_nodes = d = dict.fromkeys(IMDB_nodes.keys(), 0)
# 	patient0 = np.random.choice(IMDB_nodes.keys(), size=10)
# 	for patient in patient0:
# 		IMDB_nodes[patient] = 1
# 	done = False
# 	while not done:
# 		IMDB_nodes_ = IMDB_nodes.copy()
# 		for node, state in IMDB_nodes.items():
# 			if state == 1: 
# 				state_ = np.random.choice([-1,1])
# 				IMDB_nodes_[node] = state_
# 		for node in IMDB.Nodes(): 
# 			nid = node.GetId()
# 			deg = node.GetDeg()
# 			state = IMDB_nodes[nid]
# 			if state == 1:
# 				for i in range(deg):
# 					nbr = node.GetNbrNId(i)
# 					nbr_state = IMDB_nodes[nbr]
# 					if nbr_state == 0:
# 						infect = int(np.random.uniform(0.0,1.0) <= BETA)
# 						IMDB_nodes_[nbr] = infect

# 		IMDB_nodes = IMDB_nodes_.copy()
# 		inf_array = np.array([k for k in IMDB_nodes.values()])
# 		infected = np.sum(inf_array == 1)
# 		if infected == 0:
# 			recovered = np.sum(inf_array == -1)
# 			done = True 
# 			IMDB_infected_rnd.append(float(recovered)/float(n_IMDB))
# 	print('IMDB Simulation %r done' % (sim+1))

# #Erdos Renyi Random
# BETA = 0.05 
# DELTA = 0.5

# ER_infected_rnd = []
# for sim in range(100):
# 	ER_nodes = d = dict.fromkeys(ER_nodes.keys(), 0)
# 	patient0 = np.random.choice(ER_nodes.keys(), size=10)
# 	for patient in patient0:
# 		ER_nodes[patient] = 1
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
# 			ER_infected_rnd.append(float(recovered)/float(n_ER))
# 	print('ER Simulation %r done' % (sim+1))

# #Preferential Attachment Random
# BETA = 0.05 
# DELTA = 0.5

# PA_infected_rnd = []
# for sim in range(100):
# 	PA_nodes = d = dict.fromkeys(PA_nodes.keys(), 0)
# 	patient0 = np.random.choice(PA_nodes.keys(), size=10)
# 	for patient in patient0:
# 		PA_nodes[patient] = 1
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
# 			PA_infected_rnd.append(float(recovered)/float(n_PA))
# 	print('PA Simulation %r done' % (sim+1))

# with open('./P3_3.p', 'wb') as f: 
# 		pickle.dump({'IMDB':IMDB_infected, 'ER':ER_infected, 'PA':PA_infected, 'IMDB_rnd': IMDB_infected_rnd, 'ER_rnd': ER_infected_rnd, 'PA_rnd': PA_infected_rnd},f)
# print('Sim Results Saved')

with open('./P3_3.p', 'rb') as f: 
	data = pickle.load(f)

IMDB_infected = data['IMDB']
ER_infected = data['ER']
PA_infected = data['PA']
IMDB_infected_rnd = data['IMDB_rnd']
ER_infected_rnd = data['ER_rnd']
PA_infected_rnd = data['PA_rnd']
pdb.set_trace()
epidemic_IMDB = np.array(IMDB_infected) >= 0.5
epidemic_ER = np.array(ER_infected) >= 0.5
epidemic_PA = np.array(PA_infected) >= 0.5
epidemic_IMDB_rnd = np.array(IMDB_infected_rnd) >= 0.5
epidemic_ER_rnd = np.array(ER_infected_rnd) >= 0.5
epidemic_PA_rnd = np.array(PA_infected_rnd) >= 0.5

n_epi_IMDB = np.sum(epidemic_IMDB)
n_epi_ER = np.sum(epidemic_ER)
n_epi_PA = np.sum(epidemic_PA)
n_epi_IMDB_rnd = np.sum(epidemic_IMDB_rnd)
n_epi_ER_rnd = np.sum(epidemic_ER_rnd)
n_epi_PA_rnd = np.sum(epidemic_PA_rnd)

print('Proportion of simulations causing epidemic IMDB: %r' %(n_epi_IMDB/100.0))
print('Proportion of simulations causing epidemic Erdos Renyi: %r' %(n_epi_ER/100.0))
print('Proportion of simulations causing epidemic Preferential Attachment: %r' %(n_epi_PA/100.0))
print('Proportion of simulations causing epidemic IMDB random: %r' %(n_epi_IMDB_rnd/100.0))
print('Proportion of simulations causing epidemic Erdos Renyi random: %r' %(n_epi_ER_rnd/100.0))
print('Proportion of simulations causing epidemic Preferential Attachment random: %r' %(n_epi_PA_rnd/100.0))
