# python wrap-data.py MSA_NAME gen_code
# python wrap-data.py SanFrancisco 0
 
import setproctitle
setproctitle.setproctitle("gnn-simu-vac@chenlin")

import sys
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite

import dynalearn
import h5py

import pdb

#sys.path.append(os.path.join(os.getcwd(), '../gt-generator'))
import constants

###############################################################################
# Constants

epic_data_root = '/data/chenlin/COVID-19/Data'
gt_result_root = os.getcwd()

###############################################################################
# Main variable settings

MSA_NAME = sys.argv[1]; print('MSA_NAME: ',MSA_NAME) #MSA_NAME = 'SanFrancisco'
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME] #MSA_NAME_FULL = 'San_Francisco_Oakland_Hayward_CA'

gen_code = int(sys.argv[2])
if(gen_code==0):
    print('Generate data: synthetic CBG homogeneous network.')
elif(gen_code==1):
    print('Generate data: CBG-POI homogeneous network. Bipartite.')
elif(gen_code==2):
    print('Generate data: CBG-POI homogeneous network, with 5000 edges, toy. Bipartite.')
elif(gen_code==3):
    print('Generate data: Toy bipartite.')
else:
    print('Invalid gen_code. Please check.')
    pdb.set_trace()

###############################################################################
# Load and wrap data

# Load epidemic result data
NUM_SEEDS = 60
cases_cbg_no_vaccination = np.load(os.path.join(gt_result_root, 'cases_cbg_no_vaccination_%s_%sseeds.npy' % (MSA_NAME, NUM_SEEDS)))
print('shape of cbg daily cases: ', cases_cbg_no_vaccination.shape) #(num_days, num_cbgs)(63, 2943)
num_days = cases_cbg_no_vaccination.shape[0]; print('num_days: ', num_days)

# Load POI-CBG visiting matrices
#MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[args.MSA_NAME] #'San_Francisco_Oakland_Hayward_CA'
f = open(os.path.join(epic_data_root, MSA_NAME, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()

# Obtain CBG populations
# Load CBG ids for the MSA
cbg_ids_msa = pd.read_csv(os.path.join(epic_data_root,MSA_NAME,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
# Load SafeGraph data to obtain CBG sizes (i.e., populations)
filepath = os.path.join(epic_data_root,"safegraph_open_census_data/data/cbg_b01.csv")
cbg_agesex = pd.read_csv(filepath)
# Extract CBGs belonging to the MSA - https://covid-mobility.stanford.edu//datasets/
cbg_age_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
# Deal with NaN values
cbg_age_msa.fillna(0,inplace=True)
# Deal with CBGs with 0 populations
cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
# Obtain cbg sizes (populations)
cbg_sizes = cbg_age_msa['Sum'].values
cbg_sizes = np.array(cbg_sizes,dtype='int32')
print('Total population: ',np.sum(cbg_sizes))
del cbg_agesex
del cbg_age_msa
del cbg_ids_msa

# Get one timestep's data for test 
single_array = poi_cbg_visits_list[0].todense()
num_pois = single_array.shape[0]; print('num_pois: ', num_pois)
num_cbgs = single_array.shape[1]; print('num_cbgs: ', num_cbgs)
num_nodes = num_pois + num_cbgs; print('num_nodes (cbg+poi): ', num_nodes)
#num_days = int(len(poi_cbg_visits_list)/24) #1512h/24h


# Average all arrays
num_hours = len(poi_cbg_visits_list); print('num_hours: ', num_hours)
avg_array_path = os.path.join(gt_result_root, 'avg_array_%s.npy' % MSA_NAME)
if(os.path.exists(avg_array_path)):
    avg_array = np.load(avg_array_path)
else:
    avg_array = np.zeros(poi_cbg_visits_list[0].shape)
    for i in range(num_hours):
        if(i%50==0):print(i)
        avg_array += poi_cbg_visits_list[i]
    avg_array /= num_hours
    np.save(avg_array_path, avg_array)


# Network attributes
#edge_list = np.append(np.reshape(np.nonzero(single_array)[0], (-1,1)),np.reshape(np.nonzero(single_array)[1], (-1,1)),axis=1)
if(gen_code==0):
    num_nodes = num_cbgs
    node_list = np.arange(num_nodes)
    #node_attr = np.ones(num_nodes) #test
    #edge_list = np.append(np.random.permutation(np.reshape(np.nonzero(single_array)[1], (-1,1))),
    #                      np.reshape(np.nonzero(single_array)[1], (-1,1)),axis=1) #test
    #edge_list = np.append(np.random.permutation(np.reshape(np.nonzero(avg_array)[1], (-1,1))),
    #                      np.reshape(np.nonzero(avg_array)[1], (-1,1)),axis=1) #20220101
    edge_list = np.append(np.reshape(np.random.permutation(np.arange(num_nodes)), (-1,1)),
                          np.reshape(np.arange(num_nodes), (-1,1)),axis=1) #20220103
    
elif(gen_code==1):
    cases_cbg_no_vaccination = np.concatenate((cases_cbg_no_vaccination, np.zeros((num_days,num_pois))), axis=1) #Set all poi ground truth as 0
    node_list = np.arange(num_nodes)
    #node_attr = np.ones(num_nodes) #test
    edge_list = np.append(np.reshape(np.nonzero(avg_array)[0]+num_cbgs, (-1,1)), #POI序号整体后移
                          np.reshape(np.nonzero(avg_array)[1], (-1,1)),axis=1)
elif(gen_code==2):
    cases_cbg_no_vaccination = np.concatenate((cases_cbg_no_vaccination, np.zeros((num_days,num_pois))), axis=1) #Set all poi ground truth as 0
    node_list = np.arange(num_nodes)
    #node_attr = np.ones(num_nodes) #test
    edge_list = np.append(np.reshape(np.nonzero(avg_array)[0]+num_cbgs, (-1,1))[:5000], #POI序号整体后移
                          np.reshape(np.nonzero(avg_array)[1], (-1,1))[:5000],axis=1)
elif(gen_code==3):
    num_nodes = 52
    num_cbgs = 20
    #pdb.set_trace()
    cases_cbg_no_vaccination = cases_cbg_no_vaccination[:, :num_nodes]
    node_list = np.arange(num_nodes)
    random_bipartite = bipartite.random_graph(num_cbgs, num_nodes-num_cbgs,  0.4) #https://networkx.org/documentation/stable/reference/algorithms/bipartite.html
    cbg_set = {n for n, d in random_bipartite.nodes(data=True) if d["bipartite"] == 0}
    poi_set = list(set(random_bipartite) - cbg_set)
    cbg_list = list(cbg_set)
    poi_list = list(poi_set)
    edges = [edge for edge in random_bipartite.edges]
    print('Number of edges: ', len(edges))
    edge_cbg_end = np.array([n1 for n1,n2 in edges])
    edge_poi_end = np.array([n2 for n1,n2 in edges])
    edge_list = np.append(np.reshape(edge_cbg_end,(-1,1)), np.reshape(edge_poi_end,(-1,1)), axis=1)

    isolated_nodes = set(random_bipartite.nodes())
    for (u, v) in random_bipartite.edges():
        isolated_nodes -= {u}
        isolated_nodes -= {v}
    print('Isolated nodes: ', isolated_nodes) #should be none
    #pdb.set_trace()
    

# test attr data: 保证每个节点、每条边都有attr
#node_attr = np.ones(num_nodes) #test
node_attr = np.append(cbg_sizes.copy(), np.random.random(num_nodes-num_cbgs)*10) #0~10均匀抽样
print('mean and var of node_attr: ', np.mean(node_attr), np.var(node_attr)) 
#edge_attr = np.ones(len(edge_list)) #test #不行，edge_attr标准差为0，后面归一化时出现除以0的错误，导致出现nan #20220104
num_edges = len(edge_list)
edge_attr = np.zeros(num_edges, dtype='float32')
if(gen_code==1 or gen_code==2):
    for i in range(num_edges):
        edge_attr[i] = avg_array[edge_list[i][0]-num_cbgs, edge_list[i][1]]
elif(gen_code==3):
    for i in range(num_edges):
        edge_attr[i] = avg_array[edge_list[i][0], edge_list[i][1]] #20220104
print('mean and var of edge_attr: ', np.mean(edge_attr), np.var(edge_attr))
pdb.set_trace()
'''
# real attr data #20220103
node_attr = cbg_sizes.copy() #20220103
num_edges = len(edge_list)
edge_attr = np.zeros(num_edges, dtype='float32')
for i in range(num_edges):
    edge_attr[i] = avg_array[edge_list[i][0], edge_list[i][1]]
'''
pdb.set_trace()


# Wrap in hdf5 format
#data = h5py.File('data_%s.h5' % (MSA_NAME), 'w')
data = h5py.File('data_%s_gencode%s.h5' % (MSA_NAME, str(gen_code)), 'w')
# Epidemic data
data.create_dataset('timeseries', data=cases_cbg_no_vaccination)
# Mobility network
f = data.create_group('networks')
f.create_dataset('node_list', data=node_list)
f.create_dataset('edge_list', data=edge_list)
node_data = f.create_group('node_attr')
node_data.create_dataset('population', data=node_attr)
edge_data = f.create_group('edge_attr')
edge_data.create_dataset('weight',data=edge_attr)
networks = f

lag = 5
lagstep = 1
num_states = 1

# Wrap data
'''
#dataset = h5py.File(os.path.join(path_to_covid, "spain-covid19cases.h5"), "r")
print('path_to_covid: ', path_to_covid)
dataset = h5py.File(os.path.join(os.path.abspath('../..'),path_to_covid, "spain-covid19-dataset.h5"), "r") #20211221
num_states = 1

X = dataset["weighted-multiplex/data/timeseries/d0"][...] #20211221
Y = dataset["weighted-multiplex/data/timeseries/d0"][...] #20211221
networks = dataset["weighted-multiplex/data/networks/d0"]
'''

data = {
    "inputs": dynalearn.datasets.DataCollection(name="inputs"),
    "targets": dynalearn.datasets.DataCollection(name="targets"),
    "networks": dynalearn.datasets.DataCollection(name="networks"),
}
inputs = np.zeros((num_days - (lag - 1) * lagstep, num_nodes, num_states, lag))
targets = np.zeros((num_days - (lag - 1) * lagstep, num_nodes, num_states))
X = cases_cbg_no_vaccination
Y = cases_cbg_no_vaccination
for t in range(inputs.shape[0]):
    x = X[t : t + lag * lagstep : lagstep]
    y = Y[t + lag * lagstep - 1]
    x = x.reshape(*x.shape, 1)
    y = y.reshape(*y.shape, 1)
    x = np.transpose(x, (1, 2, 0))
    inputs[t] = x
    targets[t] = y

data["inputs"].add(dynalearn.datasets.StateData(data=inputs))
data["targets"].add(dynalearn.datasets.StateData(data=targets))
data["networks"].add(dynalearn.datasets.NetworkData(data=networks))

pdb.set_trace()
