# python pairwise_comparator.py

import setproctitle
setproctitle.setproctitle("gnn-vac@chenlin")

import socket
import argparse
import os
import sys
import numpy as np
import pandas as pd
import datetime
import time
import random
import networkx as nx
import igraph as ig
import torch
import torch.optim as optim
from torch import nn

sys.path.append(os.path.join(os.getcwd(), '../gt-generator'))
import constants
from utils import *
from config import *
from new_models import get_model

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--msa_name', default='SanFrancisco',
                    help='MSA name.')
parser.add_argument('--vaccination_ratio', type=float, default=0.02,
                    help='Vaccination ratio (w.r.t. total population).')                          
parser.add_argument('--NN', type=int, default=5, 
                    help='Num of counties to receive vaccines.')                     
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--num_pairs', type=int, default=500,
                    help='Number of randomly constructed strategy pairs.')                    
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')                                        
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--gt_root', default=os.path.abspath(os.path.join(os.pardir,'data/safegraph')),
                    help='Path to ground truth .csv files.')
parser.add_argument('--rel_result', default = False, action='store_true',
                    help='Whether retrieve results relative to no_vac.')                    
parser.add_argument('--save_checkpoint', default=False, action='store_true',
                    help='If true, save best checkpoint and final model to .pt file.')
parser.add_argument('--model_save_folder', default= 'chenlin/pygcn/pygcn/trained_model', 
                    help='Folder to save trained model.')                    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Derived variables
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[args.msa_name] #MSA_NAME_FULL = 'San_Francisco_Oakland_Hayward_CA'

# root
hostname = socket.gethostname()
print('hostname: ', hostname)
if(hostname in ['fib-dl3','rl3','rl2']):
    prime = '/data'
elif(hostname=='rl4'):
    prime = '/home'
elif(hostname=='fib-dl'): #dl2
    prime = '/data4'
community_path = os.path.join(prime, 'chenlin/pygcn/pygcn/cbg_to_cluster.npy')
result_df_root = os.path.join(prime, 'chenlin/pygcn/data/safegraph')
all_tested_strategies_path = os.path.join(prime, 'chenlin/pygcn/data/safegraph', args.msa_name, f'all_tested_strategies_{args.msa_name}.npy')
mob_data_root = os.path.join(prime, 'chenlin/COVID-19/Data') #Path to mobility data.

today = str(datetime.date.today()).replace('-','') # yyyy-mm-dd -> yyyymmdd
print('today: ', today)

checkpoint_save_path = os.path.join(prime, args.model_save_folder, f'checkpoint_pairwise_comparator_{today}.pt')
print('checkpoint_save_path: ', checkpoint_save_path)

############################################################################################
# Load simulation data # Both generated in gt-gen-community.py

# result_df
NUM_SEEDS = 30
random_seed_list = [42,43,44]
for seed_idx in range(len(random_seed_list)):
    random_seed = random_seed_list[seed_idx]
    print('random_seed: ', random_seed)
    filename = os.path.join(result_df_root, args.msa_name, 
                            f'vac_results_community_{args.msa_name}_{args.vaccination_ratio}_{args.NN}_randomseed{random_seed}_{NUM_SEEDS}seeds.csv')
    #print('filename: ', filename)
    this_result_df = pd.read_csv(filename)
    col_to_drop = []
    for i in range(len(this_result_df.columns)):
        if('Unnamed' in this_result_df.columns[i]):
            col_to_drop.append(this_result_df.columns[i])
    for i in range(len(col_to_drop)):
        this_result_df.drop(col_to_drop[i], axis=1, inplace=True)
    if(seed_idx==0):
        result_df = this_result_df.copy()
    else:
        result_df = pd.concat([result_df, this_result_df]) # 默认沿axis=0，join=‘out’的方式进行concat
    print('len(result_df): ', len(result_df))

# List of all simulated strategies
all_tested_strategies = np.load(all_tested_strategies_path).tolist()
print('len(all_tested_strategies): ', len(all_tested_strategies))

# Construct dataset (randomly construct pairs)
idx_pair_array = np.zeros((args.num_pairs, 2))
for i in range(len(args.num_pairs)):
    this_idx_pair = random.sample(list(np.arange(len(all_tested_strategies))), 2)
    this_idx_pair = np.sort(np.array(this_idx_pair))
    idx_pair_array[i] = this_idx_pair

############################################################################################
# Load mobility network (adj) and CBG features

output_root = os.path.join(result_df_root, args.msa_name) # Root to save processed mobility data
adj, node_feats = load_data(dataset=f'safegraph-',
                            msa_name=args.msa_name,
                            mob_data_root=mob_data_root,
                            output_root=output_root,
                            normalize=args.normalize,
                            rel_result=args.rel_result,
                            ) 
adj = np.array(adj)

# Calculate node centrality
start = time.time(); print('Start graph construction..')
G_nx = nx.from_numpy_array(adj)
print('Finish graph construction. Time used: ',time.time()-start)
# Convert from networkx to igraph
#d = nx.to_pandas_edgelist(G_nx).values
#G_ig = ig.Graph(d)
G_ig = ig.Graph.from_networkx(G_nx)
start = time.time(); print('Start centrality computation..')
deg_centrality = G_ig.degree()
clo_centrality = G_ig.closeness() #normalized=True
bet_centrality = G_ig.betweenness()
# Normalization
deg_centrality = preprocessing.scale(deg_centrality) #robust_scale
clo_centrality = preprocessing.scale(clo_centrality) #robust_scale
bet_centrality = preprocessing.scale(bet_centrality) #robust_scale
#start = time.time(); print('Start centrality computation..')
#deg_centrality = nx.degree_centrality(G_nx);print('Time for deg: ', time.time()-start); start=time.time()
#clo_centrality = nx.closeness_centrality(G_nx);print('Time for clo: ', time.time()-start); start=time.time()
#bet_centrality = nx.betweenness_centrality(G_nx);print('Time for bet: ', time.time()-start); start=time.time()
#print('Finish centrality computation. Time used: ',time.time()-start)

# Calculate average mobility level
mob_level = np.sum(adj, axis=1)
mob_max = np.max(mob_level)
# Normalization
mob_level = preprocessing.scale(mob_level) #20220120 #robust_scale

deg_centrality = deg_centrality.reshape(-1,1)
clo_centrality = clo_centrality.reshape(-1,1)
bet_centrality = bet_centrality.reshape(-1,1)
mob_level = mob_level.reshape(-1,1)

full_node_feats = np.concatenate((node_feats, deg_centrality, clo_centrality, bet_centrality, mob_level), axis=1)
dim_touched = full_node_feats.shape[1] #赋值给config.dim_touched 
print('full_node_feats.shape: ', full_node_feats.shape)
full_node_feats = torch.Tensor(full_node_feats)


############################################################################################
# Model and optimizer

# Model configurations
config = Config()
config.dim_touched = dim_touched # Num of feats used to calculate embedding
config.gcn_nfeat = config.dim_touched # Num of feats used to calculate embedding
config.gcn_nhid = args.hidden 
config.gcn_nclass = 32 
config.gcn_dropout = args.dropout
config.linear_nin = config.gcn_nclass + (full_node_feats.shape[1]-config.dim_touched)
config.linear_nhid1 = 32 #100 
config.linear_nhid2 = 32 #100

# Initialize model
model = get_model(config, 'PairwiseComparator'); print(model)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# Learning rate schedule
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5, patience=6, min_lr=1e-8, verbose=True) 

if args.cuda:
    model.cuda()
    adj = adj.cuda()
    full_node_feats = full_node_feats.cuda() 
    #node_feats = node_feats.cuda() 
    #deg_centrality = deg_centrality.cuda()
    #clo_centrality = clo_centrality.cuda() 
    #bet_centrality = bet_centrality.cuda() 
    #mob_level = mob_level.cuda() 

############################################################################################
# Training

loss_fn = nn.BCELoss()
train_loss_list = []
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()

    for batch in dataloader:
        

        # Forward
        pred = model(full_node_feats, adj)
        train_loss = loss_fn(pred, true)
        train_loss_list.append(train_loss.item())

        # Backprop
        with torch.autograd.set_detect_anomaly(True):
            train_loss.backward() #train_loss.backward(retain_graph=True)

        # Optimize
        optimizer.step()
        scheduler.step(train_loss)


############################################################################################
# Testing

model.eval()

'''
pairwise_comparisons = model(full_node_feats, adj)
best_strategy = pairwise_to_rank(pairwise_comparisons) #TO-DO: 在utils.py里补一个函数，把pairwise ranking转为ranking
'''
pdb.set_trace()