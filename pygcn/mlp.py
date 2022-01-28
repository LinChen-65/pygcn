# python mlp.py --msa_name SanFrancisco  --mob_data_root '/home/chenlin/COVID-19/Data' --rel_result True --epochs 100

import setproctitle
setproctitle.setproctitle("gnn-simu-vac@chenlin")

from utils import *
import argparse
import os
import sys
import networkx as nx
import igraph as ig
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import preprocessing
from models import get_model
from config import *
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
#from torch.utils.data import DataLoader, random_split

import time
import pdb

sys.path.append(os.path.join(os.getcwd(), '../gt-generator'))
import constants

# 限制显卡使用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, #100,#400, #default=16(original)
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
#20220113
parser.add_argument('--gt_root', default=os.path.abspath(os.path.join(os.pardir,'data/safegraph')),
                    help='Path to ground truth .csv files.')
parser.add_argument('--msa_name', 
                    help='MSA name.')
parser.add_argument('--mob_data_root', default = '/data/chenlin/COVID-19/Data',
                    help='Path to mobility data.')
#20220118
parser.add_argument('--normalize', default = True,
                    help='Whether normalize node features or not.')
parser.add_argument('--rel_result', default = False, action='store_true',
                    help='Whether retrieve results relative to no_vac.')
#20220123
parser.add_argument('--quicktest', default= False, action='store_true',
                    help='Whether use a small dataset to quickly test the model.')
parser.add_argument('--prefix', default= '/home', 
                    help='Prefix of data root. /home for rl4, /data for dl3.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print('args.rel_result: ', args.rel_result)


# Load data
#vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_200_randomseed66_30seeds_1000samples.csv') #20220113
#vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_100_randomseed66_30seeds_1000samples_proportional.csv') #20220118
#vac_result_path = os.path.join(args.gt_root, args.msa_name, 'test_vac_results_SanFrancisco_0.02_70_randomseed42_40seeds_1000samples_proportional.csv') #20220119
vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_70_randomseed42_40seeds_1000samples_proportional.csv') #20220120
    
output_root = os.path.join(args.gt_root, args.msa_name)
pretrain_embed_path = os.path.join(args.prefix,'chenlin/code-dynalearn/scripts/figure-6/gt-generator/covid/outputs/node_embeddings_b1.0.npy' )

adj, node_feats, graph_labels, idx_train, idx_val, idx_test = load_data(vac_result_path=vac_result_path, #20220113
                                                                dataset=f'safegraph-',
                                                                msa_name=args.msa_name,
                                                                mob_data_root=args.mob_data_root,
                                                                output_root=output_root,
                                                                pretrain_embed_path=pretrain_embed_path,
                                                                normalize=args.normalize,
                                                                rel_result=args.rel_result,
                                                                ) 

graph_labels = np.array(graph_labels)
print('total_cases, max:',graph_labels[:,0].max())
print('total_cases, min:', graph_labels[:,0].min())
print('total_cases, max-min:',graph_labels[:,0].max()-graph_labels[:,0].min())
print('total_cases, mean:',graph_labels[:,0].mean())
print('total_cases, std:',graph_labels[:,0].std())

'''
# Visualize the distribution of samples
if(graph_labels.shape[1]==4):
    graph_name = 'total_cases_hist_grouped.png'
else:
    graph_name = 'total_cases_hist_notgrouped.png'
visualization_save_path = os.path.join(args.gt_root, args.msa_name,graph_name)
visualize(np.array(graph_labels[:,0]), bins=20, save_path=visualization_save_path)
'''


# Calculate node centrality
start = time.time(); print('Start graph construction..')
adj = np.array(adj)
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
#start = time.time(); print('Start centrality computation..')
#deg_centrality = nx.degree_centrality(G_nx);print('Time for deg: ', time.time()-start); start=time.time()
#clo_centrality = nx.closeness_centrality(G_nx);print('Time for clo: ', time.time()-start); start=time.time()
#bet_centrality = nx.betweenness_centrality(G_nx);print('Time for bet: ', time.time()-start); start=time.time()
#print('Finish centrality computation. Time used: ',time.time()-start)

# Calculate average mobility level
mob_level = np.sum(adj, axis=1)
mob_max = np.max(mob_level)


num_samples = node_feats.shape[0]
deg_centrality = torch.Tensor(np.tile(deg_centrality,(num_samples,1))).unsqueeze(axis=2) #20220120
clo_centrality = torch.Tensor(np.tile(clo_centrality,(num_samples,1))).unsqueeze(axis=2) #20220120
bet_centrality = torch.Tensor(np.tile(bet_centrality,(num_samples,1))).unsqueeze(axis=2) #20220120
mob_level = torch.Tensor(np.tile(mob_level,(num_samples,1))).unsqueeze(axis=2) #20220120
vac_flag = node_feats[:,:,-1].unsqueeze(axis=2)
node_feats = np.concatenate((node_feats[:,:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, vac_flag), axis=2)
#node_feats = np.concatenate((node_feats, vac_flag),axis=2)
print('node_feats.shape: ', node_feats.shape) # (990, 2943, 9) 最后一维1=vac，0=no_vac

node_feats = torch.Tensor(node_feats)
adj = torch.Tensor(adj)
graph_labels = torch.Tensor(graph_labels)

if args.cuda:
    adj = adj.cuda()
    node_feats = node_feats.cuda() #20220114
    graph_labels = graph_labels[:,0].cuda() #20220114 #total_cases
    #graph_labels = graph_labels[:,1].cuda() #20220114 #case_std

train_loader, val_loader, test_loader = data_loader(node_feats,graph_labels,idx_train,idx_val,idx_test, batch_size=20, quicktest=args.quicktest)


# Model and optimizer
config = Config()
config.linear_nin = (node_feats.shape[2]-1)#*2 #num_feats
config.linear_nhid1 = 100 #8#64
config.linear_nhid2 = 100
config.linear_nout = 1
config.pool_nin = node_feats.shape[2] #20220120
config.pool_nout = node_feats.shape[2] #* 2 #20220120

model = get_model(config, 'MLP')
print(model)
#pdb.set_trace()

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

random.seed(42)

if args.cuda:
    model.cuda()
    ##adj = adj.cuda()
    ##idx_train = idx_train.cuda()
    ##idx_val = idx_val.cuda()
    ##idx_test = idx_test.cuda()

    ##node_feats = node_feats.cuda() #20220114
    ##graph_labels = graph_labels[:,0].cuda() #20220114 #total_cases
    #graph_labels = graph_labels[:,1].cuda() #20220114 #case_std


'''
def data_loader(dataset, batch_size):
    train_dataset, val_dataset, test_dataset = random_split(dataset, [int(0.8*num_samples), int(0.1*num_samples), int(0.1*num_samples)])
    train_loader = DataLoader(
        train_dataset, #train_dataset.dataset,
        batch_size=batch_size,
        shuffle=True)
    val_loader = DataLoader(
        val_dataset,#val_dataset.dataset,
        batch_size=batch_size,
        shuffle=True)
    test_loader = DataLoader(
        test_dataset,#test_dataset.dataset,
        batch_size=batch_size, #1,
        shuffle=False)
    return train_loader, val_loader, test_loader 
'''

def train(epoch,min_valid_loss):
    train_loss = 0.0
    model.train()
    for (batch_x, batch_y) in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = F.mse_loss(output.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        valid_loss = 0.0
        model.eval()
        for (batch_x, batch_y) in val_loader:
            output = model(batch_x)
            loss= F.mse_loss(output.squeeze(), batch_y)
            valid_loss += loss.item()

        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_loader)}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
    return min_valid_loss     
        

def test(loader):
    model.eval()
    test_loss = 0.0
    output_test_list = []
    truth_test_list = []
    for (batch_x, batch_y) in loader:
        output = model(batch_x)
        loss= F.mse_loss(output.squeeze(), batch_y)
        #loss = F.mse_loss(output.reshape(-1), batch_y)
        test_loss += loss.item()
        output_test_list = output_test_list + output.squeeze().tolist()
        truth_test_list = truth_test_list + batch_y.squeeze().tolist()

    print(f'test loss: {test_loss / len(test_loader)}')
    print('output_test_list: ', output_test_list)
    print('truth_test_list: ', truth_test_list)
    pdb.set_trace()


# Wrap data into DataLoader
'''
dataset = torch.utils.data.TensorDataset(node_feats,graph_labels)
train_loader, val_loader, test_loader = data_loader(dataset,batch_size=20)
'''

# Train model
t_total = time.time()
min_valid_loss = np.inf
for epoch in range(args.epochs):
    min_valid_loss = train(epoch,min_valid_loss)
    #early_stop,min_valid_loss = train(epoch,min_valid_loss)
    #if(early_stop):
    #    break    
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
pdb.set_trace()

# Testing
test(test_loader)

