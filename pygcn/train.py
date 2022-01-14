from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

#from pygcn.utils import load_data, accuracy #original
#from pygcn.models import GCN #original
from utils import load_data, accuracy #20220112
from models import GCN # #20220112

from torch.nn import Sequential
from models import get_model
from config import *
import random
import pdb
import os

#在命令行中输入参数了（先写参数名称，空格，再写参数的值）：python opp.py --height 5 --width 4 --length 3
# https://blog.csdn.net/fjswcjswzy/article/details/105737647?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-0.pc_relevant_paycolumn_v2&spm=1001.2101.3001.4242.1&utm_relevant_index=3

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
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
#20220113
parser.add_argument('--gtfilepath', default=os.path.abspath(os.path.join(os.pardir,'data/safegraph')),
                    help='Path for ground truth .csv files.')
parser.add_argument('--msa_name', 
                    help='MSA name.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#adj, features, labels, idx_train, idx_val, idx_test = load_data() #original
filepath = os.path.join(args.gtfilepath, args.msa_name, 'vac_results_SanFrancisco_0.02_200_randomseed66_30seeds_5samples.csv') #20220113
adj, features, labels, idx_train, idx_val, idx_test = load_data(path=filepath, dataset=f'safegraph-{args.msa_name}') #20220113
pdb.set_trace()

num_graphs = 1 #test
idx_train = torch.tensor([0])
idx_val = torch.tensor([0])
idx_test = torch.tensor([0])
#perturbed_features = [features + np.random.randn(features.shape[0], features.shape[1]) for i in range(num_graphs)] #test

perturbed_features = torch.zeros(num_graphs, features.shape[0], features.shape[1])
for i in range(num_graphs):
    perturbed_features[i] = features + torch.randn(features.shape[0], features.shape[1])

#perturbed_features = features     
graph_labels = torch.randn(num_graphs) #test



# Model and optimizer
'''
#original
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
'''
config = Config()
config.gcn_nfeat = features.shape[1]
config.gcn_nhid = args.hidden
config.gcn_nclass = labels.max().item() + 1
config.gcn_dropout = args.dropout
config.linear_nin = features.shape[0] #num_nodes
config.linear_nhid1 = 64
config.linear_nhid2 = 8
config.linear_nout = 1

model = get_model(config)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

    perturbed_features = perturbed_features.cuda() #20220112
    graph_labels = graph_labels.cuda() #20220112


def train(epoch):
    t = time.time()
    model.train()
    #sample_idx_train = random.sample(idx_train, 1)
    #sample_idx_val = random.sample(idx_val, 1)
    sample_idx_train = torch.tensor(random.sample(range(len(idx_train)), 1))
    sample_idx_val = torch.tensor(random.sample(range(len(idx_val)), 1))
    #pdb.set_trace()
    optimizer.zero_grad()
    #output = model(features, adj) #original
    gcn_output = model[0](perturbed_features[sample_idx_train].squeeze(), adj) #20220112
    compressed = torch.mean(gcn_output, axis=1) #20220112
    output = model[1](compressed)#20220112
    loss_train = F.l1_loss(output, graph_labels[sample_idx_train.squeeze()]) #20220112

    #loss_train = F.nll_loss(output[idx_train], labels[idx_train])#original
    #acc_train = accuracy(output[idx_train], labels[idx_train])#original
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        gcn_output = model[0](perturbed_features[sample_idx_val].squeeze(), adj) #20220112
        compressed = torch.mean(gcn_output, axis=1) #20220112
        output = model[1](compressed)

    loss_val = F.l1_loss(output, graph_labels[sample_idx_val]) #20220112
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    #pdb.set_trace()

    '''
    #original
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    '''

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
pdb.set_trace()

# Testing
#test()
