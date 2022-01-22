from __future__ import division
from __future__ import print_function

import setproctitle
setproctitle.setproctitle("gnn-simu-vac@chenlin")

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy #20220112
from models import GCN # #20220112

from torch.nn import Sequential
from models import get_model
from config import *
import random
import pdb
import os
import sys

sys.path.append(os.path.join(os.getcwd(), '../gt-generator'))
import constants

# 限制显卡使用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
parser.add_argument('--hidden', type=int, default=30, #100,#400, #default=16(original)
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
parser.add_argument('--rel_result', default = False,
                    help='Whether retrieve results relative to no_vac.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_200_randomseed66_30seeds_500samples.csv') #20220113
vac_result_path = os.path.join(args.gt_root, args.msa_name, 'test_vac_results_SanFrancisco_0.02_70_randomseed42_40seeds_1000samples_proportional.csv') #20220119

msa_name_full = constants.MSA_NAME_FULL_DICT[args.msa_name]
output_root = os.path.join(args.gt_root, args.msa_name)

adj, node_feats, graph_labels, idx_train, idx_val, idx_test = load_data(vac_result_path=vac_result_path, #20220113
                                                                dataset=f'safegraph-',
                                                                msa_name=args.msa_name,
                                                                mob_data_root=args.mob_data_root,
                                                                output_root=output_root,
                                                                normalize=args.normalize,
                                                                #rel_result=args.rel_result,
                                                                rel_result=False,
                                                                ) 



# Model and optimizer
config = Config()
config.gcn_nfeat = node_feats.shape[2] #num_feats #20220114
#config.gcn_nhid = args.hidden #original
config.gcn_nhid1 = args.hidden #20220119
config.gcn_nhid2 = args.hidden #20220119
#config.gcn_nhid3 = args.hidden #20220119
config.gcn_nclass = 16 #100#200 #20220119 #8(20220114)
config.gcn_dropout = args.dropout
#config.linear_nin = node_feats.shape[1] #num_nodes #20220114
config.linear_nin = config.gcn_nclass #20220119
config.linear_nhid1 = 8#64
config.linear_nhid2 = 4
config.linear_nout = 1

model = get_model(config)
print(model)
#pdb.set_trace()

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

random.seed(42)

# small dataset, check network #20220119
idx_train = idx_train[:16]#[:20]#[:100]
idx_val = idx_val[:16]#[:2]#[:5]
idx_test = idx_test[:16]#[:2]#[:5]

if args.cuda:
    model.cuda()
    adj = adj.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

    node_feats = node_feats.cuda() #20220114
    #graph_labels = graph_labels.cuda() #20220112
    graph_labels = graph_labels[:,0].cuda() #20220114 #total_cases
    #graph_labels = graph_labels[:,1].cuda() #20220114 #case_std


def train(epoch):
    t = time.time()
    model.train()

    #sample_idx_train = idx_train[random.sample(range(len(idx_train)), 1)] #20220114
    #sample_idx_val = idx_val[random.sample(range(len(idx_val)), 1)] #20220114
    #optimizer.zero_grad() #original

    output_train_list = []
    truth_train_list = []
    loss_train_list = []
    accumulation_step = 20 #20220115
    #for i in range(len(idx_train)):
    for i in range(accumulation_step):
        #sample_idx_train = idx_train[i] #20220114
        sample_idx_train = idx_train[random.sample(range(len(idx_train)), 1)] #20220114
        #output = model(features, adj) #original
        gcn_output = model[0](node_feats[sample_idx_train].squeeze(), adj) #20220114
        #compressed = torch.mean(gcn_output, axis=1) #20220112
        compressed = torch.mean(gcn_output, axis=0) #20220119
        output = model[1](compressed)#20220112
        #loss_train = F.l1_loss(output, graph_labels[sample_idx_train.squeeze()]) #20220112
        loss_train = F.mse_loss(output.squeeze(), graph_labels[sample_idx_train.squeeze()]) #20220114
        loss_train.backward() #20220115

        output_train_list.append(output.item())
        truth_train_list.append(graph_labels[sample_idx_train.squeeze()].item())
        loss_train_list.append(loss_train.item())

    #loss_train = torch.mean(torch.Tensor(np.array(loss_train_list)).requires_grad_()) #20220114
    #loss_train.backward() #20220114
    optimizer.step()
    optimizer.zero_grad()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()

        output_val_list = []
        truth_val_list = []
        loss_val_list = []
        for i in range(len(idx_val)):
            sample_idx_val = idx_val[i]
            gcn_output = model[0](node_feats[sample_idx_val].squeeze(), adj) #20220112
            #compressed = torch.mean(gcn_output, axis=1) #20220112
            compressed = torch.mean(gcn_output, axis=0) #20220119
            output = model[1](compressed)

            #loss_val = F.l1_loss(output, graph_labels[sample_idx_val]) #20220112
            loss_val = F.mse_loss(output.squeeze(), graph_labels[sample_idx_val]) #20220114
            
            output_val_list.append(output.item())
            truth_val_list.append(graph_labels[sample_idx_val.squeeze()].item())
            loss_val_list.append(loss_val.item())

        loss_val = np.mean(np.array(loss_val_list))
        #if(loss_val<=6000):
        #    print(loss_val)
        #    return True
        
        if(epoch%100==0):
            print('Epoch: {:04d}'.format(epoch+1),
                #'loss_train: {:.4f}'.format(loss_train.item()),
                'loss_train: {:.4f}'.format(np.mean(np.array(loss_train_list))),
                'loss_val: {:.4f}'.format(loss_val.item()),
                #'time: {:.4f}s'.format(time.time() - t),
                )
            #pdb.set_trace()

    return False


def test():
    model.eval()
    
    #sample_idx_test = idx_test[random.sample(range(len(idx_test)), 1)] #20220114
    output_test_list = []
    truth_test_list = []
    loss_test_list = []
    for i in range(len(idx_test)):
        sample_idx_test = idx_test[i]
        gcn_output = model[0](node_feats[sample_idx_test].squeeze(), adj) #20220112
        #compressed = torch.mean(gcn_output, axis=1) #20220112
        compressed = torch.mean(gcn_output, axis=0) #20220119
        output = model[1](compressed)
        #loss_test = F.l1_loss(output, graph_labels[sample_idx_test.squeeze()]) #20220112
        loss_test = F.mse_loss(output.squeeze(), graph_labels[sample_idx_test.squeeze()]) #20220114

        output_test_list.append(output.item())
        truth_test_list.append(graph_labels[sample_idx_test.squeeze()].item())
        loss_test_list.append(loss_test.item())
    
    loss_test = np.mean(np.array(loss_test_list))
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          )
    #print(np.array(output_list)-np.array(truth_list))
    print('output_test_list: ', output_test_list)
    print('truth_test_list: ', truth_test_list)
    pdb.set_trace()


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    #train(epoch) #original
    early_stop = train(epoch)
    if(early_stop):
        break    
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
pdb.set_trace()

# Testing
test()
