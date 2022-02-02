from pprint import PrettyPrinter
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
from pytorchtools import EarlyStopping

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
parser.add_argument('--weight_decay', type=float, default=5e-4, #L2 normalization
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
#20220126
parser.add_argument('--model_save_folder', default= 'chenlin/pygcn/pygcn/trained_model', 
                    help='Folder to save trained model.')
#20220127
parser.add_argument('--with_pretrained_embed', default= False, action='store_true',
                    help='Whether to use pretrained embeddings from GNN simulator')
parser.add_argument('--with_original_feat', default= False, action='store_true',
                    help='Whether to concat original features')                    
parser.add_argument('--target_code', type=int,
                    help='Prediction target: 0 for total_cases, 1 for case_std.')
# 20220131
parser.add_argument('--NN', type=int,
                    help='Number of CBGs to receive vaccines.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print('args.rel_result: ', args.rel_result)
print('args.quicktest: ', args.quicktest)
print('args.with_pretrained_embed: ', args.with_pretrained_embed)
print('args.with_original_feat: ', args.with_original_feat)
print('args.target_code: ', args.target_code)
print('args.normalize: ', args.normalize)

vac_result_path_list = [f'test_safe_0.01_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed42_40seeds_1000samples_proportional.csv',
                        f'test_safe_0.01_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed88_40seeds_1000samples_proportional.csv',
                        f'test_safe_0.005_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed65_40seeds_1000samples_proportional.csv',
                        f'test_safe_0.0_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed22_40seeds_1000samples_proportional.csv',
                        f'test_safe_0.0_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed56_40seeds_1000samples_proportional.csv',
                        f'safe_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed42_40seeds_1000samples_proportional.csv']
vac_result_path_combined = os.path.join(args.gt_root, args.msa_name, f'vac_results_{args.msa_name}_0.01_{args.NN}_40seeds_combined')
if(os.path.exists(vac_result_path_combined)):
    print('Continue, and vac_result_combined file exists. Wanna recombine?')
    pdb.set_trace()
count = 0
for path in vac_result_path_list:
    data_df = pd.read_csv(os.path.join(args.gt_root, args.msa_name,path));print('len(data_df): ',len(data_df))
    if(count==0):
        vac_result = data_df.copy()
    else:
        vac_result = pd.concat([vac_result,data_df],axis=0)
    count += 1
    print('len(vac_result): ', len(vac_result))
vac_result = vac_result.drop_duplicates()
print('After dropping duplicates, len(vac_result): ', len(vac_result)) 
vac_result.to_csv(vac_result_path_combined)
pdb.set_trace()

'''
vac_result_path_1 = os.path.join(args.gt_root, args.msa_name, f'test_safe_0.01_crossgroup_vac_results_SanFrancisco_0.01_{args.NN}_randomseed42_40seeds_1000samples_proportional.csv') #20220201
vac_result_path_2 = os.path.join(args.gt_root, args.msa_name, f'test_safe_0.01_crossgroup_vac_results_SanFrancisco_0.01_{args.NN}_randomseed88_40seeds_1000samples_proportional.csv') #20220201
vac_result_path_3 = os.path.join(args.gt_root, args.msa_name, 'safe_crossgroup_vac_results_SanFrancisco_0.01_20_randomseed42_40seeds_1000samples_proportional.csv')
vac_result_1 = pd.read_csv(vac_result_path_1)
vac_result_2 = pd.read_csv(vac_result_path_2)
vac_result_3 = pd.read_csv(vac_result_path_3)
len_1 = len(vac_result_1);print('len_1: ', len_1)
len_2 = len(vac_result_2);print('len_2: ', len_2)
len_3 = len(vac_result_3);print('len_3: ', len_3)
vac_result = pd.concat([vac_result_1,vac_result_2,vac_result_3],axis=0)
len_combined = len(vac_result);print('len_combined: ', len_combined)
vac_result = vac_result.drop_duplicates()
len_combined = len(vac_result);print('After dropping duplicates, len_combined: ', len_combined)           
vac_result.to_csv(vac_result_path_combined)
'''

# Load data
#vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_70_randomseed42_40seeds_1000samples_proportional.csv') #20220120
#vac_result_path = os.path.join(args.gt_root, args.msa_name, f'test_crossgroup_vac_results_SanFrancisco_0.01_{args.NN}_randomseed42_40seeds_1000samples_proportional.csv') #20220131
#vac_result_path = os.path.join(args.gt_root, args.msa_name, f'test_safe_0.01_crossgroup_vac_results_SanFrancisco_0.01_{args.NN}_randomseed66_2seeds_1000samples_proportional.csv') #20220131
#vac_result_path = os.path.join(args.gt_root, args.msa_name, f'test_safe_0.01_crossgroup_vac_results_SanFrancisco_0.01_{args.NN}_randomseed42_40seeds_1000samples_proportional.csv') #20220201
#vac_result_path = vac_result_path_3
vac_result_path = vac_result_path_combined #20220201
 
output_root = os.path.join(args.gt_root, args.msa_name)
pretrained_embed_path = os.path.join(args.prefix,'chenlin/code-dynalearn/scripts/figure-6/gt-generator/covid/outputs/node_embeddings_b1.0.npy' )

adj, node_feats, graph_labels, idx_train, idx_val, idx_test = load_data(vac_result_path=vac_result_path, #20220113
                                                                dataset=f'safegraph-',
                                                                msa_name=args.msa_name,
                                                                mob_data_root=args.mob_data_root,
                                                                output_root=output_root,
                                                                pretrain_embed_path=pretrained_embed_path,
                                                                normalize=args.normalize,
                                                                rel_result=args.rel_result,
                                                                ) 

graph_labels = np.array(graph_labels)
print('total_cases, max:',graph_labels[:,0].max())
print('total_cases, min:', graph_labels[:,0].min())
print('total_cases, max-min:',graph_labels[:,0].max()-graph_labels[:,0].min())
print('total_cases, mean:',graph_labels[:,0].mean())
print('total_cases, std:',graph_labels[:,0].std())

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


# Normalization
if(args.normalize):
    print('Normalization.')
    scaler = preprocessing.StandardScaler()
    deg_centrality = scaler.fit_transform(np.array(deg_centrality).reshape(-1,1)).squeeze()
    clo_centrality = scaler.fit_transform(np.array(clo_centrality).reshape(-1,1)).squeeze()
    bet_centrality = scaler.fit_transform(np.array(bet_centrality).reshape(-1,1)).squeeze()
    mob_level = scaler.fit_transform(np.array(mob_level).reshape(-1,1)).squeeze()

num_samples = node_feats.shape[0]
deg_centrality = torch.Tensor(np.tile(deg_centrality,(num_samples,1))).unsqueeze(axis=2) #20220120
clo_centrality = torch.Tensor(np.tile(clo_centrality,(num_samples,1))).unsqueeze(axis=2) #20220120
bet_centrality = torch.Tensor(np.tile(bet_centrality,(num_samples,1))).unsqueeze(axis=2) #20220120
mob_level = torch.Tensor(np.tile(mob_level,(num_samples,1))).unsqueeze(axis=2) #20220120
vac_flag = node_feats[:,:,-1].unsqueeze(axis=2)



if(args.target_code==0):
    target_identifier = 'total_cases' #20220127 #total_cases
elif(args.target_code==1):
    target_identifier = 'case_std' #20220127 #case_std

if((args.with_pretrained_embed) & (not args.with_original_feat)):
    node_feats = np.concatenate((node_feats, deg_centrality, clo_centrality, bet_centrality, mob_level, vac_flag), axis=2) #20220127
    feature_identifier = 'pe'
    dim_touched = node_feats.shape[2]-1
elif((args.with_pretrained_embed) & (args.with_original_feat)):
    node_feats = np.concatenate((node_feats, deg_centrality, clo_centrality, bet_centrality, mob_level,
                                 node_feats, deg_centrality, clo_centrality, bet_centrality, mob_level, 
                                 vac_flag), axis=2) #20220127
    feature_identifier = 'pe_of'
    dim_touched = int((node_feats.shape[2]-1)/2)
elif((not args.with_pretrained_embed) & (not args.with_original_feat)):
    node_feats = np.concatenate((node_feats[:,:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, vac_flag), axis=2) #20220127
    feature_identifier = ''
    dim_touched = node_feats.shape[2]-1
elif((not args.with_pretrained_embed) & (args.with_original_feat)):
    node_feats = np.concatenate((node_feats[:,:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, 
                                 node_feats[:,:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, 
                                 vac_flag), axis=2) #20220127
    feature_identifier = 'of'
    dim_touched = int((node_feats.shape[2]-1)/2)

# 无pretrain_embed，不拼接原始特征，仅把GCN output embedding和vac flag输入MLP layers 
#node_feats = np.concatenate((node_feats[:,:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, vac_flag, vac_flag), axis=2) #20220121
# 无pretrain_embed，把原始特征拼接在GCN output embedding上再输入MLP layers
#node_feats = np.concatenate((node_feats[:,:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, vac_flag, 
#                             node_feats[:,:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, vac_flag), axis=2) #20220123
# 有pretrain_embed，不拼接原始特征 #目前采用这个
#node_feats = np.concatenate((node_feats, deg_centrality, clo_centrality, bet_centrality, mob_level, vac_flag, vac_flag), axis=2) #20220123
#node_feats = np.concatenate((node_feats, deg_centrality, clo_centrality, bet_centrality, mob_level, vac_flag), axis=2) #20220127

print('node_feats.shape: ', node_feats.shape) #(num_samples, num_cbgs, dim_features) #最后一维1=vac，0=no_vac
model_save_path = os.path.join(args.prefix, args.model_save_folder, f'{target_identifier}_{feature_identifier}_{args.epochs}epochs_20220131.pt')
print('model_save_path: ', model_save_path)


node_feats = torch.Tensor(node_feats)
adj = torch.Tensor(adj)
graph_labels = torch.Tensor(graph_labels)


if args.cuda:
    adj = adj.cuda()
    node_feats = node_feats.cuda() #20220114
    if(args.target_code==0):
        graph_labels = graph_labels[:,0].cuda() #20220114 #total_cases
    elif(args.target_code==1):
        graph_labels = graph_labels[:,1].cuda() #20220114 #case_std

     
train_loader, val_loader, test_loader = data_loader(node_feats,graph_labels,idx_train,idx_val,idx_test, batch_size=20, quicktest=args.quicktest)


# Model and optimizer
config = Config()
config.NN = args.NN #20220131
#config.dim_touched = 9 # Num of feats used to calculate embedding #20220123
#config.dim_touched = node_feats.shape[2]-1 # Num of feats used to calculate embedding #20220127
config.dim_touched = dim_touched # Num of feats used to calculate embedding #20220127

config.gcn_nfeat = config.dim_touched # Num of feats used to calculate embedding #20220123
config.gcn_nhid = args.hidden 
'''#初代版本
config.gcn_nclass = 32 #50 #8 #16 #100#200 #20220119 #8(20220114)
'''
config.gcn_nclass = config.gcn_nhid #20220201
config.gcn_dropout = args.dropout


config.linear_nin = config.gcn_nclass-1 + (node_feats.shape[2]-config.dim_touched) #初代版本
#config.linear_nin = (config.gcn_nclass-1 + (node_feats.shape[2]-config.dim_touched)) * config.NN
#config.linear_nin = (config.gcn_nclass-1 + (node_feats.shape[2]-config.dim_touched)) * 4
 #初代版本
'''
config.linear_nhid1 = 100 #8#64
config.linear_nhid2 = 100
'''
config.linear_nhid1 = 32 #64 #100 #8
config.linear_nhid2 = 32 #64 #100

config.linear_nout = 1


model = get_model(config, 'GNN_OVER_MLP')
print(model)
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.1) #20220122
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5, patience=8, min_lr=1e-8, verbose=True) #20220122

# 初始化 early_stopping 对象 #20220201
patience = 40 #20	# 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
checkpoint_save_path = os.path.join(args.prefix, args.model_save_folder, f'checkpoint_{target_identifier}_{feature_identifier}_{args.epochs}epochs_20220201.pt')
print('checkpoint_save_path: ', checkpoint_save_path)
early_stopping = EarlyStopping(patience, verbose=False, path=checkpoint_save_path)	# 关于 EarlyStopping 的代码可先看博客后面的内容

#random.seed(66) #42 #没有影响


def train(epoch,min_valid_loss):
    train_loss = 0.0
    model.train()
    for (batch_x, batch_y) in train_loader:
        #pdb.set_trace()
        optimizer.zero_grad()
        output = model(batch_x, adj) #20220121
        loss = F.mse_loss(output.squeeze(), batch_y)
        loss.backward() #original
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type=2) #20220122 #gradient clipping
        optimizer.step()
        train_loss += loss.item()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        valid_loss = 0.0
        model.eval()
        for (batch_x, batch_y) in val_loader:
            output = model(batch_x, adj) #20220121
            loss= F.mse_loss(output.squeeze(), batch_y)
            valid_loss += loss.item()
        valid_loss /= len(val_loader)
        early_stopping(valid_loss, model)
        
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
        
        # 若满足 early stopping 要求 #20220201
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            return False

        return (train_loss/len(train_loader)), (valid_loss), min_valid_loss     


def test(loader,verbose=True):
    model.eval()
    test_loss = 0.0
    output_test_list = []
    truth_test_list = []
    for (batch_x, batch_y) in loader:
        output = model(batch_x, adj) #20220121
        loss= F.mse_loss(output.squeeze(), batch_y) #20220121
        #loss = F.mse_loss(output.reshape(-1), batch_y) #20220120
        test_loss += loss.item()
        output_test_list = output_test_list + output.squeeze().tolist()
        truth_test_list = truth_test_list + batch_y.squeeze().tolist()

    
    print(f'test loss: {test_loss / len(loader)}')
    if(verbose):
        print('output_test_list: ', output_test_list)
        print('truth_test_list: ', truth_test_list)



# Train model
t_total = time.time()
min_val_loss = np.inf
train_loss_record = []
val_loss_record = []
for epoch in range(args.epochs):
    result = train(epoch,min_val_loss)
    if(result==False): 
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(checkpoint_save_path))
        break
    else:
        train_loss, val_loss, min_val_loss = result
        train_loss_record.append(train_loss)
        val_loss_record.append(val_loss)
        #scheduler.step(train_loss) #val_loss
        scheduler.step(val_loss) #20220131


print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print('train_loss_record: ',train_loss_record)
print('val_loss_record: ',val_loss_record)

pdb.set_trace()

# Testing
test(test_loader)

# Save trained model
print(f'Save trained model at {model_save_path}.')
torch.save(model, model_save_path)
# Test reloading model
#model_reloaded = torch.load(model_save_path)
#test(test_loader)

pdb.set_trace()