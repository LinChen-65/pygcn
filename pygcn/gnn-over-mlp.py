#dl3: python gnn-over-mlp.py --msa_name SanFrancisco --mob_data_root '/data/chenlin/COVID-19/Data' --rel_result --epochs 250 --prefix /data --with_original_feat --target_code 0 --lr 0.005 --NN 20
#rl4: python gnn-over-mlp.py --msa_name SanFrancisco --mob_data_root '/home/chenlin/COVID-19/Data' --rel_result --epochs 250 --prefix /home --with_original_feat --target_code 0 --lr 0.005 --NN 20
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
from sklearn import preprocessing
from models import get_model
from config import *
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorchtools import EarlyStopping
from scipy.stats import spearmanr
import datetime
#from sklearn.model_selection import KFold
#from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset

import time
import pdb

#sys.path.append(os.path.join(os.getcwd(), '../gt-generator'))
#import constants

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
# 20220202
parser.add_argument('--batch_size', type=int, default=20,
                    help='Batch size.')
parser.add_argument('--kfold', default=False, action='store_true',
                    help='Whether apply k-fold cross validation.')                    
# 20220203
parser.add_argument('--resume', default=False, action='store_true',
                    help='Whether to continue training from saved checkpoint.')    

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

today = str(datetime.date.today()).replace('-','') # yyyy-mm-dd -> yyyymmdd
print('today: ', today)

# Combine multiple data files
vac_result_path_list = [f'safe_0.01_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed42_40seeds_1000samples_proportional.csv',
                        f'safe_0.01_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed88_40seeds_1000samples_proportional.csv',
                        f'test_safe_0.005_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed65_40seeds_1000samples_proportional.csv',
                        f'safe_0.0_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed22_40seeds_1000samples_proportional.csv',
                        f'safe_0.0_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed56_40seeds_1000samples_proportional.csv',
                        f'safe_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed42_40seeds_1000samples_proportional.csv',
                        f'test_safe_0.0_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed12_40seeds_2000samples_proportional.csv',
                        f'test_safe_0.0_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed68_40seeds_2000samples_proportional.csv',
                        f'test_safe_0.002_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed99_40seeds_1000samples_proportional.csv',
                        f'test_safe_0.002_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed77_40seeds_1000samples_proportional.csv',
                        f'test_safe_0.001_crossgroup_vac_results_{args.msa_name}_0.01_{args.NN}_randomseed33_40seeds_1000samples_proportional.csv',
                        #f'vac_results_SanFrancisco_0.02_20_randomseed45_40seeds_1000samples_proportional.csv',
                        ]
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
pdb.set_trace()

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
model_save_path = os.path.join(args.prefix, args.model_save_folder, f'{target_identifier}_{feature_identifier}_{args.epochs}epochs_{today}.pt')
print('model_save_path: ', model_save_path)
checkpoint_minloss_save_path = os.path.join(args.prefix, args.model_save_folder, f'checkpoint_{target_identifier}_{feature_identifier}_minloss_{today}.pt')
print('checkpoint_minloss_save_path: ', checkpoint_minloss_save_path)
checkpoint_maxcorr_save_path = os.path.join(args.prefix, args.model_save_folder, f'checkpoint_{target_identifier}_{feature_identifier}_maxcorr_{today}.pt')
print('checkpoint_maxcorr_save_path: ', checkpoint_maxcorr_save_path)

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

if(args.kfold): #20220202
    train_val_dataset, test_loader = data_loader(node_feats,graph_labels,idx_train,idx_val,idx_test, batch_size=args.batch_size, quicktest=args.quicktest, kfold=args.kfold)
else:
    train_loader, val_loader, test_loader = data_loader(node_feats,graph_labels,idx_train,idx_val,idx_test, batch_size=args.batch_size, quicktest=args.quicktest, kfold=args.kfold)


# Model and optimizer
# Network structure
config = Config()
config.NN = args.NN #20220131
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



def train(loader,min_val_loss,max_val_corr):
    train_loss = 0.0
    model.train()
    for (batch_x, batch_y) in loader:
        optimizer.zero_grad()
        output = model(batch_x, adj) #20220121
        loss = F.mse_loss(output.squeeze(), batch_y)
        loss.backward() #original
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type=2) #20220122 #gradient clipping
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(loader)
    #return train_loss #if kfold

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        val_loss = 0.0
        output_val_list = []
        truth_val_list = []
        for (batch_x, batch_y) in val_loader:
            output = model(batch_x, adj) #20220121
            loss= F.mse_loss(output.squeeze(), batch_y)
            val_loss += loss.item()
            output_val_list = output_val_list + output.squeeze().tolist()
            truth_val_list = truth_val_list + batch_y.squeeze().tolist()
        val_loss /= len(val_loader)
        val_corr = spearmanr(np.array(output_val_list),np.array(truth_val_list))[0]
        early_stopping(val_loss, model)
        
        if min_val_loss > val_loss:
            print(f'Validation Loss Decreased({min_val_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
            min_val_loss = val_loss
            save_checkpoint_state(model.state_dict(), epoch, optimizer.state_dict(), scheduler.state_dict(), checkpoint_minloss_save_path); print('minloss checkpoint renewed.')
        if max_val_corr < val_corr:
            print(f'Validation Spearman Correlation Increased({max_val_corr:.6f}--->{val_corr:.6f}) \t Saving The Model')
            max_val_corr = val_corr
            save_checkpoint_state(model.state_dict(), epoch, optimizer.state_dict(), scheduler.state_dict(), checkpoint_maxcorr_save_path); print('maxcorr checkpoint renewed.')
            #torch.save(model.state_dict(), checkpoint_maxcorr_save_path)
        # 若满足 early stopping 要求 #20220201
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            return False

        return train_loss, val_loss, val_corr, min_val_loss, max_val_corr
    


def test(loader,verbose=True):
    model.eval()
    test_loss = 0.0
    output_test_list = []
    truth_test_list = []
    for (batch_x, batch_y) in loader:
        output = model(batch_x, adj) #20220121
        #loss = F.mse_loss(output.reshape(-1), batch_y) #20220120
        loss= F.mse_loss(output.squeeze(), batch_y) #20220121
        test_loss += loss.item()
        output_test_list = output_test_list + output.squeeze().tolist()
        truth_test_list = truth_test_list + batch_y.squeeze().tolist()
    test_loss /= len(test_loader)
    corr = spearmanr(np.array(output_test_list),np.array(truth_test_list))[0]

    print(f'test loss: {test_loss}')
    print(f'Spearman correlation: ', corr)
    if(verbose):
        print('output_test_list: ', output_test_list)
        print('truth_test_list: ', truth_test_list)
    pdb.set_trace()
    return test_loss, corr



if(not args.kfold): # 初代版本, no k-fold
    # Get model
    model = get_model(config, 'GNN_OVER_MLP')
    print(model)
    if args.cuda: model.cuda()

    # Optimization tools
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',factor=0.5, patience=8, min_lr=1e-8, verbose=True) #20220122
    # 初始化 early_stopping 对象 #20220201
    patience = 30 #40 #20	# 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=False, path=checkpoint_minloss_save_path)	

    if(args.resume): #断点续训 #20220203
        model,preformed_epochs,optimizer,scheduler = get_checkpoint_state(checkpoint_maxcorr_save_path,model,optimizer,scheduler)
    else:
        preformed_epochs = 0

    # Train model 
    t_total = time.time()
    min_val_loss = np.inf
    max_val_corr = 0 #20220201
    train_loss_record = []
    val_loss_record = []
    for epoch in range(preformed_epochs,preformed_epochs+args.epochs): #20220203 #for epoch in range(args.epochs)
        print(f'\nEpoch{epoch+1}')
        result = train(train_loader,min_val_loss,max_val_corr)
        if(result==False): 
            # load the last checkpoint with the best model
            model.load_state_dict(torch.load(checkpoint_minloss_save_path))
            #model.load_state_dict(torch.load(checkpoint_maxcorr_save_path))
            break
        else:
            train_loss, val_loss, val_corr, min_val_loss, max_val_corr = result
            train_loss_record.append(train_loss)
            val_loss_record.append(val_loss)
            print(f'Training Loss: {train_loss} \t\t Validation Loss: {val_loss}')
            print(f'Spearman correlation: ', val_corr)
            scheduler.step(max_val_corr) #20220202 #train_loss #val_loss(20220131) 

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print('train_loss_record: ',train_loss_record)
    print('val_loss_record: ',val_loss_record)

else: # k-Fold validation #20220202
    pdb.set_trace()
    '''
    kfold_k = 5
    splits=KFold(n_splits=kfold_k,shuffle=True,random_state=42)
    foldperf={}
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_val_dataset)))):
        print('Fold {}'.format(fold))

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(train_val_dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_val_dataset, batch_size=args.batch_size, sampler=val_sampler)
        
        model = get_model(config, 'GNN_OVER_MLP'); 
        if(fold==0):print(model)
        if args.cuda: model.cuda()
        # Optimization tools
        optimizer = optim.Adam(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',factor=0.5, patience=8, min_lr=1e-8, verbose=True) #20220122
        # 初始化 early_stopping 对象 #20220201
        patience = 40 #20	# 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
        early_stopping = EarlyStopping(patience, verbose=False, path=checkpoint_save_path)	# 关于 EarlyStopping 的代码可先看博客后面的内容

        history = {'train_loss': [], 'val_loss': [], 'val_corr': []}
        min_val_loss = np.inf
        max_val_corr = 0 #20220201
        for epoch in range(args.epochs):
            train_loss = train(train_loader)
            val_loss, val_corr = test(val_loader,verbose=False)

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Val Loss:{:.3f} AVG Val Corr:{:.3f}".format(epoch + 1, args.epochs, train_loss, val_loss, val_corr))
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_corr'].append(val_corr)

        foldperf['fold{}'.format(fold+1)] = history  

    tl_f, vall_f, valc_f =[],[],[]
    for f in range(1,kfold_k+1):
        tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        vall_f.append(np.mean(foldperf['fold{}'.format(f)]['val_loss']))
        valc_f.append(np.mean(foldperf['fold{}'.format(f)]['val_corr']))
    print('Performance of {} fold cross validation'.format(kfold_k))
    print("Average Training Loss: {:.3f} \t Average Val Loss: {:.3f} \t ".format(np.mean(tl_f),np.mean(vall_f),np.mean(valc_f)))     
    '''

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