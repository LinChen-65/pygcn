# python mlp_new.py --rel_result --msa_name SanFrancisco --prefix /data

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
from statsmodels.stats.outliers_influence import summary_table
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time
import pdb

sys.path.append(os.path.join(os.getcwd(), '../gt-generator'))
import constants

## 输出图显示中文
#from matplotlib.font_manager import FontProperties
#fonts = FontProperties(fname = "/Library/Fonts/华文细黑.ttf",size=14)

from sklearn import metrics
from sklearn.model_selection import train_test_split
## 忽略提醒
#import warnings
#warnings.filterwarnings("ignore")
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# Training settings
parser = argparse.ArgumentParser()

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
print('args.rel_result: ', args.rel_result)

# Load data
#vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_200_randomseed66_30seeds_1000samples.csv') #20220113
#vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_100_randomseed66_30seeds_1000samples_proportional.csv') #20220118
#vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_70_randomseed42_40seeds_1000samples_proportional.csv') #20220119
vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.01_20_40seeds_combined') #20220201 #生成过程见gnn-over-mlp.py

    
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
'''
print('total_cases, max:',graph_labels[:,0].max())
print('total_cases, min:', graph_labels[:,0].min())
print('total_cases, max-min:',graph_labels[:,0].max()-graph_labels[:,0].min())
print('total_cases, mean:',graph_labels[:,0].mean())
print('total_cases, std:',graph_labels[:,0].std())

if(graph_labels.shape[1]==4):
    graph_name = 'total_cases_hist_grouped.png'
else:
    graph_name = 'total_cases_hist_notgrouped.png'
visualization_save_path = os.path.join(args.gt_root, args.msa_name,graph_name)
visualize(np.array(graph_labels[:,0]), bins=20, save_path=visualization_save_path)
'''

# Extract useful node features
cbg_sizes = np.array(node_feats[0,:,0])
cbg_elder_ratio = np.array(node_feats[0,:,1])
cbg_household_income = np.array(node_feats[0,:,2])
cbg_ew_ratio = np.array(node_feats[0,:,3])


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
deg_centrality = np.array(G_ig.degree())
clo_centrality = np.array(G_ig.closeness()) #normalized=True
bet_centrality = np.array(G_ig.betweenness())

# Calculate average mobility level
mob_level = np.sum(adj, axis=1)


# Concatenate CBG features
# population, age, income, occupation, deg_cent, bet_cent, clo_cent
start = time.time(); print('Start constructing result_df..')
num_samples = len(idx_train)+len(idx_val)+len(idx_test) #20220123
result_df = pd.DataFrame(columns=['Avg_Sizes','Avg_Elder_Ratio','Avg_Household_Income','Avg_EW_Ratio',
                                  'Avg_Deg_Centrality','Avg_Bet_Centrality','Avg_Clo_Centrality',
                                  'Avg_Mob_Level',
                                  'Std_Sizes','Std_Elder_Ratio','Std_Household_Income','Std_EW_Ratio',
                                  'Std_Deg_Centrality','Std_Bet_Centrality','Std_Clo_Centrality',
                                  'Std_Mob_Level',
                                  'Total_Cases','Case_Rates_STD','Total_Deaths','Death_Rates_STD'])

for i in range(num_samples):
    target_nodes = np.nonzero(np.array(node_feats[i,:,-1]))[0]
    result_df.loc[i] = [np.mean(cbg_sizes[target_nodes]),np.mean(cbg_elder_ratio[target_nodes]),np.mean(cbg_household_income[target_nodes]),np.mean(cbg_ew_ratio[target_nodes]),
                        np.mean(deg_centrality[target_nodes]),np.mean(bet_centrality[target_nodes]),np.mean(clo_centrality[target_nodes]),
                        np.mean(mob_level[target_nodes]),
                        np.std(cbg_sizes[target_nodes]),np.std(cbg_elder_ratio[target_nodes]),np.std(cbg_household_income[target_nodes]),np.mean(cbg_ew_ratio[target_nodes]),
                        np.std(deg_centrality[target_nodes]),np.std(bet_centrality[target_nodes]),np.std(clo_centrality[target_nodes]),
                        np.std(mob_level[target_nodes]),
                        graph_labels[i,0], graph_labels[i,1], graph_labels[i,2], graph_labels[i,3],
                    ]
print('Finish result_df construction. Time used: ',time.time()-start)
print('result_df.shape: ', result_df.shape)
data = result_df

## 准备自变量和因变量
dataX = data.iloc[:,:16]
dataY = data.iloc[:,16]
##  数据切分
train_x,test_x,train_y,test_y = train_test_split(dataX,dataY.values,test_size = 0.1,random_state = 42)
## 数据标准化
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x) #20220127
test_x = scaler.transform(test_x) #20220127
print(train_x.shape)
print(train_y.shape)

# 多元线性回归分析
train_xadd = sm.add_constant(train_x)  ## 添加常数项
lm = sm.OLS(train_y,train_xadd).fit()
print(lm.summary())
## 检查模型在测试集上的预测效果
test_xadd = sm.add_constant(test_x)  ## 添加常数项
pre_y = lm.predict(test_xadd)
print('Linear regression: ')
print("mean absolute error:", metrics.mean_absolute_error(test_y,pre_y))
print("mean squared error:", metrics.mean_squared_error(test_y,pre_y))
print('pred: ', pre_y.tolist())
print('true: ', test_y.tolist())
pdb.set_trace()

## 使用sklearn库进行MLP回归分析
## 定义含有4个隐藏层的MLP网络
mlpr = MLPRegressor(#hidden_layer_sizes=(100,100,100,100), ## 隐藏层的神经元个数
                    hidden_layer_sizes=(100,100),
                    activation='relu', #'tanh', 
                    solver='adam', 
                    alpha=0.0005, #0.0001,   ## L2惩罚参数
                    max_iter=10000, 
                    random_state=123,
                    early_stopping=True, ## 是否提前停止训练
                    validation_fraction=0.1, ## 20%作为验证集
                    tol=1e-3,
                    n_iter_no_change=50
                   )

## 拟合训练数据集
mlpr.fit(train_x,train_y)

## 可视化损失函数
plt.figure()
plt.plot(mlpr.loss_curve_)
plt.xlabel("iters")
plt.ylabel(mlpr.loss)
plt.show()

## 对测试集上进行预测
pre_y = mlpr.predict(test_x)
print('MLP: ')
print("mean absolute error:", metrics.mean_absolute_error(test_y,pre_y))
print("mean squared error:", metrics.mean_squared_error(test_y,pre_y))
## 输出在测试集上的R^2
print("在训练集上的R^2:",mlpr.score(train_x,train_y))
print("在测试集上的R^2:",mlpr.score(test_x,test_y))
pdb.set_trace()