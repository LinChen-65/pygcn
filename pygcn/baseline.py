import setproctitle
setproctitle.setproctitle("gnn-simu-vac@chenlin")

from utils import load_data
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

sys.path.append(os.path.join(os.getcwd(), '../gt-generator'))
import constants

import time
import pdb

# Training settings
parser = argparse.ArgumentParser()
#parser.add_argument('--no-cuda', action='store_true', default=False,
#                    help='Disables CUDA training.')
#parser.add_argument('--fastmode', action='store_true', default=False,
#                    help='Validate during training pass.')
#parser.add_argument('--seed', type=int, default=42, help='Random seed.')
#parser.add_argument('--epochs', type=int, default=200,
#                    help='Number of epochs to train.')
#parser.add_argument('--lr', type=float, default=0.01,
#                    help='Initial learning rate.')
#parser.add_argument('--weight_decay', type=float, default=5e-4,
#                    help='Weight decay (L2 loss on parameters).')
#parser.add_argument('--hidden', type=int, default=16,
#                    help='Number of hidden units.')
#parser.add_argument('--dropout', type=float, default=0.5,
#                    help='Dropout rate (1 - keep probability).')
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
print(args.rel_result)
pdb.set_trace()
#args.cuda = not args.no_cuda and torch.cuda.is_available()

# Load data
#vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_200_randomseed66_30seeds_1000samples.csv') #20220113
#vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_100_randomseed66_30seeds_1000samples_proportional.csv') #20220118
vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_100_randomseed66_30seeds_1000samples_proportional.csv') #20220118

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


# Extract useful node features
cbg_sizes = np.array(node_feats[0,:,0])
cbg_elder_ratio = np.array(node_feats[0,:,1])
cbg_household_income = np.array(node_feats[0,:,2])
cbg_ew_ratio = np.array(node_feats[0,:,3])

graph_labels = np.array(graph_labels)
pdb.set_trace()
# Normalization
graph_labels[:,0] = preprocessing.robust_scale(graph_labels[:,0])
graph_labels[:,1] = preprocessing.robust_scale(graph_labels[:,1])

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
# Normalization
deg_centrality = preprocessing.robust_scale(deg_centrality)
clo_centrality = preprocessing.robust_scale(clo_centrality)
bet_centrality = preprocessing.robust_scale(bet_centrality)

#clo_centrality = ig.closeness(G_ig);print('Time for deg: ', time.time()-start); start=time.time()
#bet_centrality = ig.betweenness(G_ig);print('Time for clo: ', time.time()-start); start=time.time()
#pdb.set_trace()

#start = time.time(); print('Start centrality computation..')
#deg_centrality = nx.degree_centrality(G_nx);print('Time for deg: ', time.time()-start); start=time.time()
#clo_centrality = nx.closeness_centrality(G_nx);print('Time for clo: ', time.time()-start); start=time.time()
#bet_centrality = nx.betweenness_centrality(G_nx);print('Time for bet: ', time.time()-start); start=time.time()
#print('Finish centrality computation. Time used: ',time.time()-start)

# Calculate average mobility level
mob_level = np.sum(adj, axis=1)
mob_max = np.max(mob_level)
mob_level /= mob_max

# Concatenate CBG features
# population, age, income, occupation, deg_cent, bet_cent, clo_cent
start = time.time(); print('Start constructing result_df..')
num_samples = len(idx_train)
result_df = pd.DataFrame(columns=['Avg_Sizes','Avg_Elder_Ratio','Avg_Household_Income','Avg_EW_Ratio',
                                  'Avg_Deg_Centrality','Avg_Bet_Centrality','Avg_Clo_Centrality',
                                  'Avg_Mob_Level',
                                  'Std_Sizes','Std_Elder_Ratio','Std_Household_Income','Std_EW_Ratio',
                                  'Std_Deg_Centrality','Std_Bet_Centrality','Std_Clo_Centrality',
                                  'Std_Mob_Level',
                                  'Total_Cases','Case_Rates_STD'])
#pdb.set_trace()
for i in range(num_samples):
    target_nodes = np.nonzero(np.array(node_feats[i,:,-1]))[0]
    result_df.loc[i] = [np.mean(cbg_sizes[target_nodes]),np.mean(cbg_elder_ratio[target_nodes]),np.mean(cbg_household_income[target_nodes]),np.mean(cbg_ew_ratio[target_nodes]),
                        np.mean(deg_centrality[target_nodes]),np.mean(bet_centrality[target_nodes]),np.mean(clo_centrality[target_nodes]),
                        np.mean(mob_level[target_nodes]),
                        np.std(cbg_sizes[target_nodes]),np.std(cbg_elder_ratio[target_nodes]),np.std(cbg_household_income[target_nodes]),np.mean(cbg_ew_ratio[target_nodes]),
                        np.std(deg_centrality[target_nodes]),np.std(bet_centrality[target_nodes]),np.std(clo_centrality[target_nodes]),
                        np.std(mob_level[target_nodes]),
                        graph_labels[i,0], graph_labels[i,1]
                    ]
print('Finish result_df construction. Time used: ',time.time()-start)
print('result_df.shape: ', result_df.shape)

# Linear regression
X = result_df[['Avg_Sizes','Avg_Elder_Ratio','Avg_Household_Income','Avg_EW_Ratio',
               'Avg_Deg_Centrality','Avg_Bet_Centrality','Avg_Clo_Centrality',
               'Avg_Mob_Level',
               'Std_Sizes','Std_Elder_Ratio','Std_Household_Income','Std_EW_Ratio',
                'Std_Deg_Centrality','Std_Bet_Centrality','Std_Clo_Centrality',
                'Std_Mob_Level',
               ]]
X = sm.add_constant(X) # adding a constant
Y1 = result_df['Total_Cases']
Y2 = result_df['Case_Rates_STD']
pdb.set_trace()

model = sm.OLS(Y1,X).fit()
#result = model.fit(X,Y1)
print(model.summary())

model = sm.OLS(Y2,X).fit()
#result = model.fit(X,Y2)
print(model.summary())
pdb.set_trace()
