import setproctitle
setproctitle.setproctitle("gnn-simu-vac@chenlin")

from utils import load_data, visualize
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
print('args.rel_result: ', args.rel_result)
#pdb.set_trace()

# Load data
#vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_200_randomseed66_30seeds_1000samples.csv') #20220113
#vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_100_randomseed66_30seeds_1000samples_proportional.csv') #20220118
#vac_result_path = os.path.join(args.gt_root, args.msa_name, 'test_vac_results_SanFrancisco_0.02_70_randomseed42_40seeds_1000samples_proportional.csv') #20220119
vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.01_20_40seeds_combined') #20220201 #生成过程见gnn-over-mlp.py

'''
vac_result_path = os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_100_randomseed66_30seeds_1000samples_proportional_all.csv') #20220118
if(not os.path.exists(vac_result_path)):
    df_1 = pd.read_csv(os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_70_randomseed42_40seeds_1000samples_proportional.csv'))
    df_2 = pd.read_csv(os.path.join(args.gt_root, args.msa_name, 'vac_results_SanFrancisco_0.02_70_randomseed42_40seeds_1000samples_proportional_latterpart.csv'))
    df_combined = pd.concat([df_1, df_2], axis=0)
    print(len(df_combined))
    df_combined.reset_index(inplace=True, drop=True)
    df_combined = df_combined.drop_duplicates(subset='Vaccinated_Idxs')
    print(len(df_combined))
    pdb.set_trace()
    df_combined.to_csv(vac_result_path)
'''    
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

graph_labels = np.array(graph_labels)
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
pdb.set_trace()

# Extract useful node features
cbg_sizes = np.array(node_feats[0,:,0])
cbg_elder_ratio = np.array(node_feats[0,:,1])
cbg_household_income = np.array(node_feats[0,:,2])
cbg_ew_ratio = np.array(node_feats[0,:,3])

# Normalization
for i in range(graph_labels.shape[1]):
    graph_labels[:,i] = preprocessing.robust_scale(graph_labels[:,i])
#graph_labels[:,0] = preprocessing.robust_scale(graph_labels[:,0])
#graph_labels[:,1] = preprocessing.robust_scale(graph_labels[:,1])
#graph_labels[:,2] = preprocessing.robust_scale(graph_labels[:,2])
#graph_labels[:,3] = preprocessing.robust_scale(graph_labels[:,3])

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
                                  'Total_Cases','Case_Rates_STD','Total_Deaths','Death_Rates_STD'])
#pdb.set_trace()
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
Y3 = result_df['Total_Deaths']
Y4 = result_df['Death_Rates_STD']

model = sm.OLS(Y1,X).fit()
print('Total_Cases: \n', model.summary())

model = sm.OLS(Y2,X).fit()
print('Case_Rates_STD: \n',model.summary())

model = sm.OLS(Y3,X).fit()
print('Total_Deaths: \n',model.summary())

model = sm.OLS(Y4,X).fit()
print('Death_Rates_STD: \n',model.summary())
pdb.set_trace()
