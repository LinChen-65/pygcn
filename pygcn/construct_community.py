from torch import norm
import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import socket
import os
import sys
import numpy as np
import pandas as pd
import argparse
#from sklearn.cluster import KMeans
#from equal_groups_kmeans import EqualGroupsKMeans #https://github.com/ndanielsen/Same-Size-K-Means
#from balanced_kmeans import kmeans_equal
#import torch
from equal_size_cluster_functions import cluster_equal_size_elki,cluster_equal_size,cluster_equal_size_pair_split,cluster_equal_size_swap,cluster_equal_size_detect_cycles,lloid_equal_size_linear_assignment,cluster_equal_size_mincostmaxflow
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), '../gt-generator'))
import constants

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--msa_name', default='SanFrancisco',
                    help='MSA name.')
parser.add_argument('--epic_data_root', default='/data/chenlin/COVID-19/Data',
                    help='TBA')
args = parser.parse_args()

# Derived variables
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[args.msa_name] 

# root
hostname = socket.gethostname()
print('hostname: ', hostname)
if(hostname in ['fib-dl3','rl3','rl2']):
    root = '/data/chenlin/COVID-19/Data' #dl3
    saveroot = '/data/chenlin/pygcn/pygcn'
elif(hostname=='rl4'):
    root = '/home/chenlin/COVID-19/Data' #rl4
    saveroot = '/home/chenlin/pygcn/pygcn'

# Load CBG ids for the MSA
cbg_ids_msa = pd.read_csv(os.path.join(args.epic_data_root,args.msa_name,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
num_cbgs = len(cbg_ids_msa)

# Mapping from cbg_ids to columns in hourly visiting matrices
cbgs_to_idxs = dict(zip(cbg_ids_msa['census_block_group'].values, range(num_cbgs)))
x = {}
for i in cbgs_to_idxs:
    x[str(i)] = cbgs_to_idxs[i]
# Load SafeGraph data to obtain CBG sizes (i.e., populations)
filepath = os.path.join(args.epic_data_root,"safegraph_open_census_data/data/cbg_b01.csv")
cbg_agesex = pd.read_csv(filepath)
# Extract CBGs belonging to the MSA - https://covid-mobility.stanford.edu//datasets/
cbg_age_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
del cbg_agesex
# Rename
cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
# Extract columns of interest
columns_of_interest = ['census_block_group','Sum']
cbg_age_msa = cbg_age_msa[columns_of_interest].copy()
# Deal with NaN values
cbg_age_msa.fillna(0,inplace=True)
# Deal with CBGs with 0 populations
cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
# Obtain cbg sizes (populations)
cbg_sizes = cbg_age_msa['Sum'].values
cbg_sizes = np.array(cbg_sizes,dtype='int32')
print('Total population: ',np.sum(cbg_sizes))

# Load CBG geographic data
cbg_geo = pd.read_csv(os.path.join(root, 'safegraph_open_census_data/metadata/cbg_geographic_data.csv'))
cbg_geo_msa = pd.merge(cbg_ids_msa, cbg_geo, on='census_block_group', how='left')
latitude = np.array(cbg_geo_msa['latitude'])
longitude = np.array(cbg_geo_msa['longitude'])
location_list = []
for i in range(len(cbg_geo_msa)):
    location_list.append([latitude[i], longitude[i]])
location_array = np.array(location_list)

'''
clf = KMeans(n_clusters=100, random_state=0)#新建KMeans对象，并传入参数
clf.fit(location_array)#进行训练
#print(clf.labels_)
#print(clf.cluster_centers_)
'''
'''
clf = EqualGroupsKMeans(n_clusters=2)
clf.fit(location_array)
clf.labels_
for i in range(100):
    print(len(clf.labels_[clf.labels_==i]), np.sum(cbg_sizes[np.where(clf.labels_==i)]))
'''

'''

N = len(cbg_sizes)
batch_size = 10
num_clusters = 100
device = 'cuda'

cluster_size = N // num_clusters
#X = torch.rand(batch_size, N, dim, device=device)
location_tensor = torch.from_numpy(location_array).to('cuda')
location_tensor = torch.rand(2, 8192, 30, device=device)
pdb.set_trace()
choices, centers = kmeans_equal(location_tensor, num_clusters=num_clusters, cluster_size=cluster_size)
'''

'''
configs = [(2, 8192, 30, 64)]
batch_size = 2
N = 8192
dim = 30
num_clusters = 64
cluster_size = N // num_clusters
X = torch.rand(batch_size, N, dim, device='cuda')
choices, centers = kmeans_equal(X, num_clusters=num_clusters,cluster_size=cluster_size)

def test_gpu_speed(batch_size, N, dim, num_clusters):
    cluster_size = N // num_clusters
    X = torch.rand(batch_size, N, dim, device='cuda')
    choices, centers = kmeans_equal(X, num_clusters=num_clusters,
                                    cluster_size=cluster_size)
'''



nclusters = 100
savepath = os.path.join(saveroot, 'cbg_cluster.png')
#result = cluster_equal_size_elki(location_array, nclusters, show_plt=True, savepath=savepath)
#result = cluster_equal_size(location_array, nclusters, show_plt=True, savepath=savepath)
#result = cluster_equal_size_pair_split(location_array, nclusters, show_plt=True, savepath=savepath)
#result = cluster_equal_size_swap(location_array, nclusters, show_plt=True, savepath=savepath)
#result = cluster_equal_size_detect_cycles(location_array, nclusters, show_plt=True, savepath=savepath)
#result = lloid_equal_size_linear_assignment(location_array, nclusters, show_plt=True, savepath=savepath)
result = cluster_equal_size_mincostmaxflow(location_array, nclusters, show_plt=True, savepath=savepath)


cbg_to_cluster = np.array(result[0])
cluster_size = np.zeros(nclusters)
for i in range(nclusters):
    cluster_size[i] = len(np.where(cbg_to_cluster==i)[0])
print(cluster_size)
savepath = os.path.join(saveroot, 'cbg_to_cluster.npy')
with open(savepath, 'wb') as f:
    np.save(f, cbg_to_cluster)
print(f'cbg_to_cluster saved at: {savepath}')

savepath = os.path.join(saveroot, 'filtered_cbgs.png')
#plt.scatter(location_array[:, 0], location_array[:, 1])
filtered_location_list = []
filtered_cluster_list = []
lo_mean = np.mean(location_array[:, 0])
lo_std  = np.std(location_array[:, 0])
la_mean = np.mean(location_array[:, 1])
la_std  = np.std(location_array[:, 1])
for i in range(len(location_array)):
    if((abs(location_array[i,0]-lo_mean)<=(3 * lo_std)) & (abs(location_array[i,1]-la_mean)<=(3 * la_std))):
        filtered_location_list.append([location_array[i,0], location_array[i,1]])
        filtered_cluster_list.append(cbg_to_cluster[i])
filtered_location_array = np.array(filtered_location_list)

N = nclusters
# define the colormap
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
# define the bins and normalize
bounds = np.linspace(0,N,N+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
scat = plt.scatter(filtered_location_array[:, 0], filtered_location_array[:, 1], c=filtered_cluster_list, cmap=cmap, norm=norm, s=5)
#plt.scatter(filtered_location_array[:, 0], filtered_location_array[:, 1], c=filtered_cluster_list)

# create the colorbar
cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
cb.set_label('Custom cbar')
#plt.title('Discrete color mappings')

plt.savefig(savepath, bbox_inches = 'tight')

pdb.set_trace()