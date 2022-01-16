import numpy as np
import scipy.sparse as sp
import torch

import pandas as pd
import os
import sys
import pickle
import pdb
import time
from sklearn import preprocessing

sys.path.append(os.path.join(os.getcwd(), '../gt-generator'))
import constants


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(vac_result_path="../data/cora/", dataset="cora", msa_name=None, mob_data_root=None, output_root=None):
    if('safegraph' in dataset):
        """Load safegraph dataset"""
        print('Loading {} dataset...'.format(dataset))
        msa_name_full = constants.MSA_NAME_FULL_DICT[msa_name]
        df = pd.read_csv(vac_result_path)
        num_samples = len(df)-1

        vaccination_ratio = 0.02
        nn = 200

        # graph_labels
        # 把str转为list，split flag是', '，然后再把其中每个元素由str转为int(用map函数)
        df['Vaccinated_Idxs'][0] = [] 
        df['Vaccinated_Idxs'][1:] = df['Vaccinated_Idxs'][1:].apply(lambda x : list(map(int, (x.strip('[').strip(']').split(', ')))))
        final_cases_no_vac = df['Total_Cases'].loc[0]
        case_rate_std_no_vac = df['Case_Rates_STD'].loc[0]
        graph_labels = torch.FloatTensor(np.array(pd.DataFrame(df[1:],columns=['Total_Cases','Case_Rates_STD'])))

        # idx_train, idx_val, idx_test
        # Split train, val, test
        idx_train = range(int(0.6*num_samples))
        idx_val = range(int(0.6*num_samples), int(0.8*num_samples))
        idx_test = range(int(0.8*num_samples), int(num_samples))
        # Wrap in tensor 
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        # adj
        adj_path = os.path.join(output_root, 'adj_%s.npy' % msa_name)
        if(os.path.exists(adj_path)):
            adj = np.load(adj_path)
            num_cbgs = adj.shape[0]
        else:
            # Load POI-CBG visiting matrices
            mob_data_path = os.path.join(mob_data_root, msa_name, '%s_2020-03-01_to_2020-05-02.pkl' % msa_name_full)
            f = open(mob_data_path, 'rb') 
            poi_cbg_visits_list = pickle.load(f)
            f.close()
            # Get one timestep's data for test 
            single_array = poi_cbg_visits_list[0].todense()
            num_pois = single_array.shape[0]; print('num_pois: ', num_pois)
            num_cbgs = single_array.shape[1]; print('num_cbgs: ', num_cbgs)
            num_nodes = num_pois + num_cbgs; print('num_nodes (cbg+poi): ', num_nodes)
            #num_days = int(len(poi_cbg_visits_list)/24) #1512h/24h
            # Average all poi-cbg arrays
            num_hours = len(poi_cbg_visits_list); print('num_hours: ', num_hours)
            avg_array_path = os.path.join(output_root, 'avg_array_%s.npy' % msa_name)
            if(os.path.exists(avg_array_path)):
                avg_array = np.load(avg_array_path)
            else:
                avg_array = np.zeros(poi_cbg_visits_list[0].shape)
                for i in range(num_hours):
                    if(i%50==0):print(i)
                    avg_array += poi_cbg_visits_list[i]
                avg_array /= num_hours
                np.save(avg_array_path, avg_array)
            # Construct adjacency matrix
            adj = np.zeros((num_cbgs, num_cbgs))
            for i in range(adj.shape[0]):
                if(i%200==0):print(i)
                for j in range(adj.shape[1]):
                    adj[i][j] = np.sum(avg_array[:,i] * avg_array[:,j])
            np.save(adj_path, adj)
        adj = torch.FloatTensor(adj)

        # node_feats
        pretrained_embed = np.load('/data/chenlin/code-dynalearn/scripts/figure-6/gt-generator/covid/outputs/node_embeddings_b1.0.npy')
        num_embed = pretrained_embed.shape[1]
        # normalization
        pretrained_embed = preprocessing.robust_scale(pretrained_embed)
        node_feats = np.zeros(((num_samples, num_cbgs, 2+num_embed)))

        # Obtain CBG populations
        # Load CBG ids for the MSA
        cbg_ids_msa = pd.read_csv(os.path.join(mob_data_root,msa_name,'%s_cbg_ids.csv' % msa_name_full)) 
        cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
        # Load SafeGraph data to obtain CBG sizes (i.e., populations)
        filepath = os.path.join(mob_data_root,"safegraph_open_census_data/data/cbg_b01.csv")
        cbg_agesex = pd.read_csv(filepath)
        # Extract CBGs belonging to the MSA - https://covid-mobility.stanford.edu//datasets/
        cbg_age_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
        # Deal with NaN values
        cbg_age_msa.fillna(0,inplace=True)
        # Deal with CBGs with 0 populations
        cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
        cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
        # Obtain cbg sizes (populations)
        cbg_sizes = cbg_age_msa['Sum'].values
        cbg_sizes = np.array(cbg_sizes,dtype='int32') #;print('Total population: ',np.sum(cbg_sizes))
        # normalization
        cbg_sizes = preprocessing.robust_scale(cbg_sizes.reshape(-1,1))
        node_feats[:,:,0] = cbg_sizes.reshape(1,-1)

        for i in range(num_samples):
            node_feats[i, np.array(df['Vaccinated_Idxs'][i+1]) , 1] = 1
        
        node_feats[:,:,2:] = pretrained_embed

        node_feats = torch.FloatTensor(node_feats)

        #pdb.set_trace()
        return adj, node_feats, graph_labels, idx_train, idx_val, idx_test

    elif(dataset=='cora'):
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(dataset))

        idx_features_labels = np.genfromtxt("{}{}.content".format(vac_result_path, dataset),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(vac_result_path, dataset),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = normalize(features)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test

    else:
        print('Invalid dataset. Please check.')
        pdb.set_trace()

    


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
