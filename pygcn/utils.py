#from re import M
#from this import s
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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd(), '../gt-generator'))
import constants


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_vac_results(vac_result_path, rel_result): #20220117, 拆分代码
    df = pd.read_csv(vac_result_path)
    num_samples = len(df)-1 #第0行是no_vaccination的结果

    # graph_labels
    # no_vaccination
    #df['Vaccinated_Idxs'][0] = [] 
    try:
        final_cases_no_vac = df['Total_Cases'].loc[0]
        case_rate_std_no_vac = df['Case_Rates_STD'].loc[0]
        final_deaths_no_vac = df['Total_Deaths'].loc[0]
        death_rate_std_no_vac = df['Death_Rates_STD'].loc[0]
    except:
        pass
    
    # vaccination results
    df = df[1:]
    # 把str转为list，split flag是', '，然后再把其中每个元素由str转为int(用map函数)
    df['Vaccinated_Idxs'] = df['Vaccinated_Idxs'].apply(lambda x : list(map(int, (x.strip('[').strip(']').split(', ')))))
    
    if('Total_Deaths' in df.columns):
        graph_labels = torch.FloatTensor(np.array(pd.DataFrame(df,columns=['Total_Cases','Case_Rates_STD','Total_Deaths','Death_Rates_STD'])))
    else:
        graph_labels = torch.FloatTensor(np.array(pd.DataFrame(df,columns=['Total_Cases','Case_Rates_STD'])))
    
    if(rel_result):
        print('rel_result=True')
        graph_labels[:,0] = (graph_labels[:,0]-final_cases_no_vac)
        graph_labels[:,1] = (graph_labels[:,1]-case_rate_std_no_vac)
        if(graph_labels.shape[1]==4):
            graph_labels[:,2] = (graph_labels[:,2]-final_deaths_no_vac)
            graph_labels[:,3] = (graph_labels[:,3]-death_rate_std_no_vac)

    else:
        print('rel_result=False')

    # idx_train, idx_val, idx_test
    # Split train, val, test
    #idx_train = range(int(0.8*num_samples))
    #idx_val = range(int(0.8*num_samples), int(0.9*num_samples))
    #idx_test = range(int(0.9*num_samples), int(num_samples))
    shuffled = np.arange(num_samples)
    np.random.seed(42) #20220201
    np.random.shuffle(shuffled) #20220119

    idx_train = shuffled[:int(0.8*num_samples)]
    #idx_val = shuffled[int(0.8*num_samples):int(0.9*num_samples)]
    #idx_test = shuffled[int(0.9*num_samples):]
    idx_test = shuffled[int(0.8*num_samples):int(0.9*num_samples)]
    idx_val = shuffled[int(0.9*num_samples):]

    # Wrap in tensor 
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    #vac_tags = np.array(df['Vaccinated_Idxs'][1:])
    vac_tags = np.array(df['Vaccinated_Idxs'])

    return graph_labels, idx_train, idx_val, idx_test, num_samples, vac_tags


def load_adj(msa_name,mob_data_root,output_root): #20220117, 拆分代码
    adj_path = os.path.join(output_root, 'adj_%s.npy' % msa_name)
    if(os.path.exists(adj_path)):
        adj = np.load(adj_path)
        num_cbgs = adj.shape[0]
    else:
        # Load POI-CBG visiting matrices
        msa_name_full = constants.MSA_NAME_FULL_DICT[msa_name]
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
    return adj,num_cbgs
    

def load_pretrained_embed(pretrain_embed_path): #20220117, 拆分代码
    #pretrained_embed = np.load('/data/chenlin/code-dynalearn/scripts/figure-6/gt-generator/covid/outputs/node_embeddings_b1.0.npy')
    #pretrained_embed = np.load('/home/chenlin/code-dynalearn/scripts/figure-6/gt-generator/covid/outputs/node_embeddings_b1.0.npy')
    pretrained_embed = np.load(pretrain_embed_path) #20220123

    num_embed = pretrained_embed.shape[1]
    # normalization
    #pretrained_embed = preprocessing.robust_scale(pretrained_embed) #20220127注释
    return pretrained_embed, num_embed


def load_cbg_age(mob_data_root, cbg_ids_msa, normalize=True): #20220117, 拆分代码
    filepath = os.path.join(mob_data_root,"safegraph_open_census_data/data/cbg_b01.csv")
    cbg_agesex = pd.read_csv(filepath)
    # Extract CBGs belonging to the MSA - https://covid-mobility.stanford.edu//datasets/
    cbg_age_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
    # Deal with NaN values
    cbg_age_msa.fillna(0,inplace=True)
    # Deal with CBGs with 0 populations
    cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
    cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
    # Add up males and females of the same age, according to the detailed age list (DETAILED_AGE_LIST)
    # which is defined in Constants.py
    for i in range(3,25+1): # 'B01001e3'~'B01001e25'
        male_column = 'B01001e'+str(i)
        female_column = 'B01001e'+str(i+24)
        cbg_age_msa[constants.DETAILED_AGE_LIST[i-3]] = cbg_age_msa.apply(lambda x : x[male_column]+x[female_column],axis=1)
    # Rename
    cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
    # Extract columns of interest
    #columns_of_interest = ['census_block_group','Sum'] + constants.DETAILED_AGE_LIST
    #cbg_age_msa = cbg_age_msa[columns_of_interest].copy()
    # Deal with CBGs with 0 populations
    cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
    # Calculate elder ratios
    cbg_age_msa['Elder_Absolute'] = cbg_age_msa.apply(lambda x : x['70 To 74 Years']+x['75 To 79 Years']+x['80 To 84 Years']+x['85 Years And Over'],axis=1)
    cbg_age_msa['Elder_Ratio'] = cbg_age_msa['Elder_Absolute'] / cbg_age_msa['Sum']
    cbg_elder_ratio = np.array(cbg_age_msa['Elder_Ratio'])

    # CBG sizes (populations)
    cbg_sizes = cbg_age_msa['Sum'].values
    cbg_sizes = np.array(cbg_sizes,dtype='int32') #;print('Total population: ',np.sum(cbg_sizes))
    cbg_sizes_original = cbg_sizes.copy()

    # normalization
    #if(normalize): #20220127注释
    #    cbg_sizes = preprocessing.robust_scale(cbg_sizes.reshape(-1,1)) 
    #    cbg_elder_ratio = preprocessing.robust_scale(cbg_elder_ratio.reshape(-1,1))

    return cbg_sizes, cbg_sizes_original, cbg_elder_ratio


def load_cbg_income(mob_data_root, cbg_ids_msa, normalize=True): #20220117, 拆分代码
    # Load ACS 5-year (2013-2017) Data: Mean Household Income
    filepath = os.path.join(mob_data_root,"safegraph_open_census_data/data/ACS_5years_Income_Filtered_Summary.csv")
    cbg_income = pd.read_csv(filepath)
    # Drop duplicate column 'Unnamed:0'
    cbg_income.drop(['Unnamed: 0'],axis=1, inplace=True)
    # Extract pois corresponding to the metro area, by merging dataframes
    cbg_income_msa = pd.merge(cbg_ids_msa, cbg_income, on='census_block_group', how='left')
    del cbg_income
    # Rename
    cbg_income_msa.rename(columns = {'total_households':'Total_Households',
                                    'mean_household_income':'Mean_Household_Income'},inplace=True)
    # Deal with NaN values
    cbg_income_msa.fillna(0,inplace=True)
    cbg_household_income = np.array(cbg_income_msa['Mean_Household_Income'])
    
    # Normalization
    #if(normalize): #20220127注释
    #    cbg_household_income = preprocessing.robust_scale(cbg_household_income.reshape(-1,1))
    
    return cbg_household_income


def load_cbg_occupation(mob_data_root, cbg_ids_msa, cbg_sizes, normalize=True): #20220117, 拆分代码
    # cbg_c24.csv: Occupation
    filepath = os.path.join(mob_data_root,"safegraph_open_census_data/data/cbg_c24.csv")
    cbg_occupation = pd.read_csv(filepath)
    # Extract pois corresponding to the metro area, by merging dataframes
    cbg_occupation_msa = pd.merge(cbg_ids_msa, cbg_occupation, on='census_block_group', how='left')
    del cbg_occupation

    columns_of_essential_workers = list(constants.ew_rate_dict.keys())
    for column in columns_of_essential_workers:
        cbg_occupation_msa[column] = cbg_occupation_msa[column].apply(lambda x : x*constants.ew_rate_dict[column])
    cbg_occupation_msa['Essential_Worker_Absolute'] = cbg_occupation_msa.apply(lambda x : x[columns_of_essential_workers].sum(), axis=1)
    cbg_occupation_msa['Sum'] = cbg_sizes #cbg_age_msa['Sum']
    cbg_occupation_msa['Essential_Worker_Ratio'] = cbg_occupation_msa['Essential_Worker_Absolute'] / cbg_occupation_msa['Sum']
    # Deal with NaN values
    cbg_occupation_msa.fillna(0,inplace=True)
    #columns_of_interest = ['census_block_group','Sum','Essential_Worker_Absolute','Essential_Worker_Ratio']
    #cbg_occupation_msa = cbg_occupation_msa[columns_of_interest].copy()
    cbg_ew_ratio = np.array(cbg_occupation_msa['Essential_Worker_Ratio'])

    # Normalization
    #if(normalize):#20220127注释
    #    cbg_ew_ratio = preprocessing.robust_scale(cbg_ew_ratio.reshape(-1,1))
    
    return cbg_ew_ratio


def load_cbg_demographics(msa_name, mob_data_root, normalize=True): #20220117, 拆分代码
    msa_name_full = constants.MSA_NAME_FULL_DICT[msa_name]
    # Obtain CBG populations
    # Load CBG ids for the MSA
    cbg_ids_msa = pd.read_csv(os.path.join(mob_data_root,msa_name,'%s_cbg_ids.csv' % msa_name_full)) 
    cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
    
    # CBG age (elder ratio) and size (population)
    cbg_sizes, cbg_sizes_original, cbg_elder_ratio = load_cbg_age(mob_data_root, cbg_ids_msa, normalize)
    # CBG income
    cbg_household_income = load_cbg_income(mob_data_root, cbg_ids_msa, normalize)
    # CBG essential worker ratio
    cbg_ew_ratio = load_cbg_occupation(mob_data_root, cbg_ids_msa, cbg_sizes_original, normalize)

    # Reshape #20220127
    cbg_sizes = cbg_sizes.reshape(-1,1) 
    cbg_elder_ratio = cbg_elder_ratio.reshape(-1,1)
    cbg_household_income = cbg_household_income.reshape(-1,1)
    cbg_ew_ratio = cbg_ew_ratio.reshape(-1,1)

    return cbg_sizes, cbg_elder_ratio, cbg_household_income, cbg_ew_ratio


def load_data(vac_result_path="../data/cora/", dataset="cora", msa_name=None, mob_data_root=None, output_root=None, pretrain_embed_path=None, normalize=True, rel_result=True, with_vac_flag=True):
    if(rel_result): print('rel_result=True')
    else: print('rel_result=False')
    if('safegraph' in dataset):
        """Load safegraph dataset"""
        print('Loading {} dataset...'.format(dataset))

        if(with_vac_flag): # For training predictor
            # vaccination results
            graph_labels, idx_train, idx_val, idx_test, num_samples, vac_tags = load_vac_results(vac_result_path,rel_result)
            
            # adj
            adj, num_cbgs = load_adj(msa_name,mob_data_root,output_root)

            # node_feats
            # pretrained_embed
            pretrained_embed, num_embed = load_pretrained_embed(pretrain_embed_path) 
            # cbg population and other demographic features
            cbg_sizes, cbg_elder_ratio, cbg_household_income, cbg_ew_ratio = load_cbg_demographics(msa_name, mob_data_root, normalize)
            
            if(normalize): # Data normalization #20220127
                # Obtain statistics on train set -> Apply to all data
                scaler = preprocessing.StandardScaler()
                cbg_sizes = scaler.fit_transform(cbg_sizes)
                cbg_elder_ratio = scaler.fit_transform(cbg_elder_ratio)
                cbg_household_income = scaler.fit_transform(cbg_household_income)
                cbg_ew_ratio = scaler.fit_transform(cbg_ew_ratio)
                pretrained_embed = scaler.fit_transform(pretrained_embed)
                '''
                scaler.fit(cbg_sizes[idx_train])
                cbg_sizes = scaler.transform(cbg_sizes)
                scaler.fit(cbg_elder_ratio[idx_train])
                cbg_elder_ratio = scaler.transform(cbg_elder_ratio)
                scaler.fit(cbg_household_income[idx_train])
                cbg_household_income = scaler.transform(cbg_household_income)
                scaler.fit(cbg_ew_ratio[idx_train])
                cbg_ew_ratio = scaler.transform(cbg_ew_ratio)
                scaler.fit(pretrained_embed[idx_train])
                pretrained_embed = scaler.transform(pretrained_embed)
                '''
            
            node_feats = np.zeros(((num_samples, num_cbgs, 5+num_embed)))
            node_feats[:,:,0] = cbg_sizes.reshape(1,-1)
            node_feats[:,:,1] = cbg_elder_ratio.reshape(1,-1)
            node_feats[:,:,2] = cbg_household_income.reshape(1,-1)
            node_feats[:,:,3] = cbg_ew_ratio.reshape(1,-1)
            node_feats[:,:,4:4+num_embed] = pretrained_embed
            # vac tags
            for i in range(num_samples):
                node_feats[i, vac_tags[i], -1] = 1   
            
            node_feats = torch.FloatTensor(node_feats)
            return adj, node_feats, graph_labels, idx_train, idx_val, idx_test

        else: # For training policy generator
            # adj
            adj, num_cbgs = load_adj(msa_name,mob_data_root,output_root)

            # node_feats
            # pretrained_embed
            pretrained_embed, num_embed = load_pretrained_embed(pretrain_embed_path) 
            # cbg population and other demographic features
            cbg_sizes, cbg_elder_ratio, cbg_household_income, cbg_ew_ratio = load_cbg_demographics(msa_name, mob_data_root, normalize)
            
            if(normalize): # Data normalization #20220128
                # Obtain statistics on train set -> Apply to all data
                scaler = preprocessing.StandardScaler()
                cbg_sizes = scaler.fit_transform(cbg_sizes)
                cbg_elder_ratio = scaler.fit_transform(cbg_elder_ratio)
                cbg_household_income = scaler.fit_transform(cbg_household_income)
                cbg_ew_ratio = scaler.fit_transform(cbg_ew_ratio)
                pretrained_embed = scaler.fit_transform(pretrained_embed)

            node_feats = np.zeros((num_cbgs, 4+num_embed))
            node_feats[:,0] = cbg_sizes.reshape(1,-1)
            node_feats[:,1] = cbg_elder_ratio.reshape(1,-1)
            node_feats[:,2] = cbg_household_income.reshape(1,-1)
            node_feats[:,3] = cbg_ew_ratio.reshape(1,-1)
            node_feats[:,4:4+num_embed] = pretrained_embed

            node_feats = torch.FloatTensor(node_feats)
            return adj, node_feats

    elif(dataset=='cora'):
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(dataset))
        pdb.set_trace()
        '''
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
        '''
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

def visualize(data, bins, save_path): #20220119
    plt.figure()
    plt.hist(data, bins=bins)
    plt.savefig(save_path)
    print('Figure saved at: ', save_path)


def data_loader(node_feats, graph_labels, idx_train, idx_val, idx_test, batch_size, quicktest=False): #20220127
    # Divide data into train/val/test datasets; Wrap data into DataLoader
    if(quicktest):
        batch_size = 2
        idx_train = idx_train[:batch_size*4]
        idx_val = idx_val[:batch_size]
        idx_test = idx_test[:batch_size]

    train_dataset = torch.utils.data.TensorDataset(node_feats[idx_train,:,:],graph_labels[idx_train])
    val_dataset = torch.utils.data.TensorDataset(node_feats[idx_val,:,:],graph_labels[idx_val])
    test_dataset = torch.utils.data.TensorDataset(node_feats[idx_test,:,:],graph_labels[idx_test])
        
    train_loader = DataLoader(
        train_dataset, #train_dataset.dataset,
        batch_size=batch_size,
        shuffle=True)
    val_loader = DataLoader(
        val_dataset, #val_dataset.dataset,
        batch_size=batch_size,
        shuffle=True)
    test_loader = DataLoader(
        test_dataset,#test_dataset.dataset,
        batch_size=batch_size,
        shuffle=False) #shuffle=True)
    return train_loader, val_loader, test_loader 