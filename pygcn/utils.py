#from re import M
#from this import s
from random import random
import numpy as np
import scipy.sparse as sp
import torch

import pandas as pd
import os
import sys
import pickle
import pdb
from sklearn import preprocessing
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from torch.distributions import Categorical

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

    # Split train, val, test # idx_train, idx_val, idx_test
    shuffled = np.arange(num_samples)
    np.random.seed(42) #20220201
    np.random.shuffle(shuffled) #20220119

    idx_train = shuffled[:int(0.8*num_samples)]
    idx_test = shuffled[int(0.8*num_samples):int(0.9*num_samples)]
    idx_val = shuffled[int(0.9*num_samples):]

    # Wrap in tensor 
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    vac_tags = np.array(df['Vaccinated_Idxs'])

    return graph_labels, idx_train, idx_val, idx_test, num_samples, vac_tags


def load_adj(msa_name, data_root, output_root): #20220117, 拆分代码
    adj_path = os.path.join(output_root, 'adj_%s.npy' % msa_name)
    if(os.path.exists(adj_path)):
        adj = np.load(adj_path)
        num_cbgs = adj.shape[0]
    else:
        # Load POI-CBG visiting matrices
        msa_name_full = constants.MSA_NAME_FULL_DICT[msa_name]
        mob_data_path = os.path.join(data_root, msa_name, '%s_2020-03-01_to_2020-05-02.pkl' % msa_name_full)
        f = open(mob_data_path, 'rb') 
        poi_cbg_visits_list = pickle.load(f)
        f.close()
        # Get one timestep's data for test 
        single_array = poi_cbg_visits_list[0].todense()
        num_pois = single_array.shape[0]; print('num_pois: ', num_pois)
        num_cbgs = single_array.shape[1]; print('num_cbgs: ', num_cbgs)
        num_nodes = num_pois + num_cbgs; print('num_nodes (cbg+poi): ', num_nodes)
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
    pretrained_embed = np.load(pretrain_embed_path) #20220123
    num_embed = pretrained_embed.shape[1]
    return pretrained_embed, num_embed


def load_cbg_age(data_root, cbg_ids_msa, normalize=True): #20220117, 拆分代码
    filepath = os.path.join(data_root,"safegraph_open_census_data/data/cbg_b01.csv")
    cbg_agesex = pd.read_csv(filepath)
    # Extract CBGs belonging to the MSA
    cbg_age_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
    # Deal with NaN values
    cbg_age_msa.fillna(0,inplace=True)
    # Deal with CBGs with 0 populations
    cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
    cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
    # Add up males and females of the same age, according to the detailed age list (DETAILED_AGE_LIST)
    # which is defined in constants.py
    for i in range(3,25+1): # 'B01001e3'~'B01001e25'
        male_column = 'B01001e'+str(i)
        female_column = 'B01001e'+str(i+24)
        cbg_age_msa[constants.DETAILED_AGE_LIST[i-3]] = cbg_age_msa.apply(lambda x : x[male_column]+x[female_column],axis=1)
    # Rename
    cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
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


def load_cbg_income(data_root, cbg_ids_msa, normalize=True): #20220117, 拆分代码
    filepath = os.path.join(data_root,"safegraph_open_census_data/data/ACS_5years_Income_Filtered_Summary.csv")
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
    
    return cbg_household_income


def load_cbg_occupation(data_root, cbg_ids_msa, cbg_sizes, normalize=True): #20220117, 拆分代码
    filepath = os.path.join(data_root,"safegraph_open_census_data/data/cbg_c24.csv")
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
    cbg_ew_ratio = np.array(cbg_occupation_msa['Essential_Worker_Ratio'])
    
    return cbg_ew_ratio


def load_cbg_minority(data_root, cbg_ids_msa, cbg_sizes, normalize=True): #20220401加
    # cbg_b03.csv: HISPANIC OR LATINO ORIGIN BY RACE
    filepath = os.path.join(data_root,"safegraph_open_census_data/data/cbg_b03.csv")
    cbg_ethnic = pd.read_csv(filepath)
    # Extract pois corresponding to the metro area, by merging dataframes
    cbg_ethnic_msa = pd.merge(cbg_ids_msa, cbg_ethnic, on='census_block_group', how='left')
    del cbg_ethnic
    cbg_ethnic_msa.rename(columns={'B03002e1':'Sum',
                                   'B03002e2':'NH_Total',
                                   'B03002e3':'NH_White',
                                   'B03002e4':'NH_Black',
                                   'B03002e5':'NH_Indian',
                                   'B03002e6':'NH_Asian',
                                   'B03002e7':'NH_Hawaiian',
                                   'B03002e12':'Hispanic'}, inplace=True)
    cbg_ethnic_msa['Sum'] = cbg_sizes #cbg_age_msa['Sum']
    cbg_ethnic_msa['Minority_Absolute'] = cbg_ethnic_msa['Sum'] - cbg_ethnic_msa['NH_White'] 
    cbg_ethnic_msa['Minority_Ratio'] = cbg_ethnic_msa['Minority_Absolute'] / cbg_ethnic_msa['Sum']
    # Deal with NaN values
    cbg_ethnic_msa.fillna(0,inplace=True)
    cbg_minority_ratio = np.array(cbg_ethnic_msa['Minority_Ratio'])

    return cbg_minority_ratio




def load_cbg_demographics(msa_name, data_root, normalize=True): #20220117, 拆分代码 #20220401, add race/ethnicity minority
    msa_name_full = constants.MSA_NAME_FULL_DICT[msa_name]
    # Obtain CBG populations
    # Load CBG ids for the MSA
    cbg_ids_msa = pd.read_csv(os.path.join(data_root,msa_name,'%s_cbg_ids.csv' % msa_name_full)) 
    cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
    
    # CBG age (elder ratio) and size (population)
    cbg_sizes, cbg_sizes_original, cbg_elder_ratio = load_cbg_age(data_root, cbg_ids_msa, normalize)
    # CBG income
    cbg_household_income = load_cbg_income(data_root, cbg_ids_msa, normalize)
    # CBG essential worker ratio
    cbg_ew_ratio = load_cbg_occupation(data_root, cbg_ids_msa, cbg_sizes_original, normalize)
    # CBG minority ratio #20220401
    cbg_minority_ratio = load_cbg_minority(data_root, cbg_ids_msa, cbg_sizes_original, normalize) 

    # Reshape #20220127
    cbg_sizes = cbg_sizes.reshape(-1,1) 
    cbg_elder_ratio = cbg_elder_ratio.reshape(-1,1)
    cbg_household_income = cbg_household_income.reshape(-1,1)
    cbg_ew_ratio = cbg_ew_ratio.reshape(-1,1)
    cbg_minority_ratio = cbg_minority_ratio.reshape(-1,1)

    return cbg_sizes, cbg_elder_ratio, cbg_household_income, cbg_ew_ratio, cbg_minority_ratio


def load_data(dataset, msa_name, mob_data_root, output_root, normalize=True, rel_result=True): #20220401改
    if(rel_result): print('rel_result=True')
    else: print('rel_result=False')
    if('safegraph' in dataset): 
        print('Loading {} dataset...'.format(dataset))

        # adj
        adj, num_cbgs = load_adj(msa_name, mob_data_root, output_root)

        # node_feats
        # cbg population and other demographic features
        cbg_sizes, cbg_elder_ratio, cbg_household_income, cbg_ew_ratio, cbg_minority_ratio = load_cbg_demographics(msa_name, mob_data_root, normalize)
        
        if(normalize): # Data normalization #20220128
            scaler = preprocessing.StandardScaler()
            cbg_sizes = scaler.fit_transform(cbg_sizes)
            cbg_elder_ratio = scaler.fit_transform(cbg_elder_ratio)
            cbg_household_income = scaler.fit_transform(cbg_household_income)
            cbg_ew_ratio = scaler.fit_transform(cbg_ew_ratio)
            cbg_minority_ratio = scaler.fit_transform(cbg_minority_ratio) #20220401

        node_feats = np.zeros((num_cbgs, 5))
        node_feats[:,0] = cbg_sizes.reshape(1,-1)
        node_feats[:,1] = cbg_elder_ratio.reshape(1,-1)
        node_feats[:,2] = cbg_household_income.reshape(1,-1)
        node_feats[:,3] = cbg_ew_ratio.reshape(1,-1)
        node_feats[:,4] = cbg_minority_ratio.reshape(1,-1) #20220401

        node_feats = torch.FloatTensor(node_feats)
        return adj, node_feats

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


def data_loader(node_feats, graph_labels, idx_train, idx_val, idx_test, batch_size, kfold=False, quicktest=False): #20220127
    # Divide data into train/val/test datasets; Wrap data into DataLoader
    if(quicktest):
        batch_size = 2
        idx_train = idx_train[:batch_size*4]
        idx_val = idx_val[:batch_size]
        idx_test = idx_test[:batch_size]

    train_dataset = torch.utils.data.TensorDataset(node_feats[idx_train,:,:],graph_labels[idx_train])
    val_dataset = torch.utils.data.TensorDataset(node_feats[idx_val,:,:],graph_labels[idx_val])
    test_dataset = torch.utils.data.TensorDataset(node_feats[idx_test,:,:],graph_labels[idx_test])

    if(kfold): #20220202
        train_val_dataset = ConcatDataset([train_dataset, val_dataset])
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False) 
        return train_val_dataset, test_loader 
    else:   
        train_loader = DataLoader(
            train_dataset, #train_dataset.dataset,
            batch_size=batch_size,
            shuffle=True)
        val_loader = DataLoader(
            val_dataset, #val_dataset.dataset,
            batch_size=batch_size,
            shuffle=False) #shuffle=True)
        test_loader = DataLoader(
            test_dataset,#test_dataset.dataset,
            batch_size=batch_size,
            shuffle=False) #shuffle=True)

        return train_loader, val_loader, test_loader

def save_checkpoint_state(model_state_dict, epoch, optimizer_state_dict, scheduler_state_dict, savepath): #20220203
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'scheduler_state_dict': scheduler_state_dict,
        }
    torch.save(checkpoint,savepath)

def get_checkpoint_state(path,model,optimizer,scheduler): #20220203
     # 恢复上次的训练状态
    print("Resume from checkpoint...")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch=checkpoint['epoch']

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('Sucessfully recover from the last state') #logger.info
    return model,epoch,optimizer,scheduler


class ReplayBuffer: #20220206
    def __init__(self, capacity):
        self.replay_buffer = dict() #20220205
        self.count = 0
        self.capacity = capacity
        self.min_reward = np.inf
        self.min_reward_idx = 0
        self.max_reward = -np.inf #20220329
        self.max_reward_idx = 0 #20220329

    def store_transition(self, vac_flag, reward):
        vac_idx_list = torch.nonzero(vac_flag.squeeze()).squeeze().tolist()
        #self.replay_buffer.append(np.array([a, r])) #original
        self.replay_buffer[self.count] = [vac_idx_list, reward] #20220205
        if(reward<self.min_reward):
            self.min_reward = reward
            self.min_reward_idx = self.count
        elif(reward>self.max_reward): #20220329
            self.max_reward = reward
            self.max_reward_idx = self.count 
        self.count += 1
        # TODO: 补一个renew，只保存reward最高的
        #if(self.count>self.capacity):
        #    self.replay_buffer.pop(self.min_reward_idx)

    #def get_actions(self):
    #    return np.vstack(np.vstack(self.replay_buffer)[:, 0])
    #def get_reward(self, i):
    #    return np.vstack(self.replay_buffer)[i, 1]

    def clear(self):
        self.replay_buffer = []
        self.count = 0

    def get_action_and_reward(self): #20220205
        random_idx = np.random.randint(0,self.count) # 采样一个序号
        vac_idx_list = self.replay_buffer[random_idx][0]
        reward = self.replay_buffer[random_idx][1]
        return vac_idx_list, reward
                
    def get_log_prob(self, model, vac_idx_list, gen_node_feats, adj): #20220205
        cbg_scores = model(gen_node_feats, adj) #F.softmax(cbg_scores,dim=0).squeeze().tolist() 
        cbg_sampler = Categorical(cbg_scores.squeeze())
        total_log_probs = 0
        for action in vac_idx_list:
            total_log_probs += cbg_sampler.log_prob(torch.tensor([action]).cuda()) #这个点的log prob
        return total_log_probs