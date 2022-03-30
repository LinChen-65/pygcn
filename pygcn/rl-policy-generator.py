# python rl-policy-generator.py --msa_name SanFrancisco  --rel_result True --epochs 100

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
import torch.optim as optim
import random
import datetime
from torch.distributions import Categorical
import multiprocessing

import time
import pdb

sys.path.append(os.path.join(os.getcwd(), '../gt-generator'))
import constants
import functions
import disease_model_test as disease_model #import disease_model


# 限制显卡使用
#os.environ["CUDA_VISIBLE_DEVICES"] = "2" #"1"
torch.cuda.set_device(0) #1 #nvidia-smi

############################################################################################
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
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
#20220118
parser.add_argument('--normalize', default = True,
                    help='Whether normalize node features or not.')
parser.add_argument('--rel_result', default = False, action='store_true',
                    help='Whether retrieve results relative to no_vac.')
#20220123
parser.add_argument('--prefix', default= '/home', 
                    help='Prefix of data root. /home for rl4, /data for dl3, /data4 for dl2.')
#20220127
parser.add_argument('--trained_evaluator_folder', default= 'chenlin/pygcn/pygcn/trained_model', 
                    help='Folder to reload trained evaluator model.')
# 20220129
parser.add_argument('--vaccination_ratio', default=0.01, #0.02
                    help='Vaccination ratio (w.r.t. total population).')
parser.add_argument('--vaccination_time', default=0, #31
                    help='Time to distribute vaccines.')  
parser.add_argument('--NN', type=int,
                    help='Number of CBGs to receive vaccines.')
# 20220203
parser.add_argument('--quicktest', default= False, action='store_true',
                    help='If true, perform only 2 simulations in traditional_evaluate(); else 40.')
# 20220204
parser.add_argument('--epoch_width', default=1000, type=int, 
                    help='Num of samples in an epoch.')
parser.add_argument('--model_save_folder', default= 'chenlin/pygcn/pygcn/trained_model', 
                    help='Folder to save trained model.')
# 20220205
parser.add_argument('--simulation_cache_filename', default='chenlin/pygcn/pygcn/simulation_cache_proportional_temp.pkl',
                    help='File to save traditional_simulate results.')
parser.add_argument('--replay_width', type=int, default=2,
                    help='Num of experienced actions to be replayed.')
parser.add_argument('--replay_buffer_capacity',type=int,default=200,
                    help='Maximum number of vaccine policy to be stored in replay buffer.')
# 20220206
parser.add_argument('--simulation_cache_folder', default='chenlin/pygcn/pygcn',
                    help='Folder to save traditional_simulate results.')
parser.add_argument('--save_checkpoint', default=False, action='store_true',
                    help='If true, save best checkpoint and final model to .pt file.')
# 20220327
parser.add_argument('--ema_decay', default=0.8,
                    help='Exponential decay factor for ema_baseline (\'critic\').')
parser.add_argument('--proportional', default=True,  #参考gt-gen-vac-fixed-num-cbgs-crossgroup-safedistance.py
                    help='If true, divide vaccines proportional to cbg populations.')  

args = parser.parse_args()
# Check important parameters
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda: torch.cuda.manual_seed(args.seed)
print('args.rel_result: ', args.rel_result)
print('args.quicktest: ', args.quicktest)
print('args.epochs: ', args.epochs)
print('args.epoch_width: ', args.epoch_width)
print('args.replay_width: ', args.replay_width)
print('args.save_checkpoint: ', args.save_checkpoint)
print('args.proportional: ', args.proportional) 


evaluator_path = os.path.join(args.prefix, args.trained_evaluator_folder, 'total_cases_of_100epochs_20220203.pt')
print('evaluator_path: ', evaluator_path)

today = str(datetime.date.today()).replace('-','') # yyyy-mm-dd -> yyyymmdd
print('today: ', today)

checkpoint_save_path = os.path.join(args.prefix, args.model_save_folder, f'checkpoint_generator_maxreward_{today}.pt')
print('checkpoint_save_path: ', checkpoint_save_path)
simulation_cache_save_path = os.path.join(args.prefix, args.simulation_cache_filename)
print('simulation_cache_save_path: ', simulation_cache_save_path)

cache_dict = multiprocessing.Manager().dict() 

# Load simulation cache to accelarate training 
dict_path_list = ['simulation_cache_proportional_combined.pkl', # proportional
                  'simulation_cache_proportional_temp.pkl', # proportional
                  'simulation_cache_proportional_temp_rl4.pkl'
                  #'simulation_cache_combined.pkl', # not proportional
                  #'simulation_cache_temp.pkl' # not proportional
                  ] 

combined_dict = dict()
for dict_path in dict_path_list:
    if(os.path.exists(dict_path)):
        with open(os.path.join(args.prefix, args.simulation_cache_folder,dict_path), 'rb') as f:
            new_dict = pickle.load(f)
        combined_dict = {**combined_dict,**new_dict}
        print(f'len(new_dict): {len(new_dict)}')
        print(f'len(combined_dict): {len(combined_dict)}')
with open(os.path.join(args.prefix, 'chenlin/pygcn/pygcn/simulation_cache_proportional_combined.pkl'), 'wb') as f:
    pickle.dump(combined_dict, f)


###############################################################################
# Load traditional simulator

epic_data_root = f'{args.prefix}/chenlin/COVID-19/Data'
mob_data_root = f'{args.prefix}/chenlin/COVID-19/Data' #Path to mobility data.

PROTECTION_RATE = 1
EXECUTION_RATIO = 1
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[args.msa_name]

# Random Seed
if(args.quicktest): NUM_SEEDS = 5 #2
else: NUM_SEEDS = 40 
print('NUM_SEEDS: ', NUM_SEEDS)
STARTING_SEED = range(NUM_SEEDS)
# Load POI-CBG visiting matrices
f = open(os.path.join(epic_data_root, args.msa_name, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()
# Load precomputed parameters to adjust(clip) POI dwell times
d = pd.read_csv(os.path.join(epic_data_root,args.msa_name, 'parameters_%s.csv' % args.msa_name)) 
MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
all_hours = functions.list_hours_in_range(MIN_DATETIME, MAX_DATETIME)
poi_areas = d['feet'].values#面积
poi_dwell_times = d['median'].values#平均逗留时间
poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
del d

# Load CBG ids for the MSA
cbg_ids_msa = pd.read_csv(os.path.join(epic_data_root,args.msa_name,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
num_cbgs = len(cbg_ids_msa)
# Mapping from cbg_ids to columns in hourly visiting matrices
cbgs_to_idxs = dict(zip(cbg_ids_msa['census_block_group'].values, range(num_cbgs)))
x = {}
for i in cbgs_to_idxs: 
    x[str(i)] = cbgs_to_idxs[i]
idxs_msa_all = list(x.values())

# Load SafeGraph data to obtain CBG sizes (i.e., populations)
filepath = os.path.join(epic_data_root,"safegraph_open_census_data/data/cbg_b01.csv")
cbg_agesex = pd.read_csv(filepath)
# Extract CBGs belonging to the MSA - https://covid-mobility.stanford.edu//datasets/
cbg_age_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
del cbg_agesex
# Rename
cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
# Deal with NaN values
cbg_age_msa.fillna(0,inplace=True)
# Deal with CBGs with 0 populations
cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
# Obtain cbg sizes (populations)
cbg_sizes = cbg_age_msa['Sum'].values
cbg_sizes = np.array(cbg_sizes,dtype='int32')
columns_of_interest = ['census_block_group','Sum'] 
cbg_age_msa = cbg_age_msa[columns_of_interest].copy()
#del cbg_age_msa
'''
ct_msa = pd.DataFrame(columns=['census_tract','Sum']) #20220330
ct_to_cbg_dict = dict() #20220330
for i in range(len(cbg_age_msa)):
    this_cbg = str(cbg_age_msa.loc[i, 'census_block_group'])
    this_ct = this_cbg[:-1]
    if(this_ct in list(ct_to_cbg_dict.keys())):
        ct_to_cbg_dict[this_ct].append(this_cbg)
        #ct_msa[ct_msa['census_tract']==this_ct]['Sum'] += cbg_age_msa.loc[i,'Sum']
        idx=ct_msa[ct_msa['census_tract']==this_ct].index
        ct_msa.loc[idx[0],'Sum'] += cbg_age_msa.loc[i,'Sum']
    else:
        ct_to_cbg_dict[this_ct] = [this_cbg]
        if(i==0):
            ct_msa.loc[0] = [this_ct, cbg_age_msa.loc[i,'Sum']]
        else:
            ct_msa.loc[ct_msa.index.max() + 1] = [this_ct, cbg_age_msa.loc[i,'Sum']]
pdb.set_trace() 
'''

# Load and scale age-aware CBG-specific attack/death rates (original)
cbg_death_rates_original = np.loadtxt(os.path.join(epic_data_root, args.msa_name, 'cbg_death_rates_original_'+args.msa_name))
cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)
# The scaling factors are set according to a grid search
attack_scale = 1 # Fix attack_scale
cbg_attack_rates_scaled = cbg_attack_rates_original * attack_scale
cbg_death_rates_scaled = cbg_death_rates_original * constants.death_scale_dict[args.msa_name]
# Vaccine acceptance
vaccine_acceptance = np.ones(len(cbg_sizes)) # full acceptance
# Calculate number of available vaccines, number of vaccines each cbg can have
#num_vaccines = cbg_sizes.sum() * args.vaccination_ratio / args.NN
#print('Num of vaccines per CBG: ',num_vaccines)
    
############################################################################################
# Functions

# Traditional simulator: final judge
def run_simulation(starting_seed, num_seeds, vaccination_vector, vaccine_acceptance,protection_rate=1):
    m = disease_model.Model(starting_seed=starting_seed,
                            num_seeds=num_seeds,
                            debug=False,clip_poisson_approximation=True,ipf_final_match='poi',ipf_num_iter=100)

    m.init_exogenous_variables(poi_areas=poi_areas,
                               poi_dwell_time_correction_factors=poi_dwell_time_correction_factors,
                               cbg_sizes=cbg_sizes,
                               poi_cbg_visits_list=poi_cbg_visits_list,
                               all_hours=all_hours,
                               p_sick_at_t0=constants.parameters_dict[args.msa_name][0],
                               vaccination_time=24*args.vaccination_time, # when to apply vaccination (which hour)
                               vaccination_vector = vaccination_vector,
                               vaccine_acceptance=vaccine_acceptance,
                               protection_rate = protection_rate,
                               home_beta=constants.parameters_dict[args.msa_name][1],
                               cbg_attack_rates_original = cbg_attack_rates_scaled,
                               cbg_death_rates_original = cbg_death_rates_scaled,
                               poi_psi=constants.parameters_dict[args.msa_name][2],
                               just_compute_r0=False,
                               latency_period=96,  # 4 days
                               infectious_period=84,  # 3.5 days
                               confirmation_rate=.1,
                               confirmation_lag=168,  # 7 days
                               death_lag=432
                               )

    m.init_endogenous_variables()

    #T1,L_1,I_1,R_1,C2,D2,total_affected, history_C2, history_D2, total_affected_each_cbg = m.simulate_disease_spread(no_print=True)    
    #return history_C2, history_D2
    final_cases, final_deaths = m.simulate_disease_spread(no_print=True, store_history=False) #20220327
    return final_cases, final_deaths #20220327


def traditional_evaluate(vac_flag):
    # Construct vaccination vector
    #vaccination_vector = np.zeros(len(cbg_sizes)) #严格平均分
    #vaccination_vector[torch.where(vac_flag.squeeze()!=0)[0].cpu().numpy()] = num_vaccines #严格平均分
    target_idxs = torch.nonzero(vac_flag).cpu().squeeze().numpy() #20220327
    vaccination_vector = functions.vaccine_distribution_fixed_nn(cbg_table=cbg_age_msa,  #20220327
                                                                vaccination_ratio=args.vaccination_ratio, 
                                                                nn=args.NN, 
                                                                proportional=args.proportional, 
                                                                target_idxs=target_idxs
                                                                )

    #20220327
    final_cases_cbg, final_deaths_cbg = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                vaccination_vector=vaccination_vector,
                                                vaccine_acceptance=vaccine_acceptance,
                                                protection_rate = PROTECTION_RATE)
    # Average across random seeds
    final_cases_cbg = np.mean(final_cases_cbg, axis=0)
    final_cases = final_cases_cbg.sum()
    case_rates = final_cases_cbg/cbg_sizes
    case_rates_std = case_rates.std()
    #final_deaths_cbg = deaths_cbg[-1,:] #20220118
    #death_rates = final_deaths_cbg/cbg_sizes #20220118
    #death_rates_std = death_rates.std() #20220118
    return final_cases, case_rates_std


def reset_vac_flag(vac_flag, vac_idx_list): #20220203
    vac_flag = torch.zeros_like(vac_flag)
    vac_flag[vac_idx_list] = 1
    return vac_flag


def worker(vac_flag,cache_dict,mylock): # Used in multiprocess_traditional_evaluate() #20220204
    """thread worker function"""
    this_key = tuple(vac_flag.squeeze().cpu().numpy())
    if(this_key in cache_dict):
        print('Found in cache_dict') 
        [total_cases, case_rate_std] = cache_dict[this_key]
    elif(this_key in combined_dict):
        print('Found in combined_dict') 
        [total_cases, case_rate_std] = combined_dict[this_key]
    else:
        print('Not found in cache') 
        total_cases, case_rate_std = traditional_evaluate(vac_flag) 
        cache_dict[this_key] = [total_cases, case_rate_std]
        print(len(list(cache_dict.keys())))
    return total_cases



def multiprocess_traditional_evaluate(vac_flag_list,cache_dict): #20220204
    mylock = multiprocessing.Manager().Lock()
    pool = multiprocessing.Pool() 
    temp_list = []
    total_cases_list = []
    for t in range(args.epoch_width): 
        vac_flag = vac_flag_list[t]
        temp_list.append(pool.apply_async(worker, (vac_flag,cache_dict,mylock)))
    pool.close()
    pool.join()
    for t in range(args.epoch_width): 
        total_cases_list.append(temp_list[t].get())

    return total_cases_list


def select_action(model): #20220203
    cbg_scores = model(gen_node_feats, adj) #F.softmax(cbg_scores,dim=0).squeeze().tolist() 
    cbg_sampler = Categorical(cbg_scores.squeeze())
    count = 0
    vac_idx_list = []
    log_probs_list = []
    
    # Construct policy #20220207
    vac_idx_list = torch.multinomial(cbg_scores.squeeze(), args.NN, replacement=False).tolist()
    #pdb.set_trace()
    total_log_probs = 0
    for action in vac_idx_list:
        total_log_probs = total_log_probs + cbg_sampler.log_prob(torch.tensor([action]).cuda())
    model.saved_log_probs.append(total_log_probs)

    '''
    # Construct policy (action)
    loop_i = 0
    while(count<args.NN):
        loop_i += 1
        if(loop_i>args.NN+100000): print(loop_i);pdb.set_trace() #卡住了
        action = cbg_sampler.sample() #每次只采样1个点
        log_probs = cbg_sampler.log_prob(action) #这个点的log prob
        if(action.item() in vac_idx_list): continue #采到重复点，丢弃
        vac_idx_list.append(action.item())
        log_probs_list.append(log_probs)
        count += 1
    # Record probability
    for i in range(len(log_probs_list)): #从单个点的log prob得到组合的log prob，取log后累加等价于累乘后取log
        if(i==0): total_log_probs = log_probs_list[i]
        else: total_log_probs = total_log_probs + log_probs_list[i]
    model.saved_log_probs.append(total_log_probs)
    '''
    
    '''
    cbg_sampler = sampler.WeightedRandomSampler(cbg_scores.squeeze(), args.NN, False)
    vac_idx_list = [idx for idx in cbg_sampler]; #print(vac_idx_list)
    #vac_flag = torch.zeros_like(cbg_scores)
    #vac_flag = reset_vac_flag(vac_flag, vac_idx_list)
    #cum_prod = np.cumprod(cbg_sampler.weights[vac_idx_list].cpu().detach().numpy())[-1]
    #model.saved_log_probs.append(cum_prod) #近似放回抽样
    cum_sum = np.cumsum(cbg_sampler.weights[vac_idx_list].cpu().detach().numpy())[-1]
    model.saved_log_probs.append(cum_sum) #近似放回抽样
    '''
    
    vac_flag = torch.zeros_like(cbg_scores)
    vac_flag = reset_vac_flag(vac_flag, vac_idx_list)
    return vac_flag


def finish_episode(model, max_avg_rewards): #20220203
    #R = 0
    model_loss = []
    rewards = []
    #20220205注释
    for r in model.rewards[::-1]:
        #R = r + 0.99 * R #args.gamma =0.99 #original
        #rewards.insert(0, R) #original
        rewards.insert(0, r)

    rewards = torch.tensor(rewards) #original
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps) #original 
    #rewards = rewards / (rewards.std() + eps) #仅scaling，不减均值 #20220205 还是算了，减吧
    for log_prob, reward in zip(model.saved_log_probs, rewards):
        model_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    
    for i in range(len(model_loss)):
        if(i==0): total_loss = model_loss[i]
        else: total_loss = total_loss + model_loss[i]
    total_loss.backward()

    avg_rewards = 0
    count = 0
    for r in model.rewards[::-1]:
        avg_rewards += r.item()
        count += 1
    avg_rewards /= count
    if((i_episode==0) or (avg_rewards>max_avg_rewards)): # Save checkpoint
        max_avg_rewards = avg_rewards; print('Max average rewards updated.')
        if(args.save_checkpoint): #20220206
            torch.save({
                'epoch': i_episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_rewards': avg_rewards,
                }, checkpoint_save_path)
            print('Checkpoint updated.')
    
    del model.rewards[:]
    del model.saved_log_probs[:]

    optimizer.step()
    #model.replay_buffer.clear() #20220205
    return avg_rewards, max_avg_rewards

############################################################################################
# Load trained PolicyEvaluator #20220127

if hasattr(torch.cuda, 'empty_cache'): 	torch.cuda.empty_cache()
evaluator = torch.load(evaluator_path)
print('evaluator: ', evaluator)
evaluator.eval()

############################################################################################
# Load adj and CBG features

output_root = os.path.join(args.gt_root, args.msa_name)
pretrain_embed_path = os.path.join(args.prefix,'chenlin/code-dynalearn/scripts/figure-6/gt-generator/covid/outputs/node_embeddings_b1.0.npy' )

adj, node_feats = load_data(vac_result_path=None, #20220126
                            dataset=f'safegraph-',
                            msa_name=args.msa_name,
                            mob_data_root=mob_data_root,
                            output_root=output_root,
                            pretrain_embed_path=pretrain_embed_path,
                            normalize=args.normalize,
                            rel_result=args.rel_result,
                            with_vac_flag=False, #20220126
                            ) 
adj = np.array(adj)

# Calculate node centrality
start = time.time(); print('Start graph construction..')
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
deg_centrality = preprocessing.scale(deg_centrality) #robust_scale
clo_centrality = preprocessing.scale(clo_centrality) #robust_scale
bet_centrality = preprocessing.scale(bet_centrality) #robust_scale
#start = time.time(); print('Start centrality computation..')
#deg_centrality = nx.degree_centrality(G_nx);print('Time for deg: ', time.time()-start); start=time.time()
#clo_centrality = nx.closeness_centrality(G_nx);print('Time for clo: ', time.time()-start); start=time.time()
#bet_centrality = nx.betweenness_centrality(G_nx);print('Time for bet: ', time.time()-start); start=time.time()
#print('Finish centrality computation. Time used: ',time.time()-start)

# Calculate average mobility level
mob_level = np.sum(adj, axis=1)
mob_max = np.max(mob_level)
# Normalization
mob_level = preprocessing.scale(mob_level) #20220120 #robust_scale

deg_centrality = deg_centrality.reshape(-1,1)
clo_centrality = clo_centrality.reshape(-1,1)
bet_centrality = bet_centrality.reshape(-1,1)
mob_level = mob_level.reshape(-1,1)

#random_feat = torch.randn(mob_level.shape) #20220203 #test: random feat
#gen_node_feats = np.concatenate((node_feats[:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level,random_feat), axis=1) #20220203
#gen_node_feats = np.concatenate((node_feats, deg_centrality, clo_centrality, bet_centrality, mob_level), axis=1)
gen_node_feats = np.concatenate((node_feats[:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level), axis=1)

dim_touched = gen_node_feats.shape[1] #赋值给config.dim_touched #20220203
print('node_feats.shape: ', node_feats.shape) 
gen_node_feats = np.tile(gen_node_feats,(1,2)) #加回原始feature
print('node_feats.shape: ', gen_node_feats.shape) 

gen_node_feats = torch.Tensor(gen_node_feats)
adj = torch.Tensor(adj)
deg_centrality = torch.Tensor(deg_centrality)
clo_centrality = torch.Tensor(clo_centrality)
bet_centrality = torch.Tensor(bet_centrality)
mob_level = torch.Tensor(mob_level)

############################################################################################
# Model and optimizer

config = Config()
config.dim_touched = dim_touched # Num of feats used to calculate embedding #20220123
config.gcn_nfeat = config.dim_touched # Num of feats used to calculate embedding #20220123
config.gcn_nhid = args.hidden 
config.gcn_nclass = 32 
config.gcn_dropout = args.dropout
config.linear_nin = config.gcn_nclass + (gen_node_feats.shape[1]-config.dim_touched)
config.linear_nhid1 = 32 #100 
config.linear_nhid2 = 32 #100
config.linear_nout = 1 #20220126 
config.NN = args.NN #20220201
config.replay_buffer_capacity = args.replay_buffer_capacity #20220205

#model = get_model(config, 'Generator')
model = get_model(config, 'SoftGenerator'); print(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5, patience=6, min_lr=1e-8, verbose=True) #20220122

random.seed(42)
eps = np.finfo(np.float32).eps.item() #202202013

if args.cuda:
    model.cuda()
    adj = adj.cuda()
    gen_node_feats = gen_node_feats.cuda() #20220114
    node_feats = node_feats.cuda() #20220128
    deg_centrality = deg_centrality.cuda()
    clo_centrality = clo_centrality.cuda() 
    bet_centrality = bet_centrality.cuda() 
    mob_level = mob_level.cuda() 

############################################################################################
# Baseline
# No vac
'''
vac_flag = select_action(model)
vac_flag = torch.zeros_like(vac_flag)
total_cases, case_rate_std = traditional_evaluate(vac_flag) #20220204
no_vac_baseline = total_cases
print('no_vac_baseline； ', no_vac_baseline)
'''
# Baseline: no_vaccination, random
no_vac_baseline = 7425 #NUM_SEEDS=40
random_baseline = 7280 #NUM_SEEDS=40

############################################################################################
# Start training

# Multiprocessing to accelarate traditional simulator #20220204
avg_rewards_list = []
avg_total_cases_list = [] #20220328
max_avg_rewards = 0
start = time.time()
for i_episode in range(args.epochs):
    start_episode = time.time()
    vac_flag_list = []
    for t in range(args.epoch_width):  
        vac_flag = select_action(model)
        vac_flag_list.append(vac_flag.cpu()) 
    total_cases_list = multiprocess_traditional_evaluate(vac_flag_list,cache_dict)
    avg_total_cases_list.append(np.mean(np.array(total_cases_list))) #20220328
    if(i_episode%10==0): print(avg_total_cases_list) #20220328
    max_reward_1 = -np.inf #20220205
    max_reward_idx_1 = 0 #20220205
    max_reward_2 = -np.inf #20220205
    max_reward_idx_2 = 0 #20220205
    if(i_episode==0):  #20220327
        ema_baseline = np.mean(total_cases_list)
    else: 
        ema_baseline = ema_baseline*args.ema_decay + np.mean(total_cases_list)*(1-args.ema_decay)
    ema_baseline = min(ema_baseline, random_baseline)
    print(f'ema_baseline at episode {i_episode}: {ema_baseline}')
    for t in range(args.epoch_width): 
        #reward = no_vac_baseline - total_cases_list[t] #20220204
        #reward = random_baseline - total_cases_list[t] #20220204
        reward = ema_baseline - total_cases_list[t] # moving average baseline #20220327
        model.rewards.append(reward)
        if(reward>max_reward_1): #20220205
            max_reward_2 = max_reward_1
            max_reward_idx_2 = max_reward_idx_1
            max_reward_1 = reward #20220205
            max_reward_idx_1 = t #20220205
    if(max_reward_1!=(-np.inf)):
        model.replay_buffer.store_transition(vac_flag_list[max_reward_idx_1], max_reward_1) #20220205
    if(max_reward_2!=(-np.inf)):
        model.replay_buffer.store_transition(vac_flag_list[max_reward_idx_2], max_reward_2) #20220205
    for t in range(min(args.replay_width,model.replay_buffer.count)): #20220205
        # Sample from replay_buffer
        vac_idx_list, reward = model.replay_buffer.get_action_and_reward()
        total_log_probs = model.replay_buffer.get_log_prob(model, vac_idx_list, gen_node_feats, adj)
        model.rewards.append(reward)
        model.saved_log_probs.append(total_log_probs)

    avg_rewards, max_avg_rewards = finish_episode(model, max_avg_rewards)
    avg_rewards_list.append(avg_rewards)

    print(f'Episode {i_episode}: avg rewards={avg_rewards}.')
    print('Num of cached results: ', len(cache_dict))

    key_list = list(cache_dict.keys())
    value_list = list(cache_dict.values())
    dict_to_store = dict()
    for i in range(len(key_list)):
        dict_to_store[key_list[i]] = value_list[i]
    print(len(list(dict_to_store.keys())))
    print(len(list(dict_to_store.keys()))+len(list(combined_dict.keys())))
    with open(simulation_cache_save_path, 'wb') as f:
        pickle.dump(dict_to_store, f)
    print('cache_dict updated.') #test loading: with open(simulation_cache_save_path, 'rb') as f:saved_dict = pickle.load(f)
    if(i_episode%1==0): print(f'Episode uses time: {time.time()-start_episode}')

# Save final episode model
if(args.save_checkpoint):
    torch.save({'epoch': i_episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_rewards': avg_rewards,}, os.path.join(args.prefix, args.model_save_folder, f'checkpoint_generator_maxreward_episode{args.epochs}_{today}.pt'))

print(f'{args.epochs} episodes uses time: {time.time()-start}')
pdb.set_trace()

# Single processing for gnn predictor
'''
#for i_episode in count(1): #original
for i_episode in range(args.epochs): #20220203
    for t in range(args.epoch_width):  # Don't infinite loop while learning
        vac_flag = select_action(model)
        # Prepare inputs for PolicyEvaluator
        eval_node_feats = torch.cat((node_feats[:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, node_feats[:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, vac_flag), axis=1) #20220201
        eval_node_feats = eval_node_feats.unsqueeze(axis=0)
        if args.cuda: eval_node_feats = eval_node_feats.cuda()
        #reward = -evaluator(eval_node_feats, adj)
        total_cases, case_rate_std = traditional_evaluate(vac_flag) #20220204
        #reward = no_vac_baseline - total_cases #20220204
        reward = random_baseline - total_cases #20220204
        model.rewards.append(reward)
    avg_rewards = finish_episode(model)
    print(f'Episode {i_episode}: avg rewards={avg_rewards}.')
'''

# Inference Method 1: stochastic
# vac_flag = select_action(model)  
# Inference Method 2: deterministic 
'''
cbg_scores = model(gen_node_feats, adj)  
sorted_indices = torch.argsort(cbg_scores,dim=0,descending=True) #20220128 # 返回从大到小的索引
reverse = torch.reciprocal(cbg_scores.detach())
zero = torch.zeros_like(cbg_scores.detach())
topk_mask = torch.where(cbg_scores>cbg_scores[sorted_indices[args.NN]], reverse, zero)
vac_flag = cbg_scores * topk_mask

# Prepare inputs for PolicyEvaluator
eval_node_feats = torch.cat((node_feats[:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, 
                             node_feats[:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, 
                             vac_flag.unsqueeze(1)), axis=1) #20220206
eval_node_feats = eval_node_feats.unsqueeze(axis=0)
if args.cuda: eval_node_feats = eval_node_feats.cuda()
reward = -evaluator(eval_node_feats, adj)
print('reward: ', reward)
pdb.set_trace()
total_cases, case_rate_std = traditional_evaluate(vac_flag) #20220204
print('Traditional evaluated: ', total_cases, case_rate_std)
'''
# Inference Method 3: stochastic, sampling #20220326
for t in range(args.epoch_width): 
    vac_flag = select_action(model) # stochastic
    vac_flag_list.append(vac_flag.cpu()) 
total_cases_list = multiprocess_traditional_evaluate(vac_flag_list,cache_dict)
reward_array = random_baseline - np.array(total_cases_list)
print(np.max(reward_array))
pdb.set_trace()

if(args.save_checkpoint):
    print('If proceed, change model to best checkpoint.')
    pdb.set_trace()
    # Load best model
    model.load_state_dict(torch.load(checkpoint_save_path)['model_state_dict'])
    # Inference Method 2: deterministic 
    '''
    cbg_scores = model(gen_node_feats, adj)
    sorted_indices = torch.argsort(cbg_scores,dim=0,descending=True) #20220128 # 返回从大到小的索引
    reverse = torch.reciprocal(cbg_scores.detach())
    zero = torch.zeros_like(cbg_scores.detach())
    topk_mask = torch.where(cbg_scores>cbg_scores[sorted_indices[args.NN]], reverse, zero)
    vac_flag = cbg_scores * topk_mask
    total_cases, case_rate_std = traditional_evaluate(vac_flag) #20220204
    print('Traditional evaluated: ', total_cases, case_rate_std)
    '''
    # Inference Method 3: stochastic, sampling #20220326
    for t in range(args.epoch_width): 
        vac_flag = select_action(model) # stochastic
        vac_flag_list.append(vac_flag.cpu()) 
    total_cases_list = multiprocess_traditional_evaluate(vac_flag_list,cache_dict)
    reward_array = random_baseline - np.array(total_cases_list)
    print(np.max(reward_array))

pdb.set_trace()