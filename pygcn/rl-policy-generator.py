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
#import torch.nn.functional as F
import torch.optim as optim
import random
import datetime
from torch.distributions import Categorical
#from itertools import count
#from torch.utils.data import sampler
import multiprocessing

import time
import pdb

sys.path.append(os.path.join(os.getcwd(), '../gt-generator'))
import constants
import functions
import disease_model

print('new')
# 限制显卡使用
#os.environ["CUDA_VISIBLE_DEVICES"] = "2" #"1"
torch.cuda.set_device(0) #1 #nvidia-smi

############################################################################################
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

args = parser.parse_args()
# Check important parameters
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print('args.rel_result: ', args.rel_result)
print('args.quicktest: ', args.quicktest)
print('args.epochs: ', args.epochs)
print('args.epoch_width: ', args.epoch_width)

#evaluator_path = os.path.join(args.prefix, args.trained_evaluator_folder, 'total_cases_20220126.pt')
#evaluator_path = os.path.join(args.prefix, args.trained_evaluator_folder, 'total_cases_of_250epochs_20220131.pt')
evaluator_path = os.path.join(args.prefix, args.trained_evaluator_folder, 'total_cases_of_100epochs_20220203.pt')
print('evaluator_path: ', evaluator_path)

today = str(datetime.date.today()).replace('-','') # yyyy-mm-dd -> yyyymmdd
print('today: ', today)
checkpoint_save_path = os.path.join(args.prefix, args.model_save_folder, f'checkpoint_generator_maxreward_{today}.pt')
print('checkpoint_save_path: ', checkpoint_save_path)

###############################################################################
# Load traditional simulator

epic_data_root = f'{args.prefix}/chenlin/COVID-19/Data'
mob_data_root = f'{args.prefix}/chenlin/COVID-19/Data' #Path to mobility data.
MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)

# Vaccination protection rate
PROTECTION_RATE = 1
# Policy execution ratio
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
# No clipping
new_d = d
all_hours = functions.list_hours_in_range(MIN_DATETIME, MAX_DATETIME)
poi_areas = new_d['feet'].values#面积
poi_dwell_times = new_d['median'].values#平均逗留时间
poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
del new_d
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
del cbg_age_msa

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
num_vaccines = cbg_sizes.sum() * args.vaccination_ratio / args.NN
print('Num of vaccines per CBG: ',num_vaccines)
    
############################################################################################
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
                               #vaccination_time=24*31, # when to apply vaccination (which hour)
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

    T1,L_1,I_1,R_1,C2,D2,total_affected, history_C2, history_D2, total_affected_each_cbg = m.simulate_disease_spread(no_print=True)    
    return history_C2, history_D2


def traditional_evaluate(vac_flag):
    start = time.time()
    # Construct vaccination vector
    vaccination_vector = np.zeros(len(cbg_sizes))
    vaccination_vector[torch.where(vac_flag.squeeze()!=0)[0].cpu().numpy()] = num_vaccines
    history_C2, history_D2 = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                            vaccination_vector=vaccination_vector,
                                            vaccine_acceptance=vaccine_acceptance,
                                            protection_rate = PROTECTION_RATE)
    # Average history records across random seeds
    cases_cbg, deaths_cbg,_,_ = functions.average_across_random_seeds(history_C2,history_D2, 
                                                                    num_cbgs, idxs_msa_all, 
                                                                    print_results=False,draw_results=False)

    final_cases_cbg = cases_cbg[-1,:]
    final_cases = final_cases_cbg.sum()
    case_rates = final_cases_cbg/cbg_sizes
    case_rates_std = case_rates.std()
    #final_deaths_cbg = deaths_cbg[-1,:] #20220118
    #death_rates = final_deaths_cbg/cbg_sizes #20220118
    #death_rates_std = death_rates.std() #20220118
    end = time.time()
    #print('Used time: ', end-start)
    return final_cases, case_rates_std


def reset_vac_flag(vac_flag, vac_idx_list): #20220203
    vac_flag = torch.zeros_like(vac_flag)
    vac_flag[vac_idx_list] = 1
    return vac_flag


def worker(vac_flag,t): # Used in multiprocess_traditional_evaluate() #20220204
    """thread worker function"""
    #print(f'worker {t}')
    total_cases, case_rate_std = traditional_evaluate(vac_flag) 
    return total_cases

def multiprocess_traditional_evaluate(vac_flag_list): #20220204
    pool = multiprocessing.Pool()   #processes=10 #processes=args.epoch_width
    temp_list = []
    total_cases_list = []
    for t in range(args.epoch_width): 
        vac_flag = vac_flag_list[t]
        #total_cases_list.append(pool.apply_async(worker, (vac_flag,t,)).get())
        temp_list.append(pool.apply_async(worker, (vac_flag,t,)))
    pool.close()
    pool.join()
    for t in range(args.epoch_width): 
        total_cases_list.append(temp_list[t].get())
    return total_cases_list

# Multiprocessing test
'''
vac_flag_list = []
for i in range(10):
    vac_flag = torch.zeros(2943,1)
    vac_flag_list.append(vac_flag)

pool = multiprocessing.Pool(processes=6)
result = dict()
for i in range(10):
    vac_flag = vac_flag_list[i]
    result[i] = (pool.apply_async(worker, (vac_flag,)).get())
pool.close()
pool.join()
print('Done.')
'''


# Baseline:no_vaccination
'''
vaccination_vector_no_vaccination = np.zeros(len(cbg_sizes))
# Run simulations
history_C2_no_vaccination, history_D2_no_vaccination = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                            vaccination_vector=vaccination_vector_no_vaccination,
                                                            vaccine_acceptance=vaccine_acceptance,
                                                            protection_rate = PROTECTION_RATE)
# Average history records across random seeds
cases_cbg_no_vaccination, deaths_cbg_no_vaccination,_,_ = functions.average_across_random_seeds(history_C2_no_vaccination,history_D2_no_vaccination, 
                                                                                        num_cbgs, idxs_msa_all, 
                                                                                        print_results=False,draw_results=False)

final_cases_cbg_no_vaccination = cases_cbg_no_vaccination[-1,:]
final_cases_no_vaccination = final_cases_cbg_no_vaccination.sum()
case_rates_no_vaccination = final_cases_cbg_no_vaccination/cbg_sizes
case_rates_std_no_vaccination = case_rates_no_vaccination.std()
print(final_cases_no_vaccination, case_rates_std_no_vaccination)
#final_deaths_cbg_no_vaccination = deaths_cbg_no_vaccination[-1,:] #20220118
#death_rates_no_vaccination = final_deaths_cbg_no_vaccination/cbg_sizes #20220118
#death_rates_std_no_vaccination = death_rates_no_vaccination.std() #20220118
'''
############################################################################################
# Load trained PolicyEvaluator #20220127

if hasattr(torch.cuda, 'empty_cache'):
	torch.cuda.empty_cache()
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

#model = get_model(config, 'Generator')
model = get_model(config, 'SoftGenerator')
print(model)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
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


def select_action(policy): #20220203
    '''
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()
    '''
    
    cbg_scores = model(gen_node_feats, adj) #F.softmax(cbg_scores,dim=0).squeeze().tolist() 
    cbg_sampler = Categorical(cbg_scores.squeeze())
    count = 0
    vac_idx_list = []
    log_probs_list = []
    
    # Construct policy (action)
    loop_i = 0
    while(count<args.NN):
        loop_i += 1
        if(loop_i>args.NN+100000): print(loop_i);pdb.set_trace()
        action = cbg_sampler.sample() #每次只采样1个点
        log_probs = cbg_sampler.log_prob(action)
        if(action.item() in vac_idx_list):
            continue
        vac_idx_list.append(action.item())
        log_probs_list.append(log_probs)
        count += 1
    # Record probability
    for i in range(len(log_probs_list)):
        if(i==0): 
            total_log_probs = log_probs_list[i]
        else:
            total_log_probs = total_log_probs + log_probs_list[i]
    policy.saved_log_probs.append(total_log_probs)
    
    '''
    cbg_sampler = sampler.WeightedRandomSampler(cbg_scores.squeeze(), args.NN, False)
    vac_idx_list = [idx for idx in cbg_sampler]; #print(vac_idx_list)
    #vac_flag = torch.zeros_like(cbg_scores)
    #vac_flag = reset_vac_flag(vac_flag, vac_idx_list)
    
    #cum_prod = np.cumprod(cbg_sampler.weights[vac_idx_list].cpu().detach().numpy())[-1]
    #policy.saved_log_probs.append(cum_prod) #近似放回抽样
    cum_sum = np.cumsum(cbg_sampler.weights[vac_idx_list].cpu().detach().numpy())[-1]
    policy.saved_log_probs.append(cum_sum) #近似放回抽样
    '''
    
    vac_flag = torch.zeros_like(cbg_scores)
    vac_flag = reset_vac_flag(vac_flag, vac_idx_list)
    return vac_flag

#max_avg_rewards = -np.inf
def finish_episode(policy,i_episode,max_avg_rewards): #20220203
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        #R = r + 0.99 * R #args.gamma =0.99 #original
        #rewards.insert(0, R) #original
        rewards.insert(0, r)

    rewards = torch.tensor(rewards) #original
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps) #original
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    
    for i in range(len(policy_loss)):
        if(i==0): total_loss = policy_loss[i]
        else: total_loss = total_loss + policy_loss[i]
    total_loss.backward()
    #policy_loss = torch.cat(policy_loss).sum() #original #报错: RuntimeError: zero-dimensional tensor (at position 0) cannot be concatenated
    #print('policy_loss:', policy_loss)
    #policy_loss.backward()
    #avg_rewards = np.mean(np.array(policy.rewards[:])) #original
    avg_rewards = 0
    count = 0
    for r in policy.rewards[::-1]:
        avg_rewards += r.item()
        count += 1
    avg_rewards /= count
    if((i_episode==0) or (avg_rewards>max_avg_rewards)): # Save checkpoint
        max_avg_rewards = avg_rewards
        torch.save({
            'epoch': i_episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_rewards': avg_rewards,
            }, checkpoint_save_path)
        print('Max average rewards updated. Checkpoint updated.')

    del policy.rewards[:]
    del policy.saved_log_probs[:]

    optimizer.step()

    #return avg_rewards
    return avg_rewards,max_avg_rewards

# No vac
'''
vac_flag = select_action(model)
vac_flag = torch.zeros_like(vac_flag)
total_cases, case_rate_std = traditional_evaluate(vac_flag) #20220204
no_vac_baseline = total_cases
print('no_vac_baseline； ', no_vac_baseline)
'''

# Multiprocessing to accelarate traditional simulator #20220204 
no_vac_baseline = 7425 #NUM_SEEDS=40
random_baseline = 7280 #NUM_SEEDS=40
avg_rewards_list = []
max_avg_rewards = 0
for i_episode in range(args.epochs):
    start = time.time()
    vac_flag_list = []
    for t in range(args.epoch_width):  
        vac_flag = select_action(model)
        #vac_flag_list.append(vac_flag) 
        vac_flag_list.append(vac_flag.cpu()) 
    total_cases_list = multiprocess_traditional_evaluate(vac_flag_list)
    for t in range(args.epoch_width): 
        #reward = no_vac_baseline - total_cases_list[t] #20220204
        reward = random_baseline - total_cases_list[t] #20220204
        model.rewards.append(reward)
    avg_rewards, max_avg_rewards = finish_episode(model,i_episode,max_avg_rewards)
    avg_rewards_list.append(avg_rewards)
    '''
    if((i_episode==0) or (avg_rewards>max_avg_rewards)): # Save checkpoint
        max_avg_rewards = avg_rewards
        torch.save({
            'epoch': i_episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_rewards': avg_rewards,
            }, checkpoint_save_path)
        print('Max average rewards updated. Checkpoint updated.')
    '''

    print(f'Episode {i_episode}: avg rewards={avg_rewards}.')
    print(f'Episode uses time: {time.time()-start}')
pdb.set_trace()

# Single processing for gnn predictor
'''
no_vac_baseline = 7425
random_baseline = 7280
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

#vac_flag = select_action(model)
cbg_scores = model(gen_node_feats, adj)
sorted_indices = torch.argsort(cbg_scores,dim=0,descending=True) #20220128 # 返回从大到小的索引
reverse = torch.reciprocal(cbg_scores.detach())
zero = torch.zeros_like(cbg_scores.detach())
topk_mask = torch.where(cbg_scores>cbg_scores[sorted_indices[args.NN]], reverse, zero)
vac_flag = cbg_scores * topk_mask
# Prepare inputs for PolicyEvaluator
eval_node_feats = torch.cat((node_feats[:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, node_feats[:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, vac_flag), axis=1) #20220201
eval_node_feats = eval_node_feats.unsqueeze(axis=0)
if args.cuda: eval_node_feats = eval_node_feats.cuda()
reward = -evaluator(eval_node_feats, adj)
print('reward: ', reward)
pdb.set_trace()
total_cases, case_rate_std = traditional_evaluate(vac_flag) #20220204
print('Traditional evaluated: ', total_cases, case_rate_std)

print('If proceed, change model to best checkpoint.')
pdb.set_trace()
# Load best model
model.load_state_dict(torch.load(checkpoint_save_path)['model_state_dict'])
cbg_scores = model(gen_node_feats, adj)
sorted_indices = torch.argsort(cbg_scores,dim=0,descending=True) #20220128 # 返回从大到小的索引
reverse = torch.reciprocal(cbg_scores.detach())
zero = torch.zeros_like(cbg_scores.detach())
topk_mask = torch.where(cbg_scores>cbg_scores[sorted_indices[args.NN]], reverse, zero)
vac_flag = cbg_scores * topk_mask
total_cases, case_rate_std = traditional_evaluate(vac_flag) #20220204
print('Traditional evaluated: ', total_cases, case_rate_std)

###########################################################################
# Training
#no_vac_cases = 7425
traditional_evaluated_cases = []
outcome_list = []
current_best = 0 #cbg_sizes.sum()
policy_list = []
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()

    # Obtain indices of top-NN CBGs to be vaccinated
    vac_flag = model(gen_node_feats, adj) #policy = model(gen_node_feats, adj)
    new_policy = torch.where(vac_flag.squeeze()!=0)[0].tolist() #20220203
    print('New policy: ', new_policy)
    if(new_policy not in policy_list): #20220203
        policy_list.append(new_policy) #20220203
    
    # Prepare inputs for PolicyEvaluator
    #eval_node_feats = torch.cat((node_feats, vac_flag, deg_centrality, clo_centrality, bet_centrality, mob_level, vac_flag, vac_flag), axis=1) #20220128 
    #eval_node_feats = torch.cat((node_feats[:,:4], vac_flag, deg_centrality, clo_centrality, bet_centrality, mob_level, vac_flag, vac_flag), axis=1) #20220128 
    eval_node_feats = torch.cat((node_feats[:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, node_feats[:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, vac_flag), axis=1) #20220201
    eval_node_feats = eval_node_feats.unsqueeze(axis=0)
    if args.cuda: eval_node_feats = eval_node_feats.cuda()

    # Run evaluation
    # Evaluate with traditional simulator (slow)
    '''
    total_cases, case_std = traditional_evaluate(vac_flag)
    print(f'total_cases: {total_cases}, case_std: {case_std}.')
    traditional_evaluated_cases.append(total_cases)
    '''
    
    # Evaluate with trained GNN predictor (fast but less accurate)
    #outcome = evaluator(eval_node_feats, adj); print('outcome: ', outcome)
    #train_loss = -(evaluator(eval_node_feats, adj)) * (1+(-(evaluator(eval_node_feats, adj)))/current_best)#-outcome * (1+(-(outcome.detach()))/current_best)
    #train_loss = -outcome * (1+(-outcome)/current_best) #直接这么写会报错:RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
    #train_loss = -outcome * (1+(-(outcome.detach()))/current_best)
    train_loss = evaluator(eval_node_feats, adj)
    print('train_loss: ', train_loss)

    # Backprop
    with torch.autograd.set_detect_anomaly(True):
        train_loss.backward(retain_graph=True) #loss.backward()
    
   # print('traditional_evaluated_cases: ', traditional_evaluated_cases, ', current_best: ', current_best)
    #current_best = np.array(outcome_list).min()
    #current_best = min(current_best, outcome)

    # Optimize
    optimizer.step()
    scheduler.step(train_loss)

#print('traditional_evaluated_cases: ', traditional_evaluated_cases)
num_policies = len(policy_list); print('Num of different policies:', num_policies)
final_cases_list = []
for i in range(num_policies):
    policy = policy_list[i]; print('policy: ', policy)
    vac_flag = reset_vac_flag(vac_flag, policy)
    final_cases, case_rates_std = traditional_evaluate(vac_flag)
    print(final_cases, case_rates_std)
    final_cases_list.append(final_cases)

print('final_cases_list: ', final_cases_list)
pdb.set_trace()
