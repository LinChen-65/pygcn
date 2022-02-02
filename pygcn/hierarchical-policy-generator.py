# python hierarchical-policy-generator.py --msa_name SanFrancisco  --mob_data_root '/home/chenlin/COVID-19/Data' --rel_result True --epochs 100

from tokenize import group
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

import time
import pdb

sys.path.append(os.path.join(os.getcwd(), '../gt-generator'))
import constants
import functions
import disease_model


# 限制显卡使用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
parser.add_argument('--mob_data_root', default = '/data/chenlin/COVID-19/Data',
                    help='Path to mobility data.')
#20220118
parser.add_argument('--normalize', default = True,
                    help='Whether normalize node features or not.')
parser.add_argument('--rel_result', default = False, action='store_true',
                    help='Whether retrieve results relative to no_vac.')
#20220123
parser.add_argument('--prefix', default= '/home', 
                    help='Prefix of data root. /home for rl4, /data for dl3.')
#20220127
parser.add_argument('--trained_evaluator_folder', default= 'chenlin/pygcn/pygcn/trained_model', 
                    help='Folder to reload trained evaluator model.')
#20220129
parser.add_argument('--vaccination_ratio', default=0.01, #0.02
                    help='Vaccination ratio (w.r.t. total population).')
parser.add_argument('--vaccination_time', default=0, #31
                    help='Time to distribute vaccines.')                    
parser.add_argument('--NN', type=int, default=70,
                    help='Num of CBGs to receive vaccines.')


args = parser.parse_args()
# Check important parameters
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print('args.rel_result: ', args.rel_result)

#evaluator_path = os.path.join(args.prefix, args.trained_evaluator_folder, 'total_cases_20220126.pt')
evaluator_path = os.path.join(args.prefix, args.trained_evaluator_folder, 'total_cases_of_250epochs_20220131.pt')
print('evaluator_path: ', evaluator_path)


###############################################################################
# Load traditional simulator

epic_data_root = '/data/chenlin/COVID-19/Data'
MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
NUM_GROUPS = 3 #5

MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[args.msa_name]
# Random Seed
NUM_SEEDS = 40; print('NUM_SEEDS: ', NUM_SEEDS)
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

# Load SafeGraph data to obtain CBG sizes and other demographic features(i.e., populations) #20220129
df_filename = os.path.join(args.gt_root, args.msa_name, 'demographic_dataframe_%s_%sgroupsperfeat.csv' % (args.msa_name,NUM_GROUPS))
data_df = pd.read_csv(df_filename)
cbg_sizes = np.array(data_df['Sum'])
hybrid_idx = np.array(data_df['Hybrid_Group']) 
vac_flag_pos = np.zeros(cbg_sizes.shape)

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


def traditional_evaluate(vac_flag, is_torch=False):
    if(is_torch):
        # Construct vaccination vector
        vaccination_vector = torch.zeros(len(cbg_sizes))
        vaccination_vector[torch.where(vac_flag.squeeze()!=0)[0]] = num_vaccines
        pdb.set_trace()#没想好
    else:
        # Construct vaccination vector
        vaccination_vector = np.zeros(len(cbg_sizes))
        vaccination_vector[torch.where(vac_flag.squeeze()!=0)[0].cpu().numpy()] = num_vaccines
        history_C2, history_D2 = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                vaccination_vector=vaccination_vector,
                                                vaccine_acceptance=vaccine_acceptance)
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
    return final_cases, case_rates_std

'''
# Baseline:no_vaccination
vaccination_vector_no_vaccination = np.zeros(len(cbg_sizes))
# Run simulations
history_C2_no_vaccination, history_D2_no_vaccination = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                            vaccination_vector=vaccination_vector_no_vaccination,
                                                            vaccine_acceptance=vaccine_acceptance)
# Average history records across random seeds
cases_cbg_no_vaccination, deaths_cbg_no_vaccination,_,_ = functions.average_across_random_seeds(history_C2_no_vaccination,history_D2_no_vaccination, 
                                                                                        num_cbgs, idxs_msa_all, 
                                                                                        print_results=False,draw_results=False)

final_cases_cbg_no_vaccination = cases_cbg_no_vaccination[-1,:]
final_cases_no_vaccination = final_cases_cbg_no_vaccination.sum()
case_rates_no_vaccination = final_cases_cbg_no_vaccination/cbg_sizes
case_rates_std_no_vaccination = case_rates_no_vaccination.std()
#final_deaths_cbg_no_vaccination = deaths_cbg_no_vaccination[-1,:] #20220118
#death_rates_no_vaccination = final_deaths_cbg_no_vaccination/cbg_sizes #20220118
#death_rates_std_no_vaccination = death_rates_no_vaccination.std() #20220118
'''
############################################################################################
# Load trained PolicyEvaluator #20220127

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
                            mob_data_root=args.mob_data_root,
                            output_root=output_root,
                            pretrain_embed_path=pretrain_embed_path,
                            normalize=args.normalize,
                            rel_result=args.rel_result,
                            with_vac_flag=False, #20220126
                            ) 

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
hybrid_idx = hybrid_idx.reshape(-1,1)
vac_flag_pos = vac_flag_pos.reshape(-1,1)

#gen_node_feats = np.concatenate((node_feats[:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level), axis=1)
#gen_node_feats = np.concatenate((node_feats, deg_centrality, clo_centrality, bet_centrality, mob_level), axis=1)
#gen_node_feats = np.concatenate((node_feats[:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, hybrid_idx, vac_flag_pos), axis=1) #20220129
gen_node_feats = np.concatenate((node_feats[:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, hybrid_idx), axis=1) #20220129

print('node_feats.shape: ', node_feats.shape) 
gen_node_feats = np.tile(gen_node_feats,(1,2))
print('node_feats.shape: ', gen_node_feats.shape) 

gen_node_feats = torch.Tensor(gen_node_feats)
adj = torch.Tensor(adj)
deg_centrality = torch.Tensor(deg_centrality)
clo_centrality = torch.Tensor(clo_centrality)
bet_centrality = torch.Tensor(bet_centrality)
mob_level = torch.Tensor(mob_level)
hybrid_idx = torch.Tensor(hybrid_idx) #20220129

# Model and optimizer
config = Config()
config.dim_touched = 8 # Num of feats used to calculate embedding #20220123
config.gcn_nfeat = config.dim_touched # Num of feats used to calculate embedding #20220123
config.gcn_nhid = args.hidden 
config.gcn_nclass = 32 
config.gcn_dropout = args.dropout
#config.linear_nin = config.gcn_nclass + (gen_node_feats.shape[1]-config.dim_touched)
config.linear_nin = config.gcn_nclass + (gen_node_feats.shape[1]-config.dim_touched)-1 #20220129 #最后一维是hybrid_group

config.linear_nhid1 = 100 
config.linear_nhid2 = 100
config.linear_nout = 1 #20220126 

config.NN = args.NN #20220201

#model = get_model(config, 'Generator')
model = get_model(config, 'Hierarchical_Generator')
print(model)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5, patience=6, min_lr=1e-8, verbose=True) #20220122

random.seed(42)

if args.cuda:
    model.cuda()
    adj = adj.cuda()
    gen_node_feats = gen_node_feats.cuda() #20220114
    node_feats = node_feats.cuda() #20220128
    deg_centrality = deg_centrality.cuda()
    clo_centrality = clo_centrality.cuda() 
    bet_centrality = bet_centrality.cuda() 
    mob_level = mob_level.cuda() 
    hybrid_idx = hybrid_idx.cuda() #20220129
    
# Training
traditional_evaluated_cases = []
outcome_list = []
train_loss_list = []
current_best = 0 
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()

    # Construct batch training sample
    '''
    group_idx_list = sorted(set(gen_node_feats[:,-2].cpu().numpy())) #最后一维用来当vac_flag，倒数第二维是所属的group
    num_groups = len(group_idx_list)
    num_samples_per_group = 5 #test
    sampled_x = gen_node_feats.unsqueeze(0).repeat(num_groups*num_samples_per_group,1,1)
    # Random sampling for each group
    #group_feats =  torch.zeros(num_groups*num_samples_per_group, args.NN, x.shape[1]) #(22,5,70,18)
    group_count = 0
    pdb.set_trace()
    for group_idx in group_idx_list:
        #sampled_feats = torch.zeros(num_samples_per_group, args.NN, x.shape[1]); #print('sampled_feats.shape: ', sampled_feats.shape)
        nodes = list(torch.where(gen_node_feats[:,-2]==group_idx)[0].cpu().numpy())
        for sample_idx in range(num_samples_per_group):
            sampled_nodes = random.sample(nodes, args.NN) # 从每个组采样NN个节点  
            #sampled_feats[sample_idx] = x[sampled_nodes]
            sampled_vac_flag = torch.zeros(gen_node_feats.shape[0])
            sampled_vac_flag[sampled_nodes] = 1
            sampled_x[sample_idx, :, -1] = sampled_vac_flag
        group_count += 1
    '''


    # Obtain indices of top-NN CBGs to be vaccinated
    vac_flag = model(gen_node_feats, adj) #policy = model(gen_node_feats, adj)
    print(torch.where(vac_flag.squeeze()!=0))
    
    #pdb.set_trace()
    # Prepare inputs for PolicyEvaluator
    #eval_node_feats = torch.cat((node_feats, vac_flag, deg_centrality, clo_centrality, bet_centrality, mob_level, vac_flag, vac_flag), axis=1) #20220128 
    eval_node_feats = torch.cat((node_feats[:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, node_feats[:,:4], deg_centrality, clo_centrality, bet_centrality, mob_level, vac_flag), axis=1) #20220201
    
    eval_node_feats = eval_node_feats.unsqueeze(axis=0)
    if args.cuda: eval_node_feats = eval_node_feats.cuda()

    # Run evaluation
    # Evaluate with traditional simulator (slow)
    '''
    total_cases, case_std = traditional_evaluate(vac_flag)
    print(f'total_cases: {total_cases}, case_std: {case_std}.')
    traditional_evaluated_cases.append(total_cases)
    # train_loss = -total_cases
    '''
    # Evaluate with trained GNN predictor (fast but less accurate)
    #outcome = evaluator(eval_node_feats, adj); print('outcome: ', outcome)
    #train_loss = -(evaluator(eval_node_feats, adj)) * (1+(-(evaluator(eval_node_feats, adj)))/current_best)#-outcome * (1+(-(outcome.detach()))/current_best)
    train_loss = evaluator(eval_node_feats, adj)
    outcome = train_loss.detach()
    outcome_list.append(outcome)
    #train_loss = -outcome
    #train_loss = -outcome * (1+(-outcome)/current_best) #直接这么写会报错:RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
    #train_loss = -outcome * (1+(-(outcome.detach()))/current_best)
    print('train_loss: ', train_loss)
    train_loss_list.append(train_loss.item())

    # Backprop
    with torch.autograd.set_detect_anomaly(True):
        train_loss.backward(retain_graph=True) #loss.backward()
    
    #print('traditional_evaluated_cases: ', traditional_evaluated_cases, ', current_best: ', current_best)
    #outcome_list.append(outcome)
    #current_best = np.array(outcome_list).min()
    #current_best = min(current_best, outcome)

    # Optimize
    optimizer.step()
    #scheduler.step(train_loss)

print('traditional_evaluated_cases: ', traditional_evaluated_cases)
print('trained_loss_list: ', train_loss_list)

pdb.set_trace()
