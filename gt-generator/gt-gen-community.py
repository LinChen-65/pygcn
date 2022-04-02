# python gt-gen-community.py --num_strategies 100 --random_seed 42

import setproctitle
setproctitle.setproctitle("gnn-vac@chenlin")

import socket
import argparse
import os
import numpy as np
import pandas as pd
import datetime
import pickle
import time
import random

import constants
import functions
import disease_model_test as disease_model

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--msa_name', default='SanFrancisco',
                    help='MSA name.') 
parser.add_argument('--NN', type=int, default=5, 
                    help='Num of counties to receive vaccines.')                    
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed.')                          
parser.add_argument('--vaccination_ratio', type=float, default=0.02,
                    help='Vaccination ratio (w.r.t. total population).')     
parser.add_argument('--vaccination_time', type=int, default=0,
                    help='Vaccination time.') 
parser.add_argument('--protection_rate', type=float, default=1,
                    help='Protection rate.') 
parser.add_argument('--min_datetime', default=datetime.datetime(2020, 3, 1, 0),
                    help='Start date & time.')    
parser.add_argument('--max_datetime', default=datetime.datetime(2020, 5, 2, 23),
                    help='End date & time.')  
parser.add_argument('--num_days', type=int, default=63,
                    help='Num of simulation days.')  
parser.add_argument('--num_strategies', type=int, default=100, 
                    help='Num of randombags (i.e., random strategies).')          
parser.add_argument('--quick_test', default=False, action='store_true',
                    help='Quick Test: prototyping.')   
parser.add_argument('--proportional', default=True, 
                    help='If true, divide vaccines proportional to cbg populations.')  
args = parser.parse_args()

# Derived variables
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[args.msa_name] #MSA_NAME_FULL = 'San_Francisco_Oakland_Hayward_CA'

# Random Seed
print('Random_seed:',args.random_seed)
# Divide the available vaccines to how many communities
print('Num of communities to receive vaccines: ', args.NN)
# Number of random strategies
print('Number of random strategies: ', args.num_strategies)
# Quick Test: prototyping
print('Quick testing?', args.quick_test)
if(args.quick_test == True): NUM_SEEDS = 2
else: NUM_SEEDS = 30 #40 #60
print('NUM_SEEDS: ', NUM_SEEDS)
STARTING_SEED = range(NUM_SEEDS)  

# root
hostname = socket.gethostname()
print('hostname: ', hostname)
if(hostname in ['fib-dl3','rl3','rl2']):
    prime = '/data'
elif(hostname=='rl4'):
    prime = '/home'
elif(hostname=='fib-dl'): #dl2
    prime = '/data4'
root = os.path.join(prime, 'chenlin/COVID-19/Data')
saveroot = os.path.join(prime, 'chenlin/pygcn/data/safegraph')
community_path = os.path.join(prime, 'chenlin/pygcn/pygcn/cbg_to_cluster.npy')
all_tested_strategies_path = os.path.join(prime, 'chenlin/pygcn/data/safegraph', args.msa_name, f'all_tested_strategies_{args.msa_name}.npy')

# Store filename
filename = os.path.join(saveroot, args.msa_name, 
                        f'vac_results_community_{args.msa_name}_{args.vaccination_ratio}_{args.NN}_randomseed{args.random_seed}_{NUM_SEEDS}seeds.csv')
print('filename: ', filename)
if(os.path.exists(filename)):
    print('There exists result_df of the same name. Load it.')
    previous_result_exists = True
    result_df = pd.read_csv(filename)
    print('len(result_df): ', len(result_df))
    print('result_df.columns: ', result_df.columns)
    col_to_drop = []
    for i in range(len(result_df.columns)):
        if('Unnamed' in result_df.columns[i]):
            col_to_drop.append(result_df.columns[i])
    for i in range(len(col_to_drop)):
        result_df.drop(col_to_drop[i], axis=1, inplace=True)
    print('After removing unnamed columns, result_df.columns: ', result_df.columns)
else:
    previous_result_exists = False


# All simulated strategies
if(os.path.exists(all_tested_strategies_path)):
    print('There exists simulated strategies. Load them.')
    all_tested_strategies = np.load(all_tested_strategies_path).tolist()
else:
    all_tested_strategies = []
print('len(all_tested_strategies): ', len(all_tested_strategies))

###############################################################################
# Functions

def run_simulation(starting_seed, num_seeds, vaccination_vector, vaccine_acceptance,protection_rate=1):
    m = disease_model.Model(starting_seed=starting_seed, num_seeds=num_seeds,
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

    final_cases, final_deaths = m.simulate_disease_spread(no_print=True, store_history=False) #20220304
    return final_cases, final_deaths


def simulate(vaccination_vector): #20220331
    final_cases_cbg, final_deaths_cbg = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                       vaccination_vector=vaccination_vector,
                                                       vaccine_acceptance=vaccine_acceptance,
                                                       protection_rate = args.protection_rate)
    # Average across random seeds
    final_cases_cbg = np.mean(final_cases_cbg, axis=0)
    final_cases = final_cases_cbg.sum()
    case_rates = final_cases_cbg/cbg_sizes
    case_rates_std = case_rates.std()
    
    final_deaths_cbg = np.mean(final_deaths_cbg, axis=0)
    final_deaths = final_deaths_cbg.sum()
    death_rates = final_deaths_cbg/cbg_sizes
    death_rates_std = death_rates.std()

    return final_cases, case_rates_std, final_deaths, death_rates_std

###############################################################################
# Load Data

# Load POI-CBG visiting matrices
f = open(os.path.join(root, args.msa_name, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()

# Load precomputed parameters to adjust(clip) POI dwell times
d = pd.read_csv(os.path.join(root,args.msa_name, 'parameters_%s.csv' % args.msa_name)) 
all_hours = functions.list_hours_in_range(args.min_datetime, args.max_datetime)
poi_areas = d['feet'].values#面积
poi_dwell_times = d['median'].values#平均逗留时间
poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
del d

# Load ACS Data for MSA-county matching
acs_data = pd.read_csv(os.path.join(root,'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
good_list = list(msa_data['FIPS Code'].values);#print('CBG included: ', good_list)
del acs_data

# Load CBG ids for the MSA
cbg_ids_msa = pd.read_csv(os.path.join(root,args.msa_name,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
num_cbgs = len(cbg_ids_msa)
print('Number of CBGs in this metro area:', num_cbgs)

# Load SafeGraph data to obtain CBG sizes (i.e., populations)
filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_b01.csv")
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

# Mapping from cbg_ids to columns in hourly visiting matrices
cbgs_to_idxs = dict(zip(cbg_ids_msa['census_block_group'].values, range(num_cbgs)))
x = {}
for i in cbgs_to_idxs:
    x[str(i)] = cbgs_to_idxs[i]
idxs_msa_all = list(x.values())

# Load community clustering #20220331
with open(community_path, 'rb') as f:
    cbg_to_cluster = np.load(f)
cbg_age_msa['Community'] = cbg_to_cluster
num_communities = len(set(cbg_to_cluster))
# Check community population
community_population = []
for i in range(num_communities):
    this_population = 0
    for j in range(len(cbg_age_msa)):
        if(cbg_age_msa.loc[j, 'Community']==i):
            this_population += cbg_age_msa.loc[j, 'Sum']
    community_population.append(this_population)
print('community_population: ', community_population)


##############################################################################
# Load and scale age-aware CBG-specific attack/death rates (original)

cbg_death_rates_original = np.loadtxt(os.path.join(root,args.msa_name, 'cbg_death_rates_original_'+args.msa_name))
cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)

# The scaling factors are set according to a grid search
attack_scale = 1 # Fix attack_scale
cbg_attack_rates_scaled = cbg_attack_rates_original * attack_scale
cbg_death_rates_scaled = cbg_death_rates_original * constants.death_scale_dict[args.msa_name]

###############################################################################
# Start simulation

# Vaccine acceptance
vaccine_acceptance = np.ones(len(cbg_sizes)) # full acceptance

# Make dataframe to store results
if(not previous_result_exists):
    result_df = pd.DataFrame(columns=['Vaccinated_Communities',
                                      'Total_Cases','Case_Rates_STD','Total_Deaths','Death_Rates_STD'])

    # Baseline:no_vaccination
    vaccination_vector_no_vaccination = np.zeros(len(cbg_sizes))
    # Run simulation
    final_cases_no_vaccination, case_rates_std_no_vaccination, final_deaths_no_vaccination, death_rates_std_no_vaccination = simulate(vaccination_vector_no_vaccination)
    # Add to result_df
    result_df = result_df.append({'Vaccinated_Communities':[],
                                'Total_Cases':final_cases_no_vaccination,
                                'Case_Rates_STD':case_rates_std_no_vaccination,
                                'Total_Deaths':final_deaths_no_vaccination,
                                'Death_Rates_STD':death_rates_std_no_vaccination,
                                }, ignore_index=True)
    print(result_df) 


###############################################################################
# Randomly choose args.NN communities
# and distribute vaccines to each of these CBGs proportionally to CBG population

random.seed(args.random_seed)

start = time.time()
total_count = 0
valid_count = 0
while valid_count < args.num_strategies:
    print(f'total_count: {total_count}, valid_count: {valid_count}.')
    start1 = time.time()

    # Construct the vaccination vector
    target_communities = random.sample(list(np.arange(num_communities)), args.NN)
    # Sort, ascendant
    target_communities = list(np.sort(np.array(target_communities)))
    print('Target communities: ', target_communities)
    # Check whether this result already exists
    if(target_communities in all_tested_strategies):
        total_count += 1
        continue
    else:
        target_cbgs = []
        for i in range(len(cbg_age_msa)):
            if(cbg_age_msa.loc[i, 'Community'] in target_communities):
                target_cbgs.append(i)
        print('len(target_cbgs): ', len(target_cbgs))
        vaccination_vector = functions.vaccine_distribution_fixed_nn(cbg_table=cbg_age_msa, 
                                                                    vaccination_ratio=args.vaccination_ratio, 
                                                                    nn=len(target_cbgs),#args.NN, 
                                                                    proportional=args.proportional, #20220117
                                                                    target_idxs=target_cbgs
                                                                    )

        # Run simulations
        final_cases, case_rates_std, final_deaths, death_rates_std = simulate(vaccination_vector)
        print(final_cases, case_rates_std, final_deaths, death_rates_std)
        # Add to result_df
        result_df = result_df.append({'Vaccinated_Communities':target_communities,
                                    'Total_Cases':final_cases,
                                    'Case_Rates_STD':case_rates_std,
                                    'Total_Deaths':final_deaths,
                                    'Death_Rates_STD':death_rates_std,
                                    }, ignore_index=True)
        result_df.to_csv(filename)
        print('len(result_df): ', len(result_df))

        total_count += 1
        valid_count += 1

        all_tested_strategies.append(target_communities)
        all_tested_strategies_array = np.array(all_tested_strategies)
        print(f'This strategy uses time: {time.time()-start1}')
        np.save(all_tested_strategies_path,all_tested_strategies_array) 

print(f'Obtaining {args.num_strategies} valid strategies uses time: {time.time()-start}')


pdb.set_trace()
