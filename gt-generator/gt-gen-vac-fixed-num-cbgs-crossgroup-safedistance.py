# python gt-gen-vac-fixed-num-cbgs-crossgroup-safedistance.py  
# python gt-gen-vac-fixed-num-cbgs-crossgroup-safedistance.py 

import setproctitle
setproctitle.setproctitle("gnn-simu-vac@chenlin")

import os
import datetime
import pandas as pd
import numpy as np
import pickle
import random
import argparse
from sklearn import preprocessing
from scipy import spatial

import constants
import functions
import disease_model #disease_model_only_modify_attack_rates

import time
import pdb

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--epic_data_root', default='/data/chenlin/COVID-19/Data',
                    help='TBA')
parser.add_argument('--gt_result_root', default=os.path.abspath(os.path.join(os.pardir,'data/safegraph')),
                    help='TBA')
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
parser.add_argument('--num_groups', type=int, default=3,
                    help='Num of groups in each demographic dimension.')
parser.add_argument('--msa_name', default='SanFrancisco',
                    help='MSA name.')   
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed.')         
parser.add_argument('--NN', type=int, default=70, 
                    help='Num of CBGs to receive vaccines.')
parser.add_argument('--num_experiments', type=int, default=100, 
                    help='Num of randombags (i.e., random strategies).')          
parser.add_argument('--quick_test', default=False, action='store_true',
                    help='Quick Test: prototyping.')   
parser.add_argument('--proportional', default=True, 
                    help='If true, divide vaccines proportional to cbg populations.')   
parser.add_argument('--grouping', default=False, action='store_true',
                    help='If true, only generate samples containing CBGs from the same demographic group.')   
parser.add_argument('--safe_distance', type=float,
                    help='Safe distance between samples.')   

args = parser.parse_args()

# Constants
#gt_result_root = os.path.abspath(os.path.join(os.pardir,'data/safegraph')); print('gt_result_root: ', gt_result_root)

###############################################################################
# Main variable settings

MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[args.msa_name] #MSA_NAME_FULL = 'San_Francisco_Oakland_Hayward_CA'

# Random Seed
print('Random_seed:',args.random_seed)
# Divide the available vaccines to how many CBGs
print('Num of CBGs to receive vaccines: ', args.NN)
# Number of randombag experiments
print('Number of randombag experiments: ', args.num_experiments)
# Quick Test: prototyping
print('Quick testing?', args.quick_test)
if(args.quick_test == True): NUM_SEEDS = 2
else: NUM_SEEDS = 40 #30 #60
print('NUM_SEEDS: ', NUM_SEEDS)
STARTING_SEED = range(NUM_SEEDS)

if(args.proportional==True):
    extra_string = 'proportional'
else:
    extra_string = 'identical'

# Store filename
if(args.grouping):
    filename = os.path.join(args.gt_result_root, args.msa_name, 
                        'vac_results_%s_%s_%s_randomseed%s_%sseeds_%ssamples_%s.csv'
                        %(args.msa_name,args.vaccination_ratio,args.NN,args.random_seed,NUM_SEEDS, args.num_experiments, extra_string))
else:
    filename = os.path.join(args.gt_result_root, args.msa_name, 
                            'safe_%s_crossgroup_vac_results_%s_%s_%s_randomseed%s_%sseeds_%ssamples_%s.csv'
                            %(args.safe_distance,args.msa_name,args.vaccination_ratio,args.NN,args.random_seed,NUM_SEEDS, args.num_experiments, extra_string))
print('filename: ', filename)
if(os.path.exists(filename)):
    print('This file already exists. Better have a check?')
    pdb.set_trace()
'''
# Compared filename (existing data) #20220130
compared_filename = os.path.join(args.gt_result_root, args.msa_name, 
                                 'vac_results_SanFrancisco_0.02_70_randomseed42_40seeds_1000samples_proportional.csv')
print('Compared_filename: ', filename)
# Compute mean, safegap of existing samples. 
vac_data_df = pd.read_csv(compared_filename)
'''

###############################################################################
# Functions

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
    del T1
    del L_1
    del I_1
    del C2
    del D2
    #return total_affected, history_C2, history_D2, total_affected_each_cbg
    return history_C2, history_D2
  
# Hybrid Grouping
def assign_hybrid_group(data):
    return (data['Elder_Ratio_Group']*9 + data['Mean_Household_Income_Group']*3 + data['Essential_Worker_Ratio_Group'])


def load_vac_results(vac_data_df, demo_data_df): #20220130 #Just for testing the safety check
    num_samples = len(vac_data_df)-1 #第0行是no_vaccination的结果
    
    # vaccination results
    vac_data_df = vac_data_df[1:]
    # 把str转为list，split flag是', '，然后再把其中每个元素由str转为int(用map函数)
    vac_data_df['Vaccinated_Idxs'] = vac_data_df['Vaccinated_Idxs'].apply(lambda x : list(map(int, (x.strip('[').strip(']').split(', ')))))
    vac_tags = np.array(vac_data_df['Vaccinated_Idxs'])

    num_feats = 3
    avg_feat_array = np.zeros((num_samples, num_feats))
    
    for i in range(num_samples):
        avg_feat_array[i,0] = np.array(demo_data_df['Elder_Ratio_Normed'])[vac_tags[i]].mean()
        avg_feat_array[i,1] = np.array(demo_data_df['Mean_Household_Income_Normed'])[vac_tags[i]].mean()
        avg_feat_array[i,2] = np.array(demo_data_df['Essential_Worker_Ratio_Normed'])[vac_tags[i]].mean()

    # Construct kd-tree
    '''
    kdtree_0 = spatial.KDTree(data=avg_feat_array[:,0], leafsize=10)
    find_point = avg_feat_array[i,0]  # 原点
    d, x = kdtree_0.query(find_point)
    '''
    pdb.set_trace()

    return avg_feat_array


def get_avg_feats(demo_data_df, vac_tags): #20220131
    num_feats = 3
    avg_feats = np.zeros(num_feats)
    
    avg_feats[0] = np.array(demo_data_df['Elder_Ratio_Normed'])[vac_tags].mean()
    avg_feats[1] = np.array(demo_data_df['Mean_Household_Income_Normed'])[vac_tags].mean()
    avg_feats[2] = np.array(demo_data_df['Essential_Worker_Ratio_Normed'])[vac_tags].mean()

    avg_feats = avg_feats.reshape(-1,1)
    return avg_feats

def get_std_feats(demo_data_df, vac_tags): #20220131
    num_feats = 3
    std_feats = np.zeros(num_feats)
    
    std_feats[0] = np.array(demo_data_df['Elder_Ratio_Normed'])[vac_tags].std()
    std_feats[1] = np.array(demo_data_df['Mean_Household_Income_Normed'])[vac_tags].std()
    std_feats[2] = np.array(demo_data_df['Essential_Worker_Ratio_Normed'])[vac_tags].std()

    std_feats = std_feats.reshape(-1,1)
    return std_feats

def check_safety(avg_feat_array, new_point, metric, safety_margin): #20220131
    num_feats = len(new_point)
    safety_flag = False
    if(metric=='single-dim'):
        smallest_distance = np.zeros(num_feats)
        
        for i in range(num_feats):
            smallest_distance[i] = np.min(abs(new_point[i]-avg_feat_array[i,:]))
        print('smallest_distance:',smallest_distance)
        if((smallest_distance>safety_margin).any()):
            safety_flag = True
        
    elif(metric=='l1'):
        smallest_distance = np.min(np.sum(np.abs(new_point-avg_feat_array),axis=1))
        print('smallest_distance:',smallest_distance)
        if((smallest_distance>safety_margin)):   
            safety_flag = True

    elif(metric=='l2'):
        smallest_distance = np.min(np.sqrt(np.sum(np.square(new_point-avg_feat_array),axis=1))) #https://aakashkh.github.io/python/machine%20learning/2019/07/20/Nearest-Neighbors-L2-L1-Distance.html
        print('smallest_distance:',smallest_distance,'safety_margin:', safety_margin)
        if((smallest_distance>safety_margin)):   
            safety_flag = True

    if(safety_flag):
        print('Passed safety check.')
        return True
    else:
        print('Too close to existing data. Drop it.')
        return False

###############################################################################
# Load Data

# Load POI-CBG visiting matrices
f = open(os.path.join(args.epic_data_root, args.msa_name, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()

# Load precomputed parameters to adjust(clip) POI dwell times
d = pd.read_csv(os.path.join(args.epic_data_root,args.msa_name, 'parameters_%s.csv' % args.msa_name)) 

# No clipping
new_d = d

all_hours = functions.list_hours_in_range(args.min_datetime, args.max_datetime)
poi_areas = new_d['feet'].values#面积
poi_dwell_times = new_d['median'].values#平均逗留时间
poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
del new_d
del d

# Load ACS Data for MSA-county matching
acs_data = pd.read_csv(os.path.join(args.epic_data_root,'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
good_list = list(msa_data['FIPS Code'].values);#print('CBG included: ', good_list)
del acs_data

# Load CBG ids for the MSA
cbg_ids_msa = pd.read_csv(os.path.join(args.epic_data_root,args.msa_name,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
num_cbgs = len(cbg_ids_msa)
print('Number of CBGs in this metro area:', num_cbgs)

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
# Add up males and females of the same age, according to the detailed age list (DETAILED_AGE_LIST)
# which is defined in constants.py
for i in range(3,25+1): # 'B01001e3'~'B01001e25'
    male_column = 'B01001e'+str(i)
    female_column = 'B01001e'+str(i+24)
    cbg_age_msa[constants.DETAILED_AGE_LIST[i-3]] = cbg_age_msa.apply(lambda x : x[male_column]+x[female_column],axis=1)
# Rename
cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
# Extract columns of interest
columns_of_interest = ['census_block_group','Sum'] + constants.DETAILED_AGE_LIST
cbg_age_msa = cbg_age_msa[columns_of_interest].copy()
# Deal with NaN values
cbg_age_msa.fillna(0,inplace=True)
# Deal with CBGs with 0 populations
cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
cbg_age_msa['Elder_Absolute'] = cbg_age_msa.apply(lambda x : x['70 To 74 Years']+x['75 To 79 Years']+x['80 To 84 Years']+x['85 Years And Over'],axis=1)
cbg_age_msa['Elder_Ratio'] = cbg_age_msa['Elder_Absolute'] / cbg_age_msa['Sum']

# Obtain cbg sizes (populations)
cbg_sizes = cbg_age_msa['Sum'].values
cbg_sizes = np.array(cbg_sizes,dtype='int32')
print('Total population: ',np.sum(cbg_sizes))

# Select counties belonging to the MSA
y = []
for i in x:
    if((len(i)==12) & (int(i[0:5])in good_list)):
        y.append(x[i])
    if((len(i)==11) & (int(i[0:4])in good_list)):
        y.append(x[i])
     
idxs_msa_all = list(x.values())
 
# Load other demographic data
# Income
filepath = os.path.join(args.epic_data_root,"ACS_5years_Income_Filtered_Summary.csv")
cbg_income = pd.read_csv(filepath)
# Drop duplicate column 'Unnamed:0'
cbg_income.drop(['Unnamed: 0'],axis=1, inplace=True)
# Income Data Resource 1: ACS 5-year (2013-2017) Data
# Extract pois corresponding to the metro area (Philadelphia), by merging dataframes
cbg_income_msa = pd.merge(cbg_ids_msa, cbg_income, on='census_block_group', how='left')
del cbg_income
# Deal with NaN values
cbg_income_msa.fillna(0,inplace=True)
# Add information of cbg populations, from cbg_age_Phi(cbg_b01.csv)
cbg_income_msa['Sum'] = cbg_age_msa['Sum'].copy()
# Rename
cbg_income_msa.rename(columns = {'total_household_income':'Total_Household_Income', 
                                 'total_households':'Total_Households',
                                 'mean_household_income':'Mean_Household_Income',
                                 'median_household_income':'Median_Household_Income'},inplace=True)

# Occupation                             
filepath = os.path.join(args.epic_data_root,"safegraph_open_census_data/data/cbg_c24.csv")
cbg_occupation = pd.read_csv(filepath)
# Extract pois corresponding to the metro area, by merging dataframes
cbg_occupation_msa = pd.merge(cbg_ids_msa, cbg_occupation, on='census_block_group', how='left')
del cbg_occupation
columns_of_essential_workers = list(constants.ew_rate_dict.keys())
for column in columns_of_essential_workers:
    cbg_occupation_msa[column] = cbg_occupation_msa[column].apply(lambda x : x*constants.ew_rate_dict[column])
cbg_occupation_msa['Essential_Worker_Absolute'] = cbg_occupation_msa.apply(lambda x : x[columns_of_essential_workers].sum(), axis=1)
cbg_occupation_msa['Sum'] = cbg_age_msa['Sum']
cbg_occupation_msa['Essential_Worker_Ratio'] = cbg_occupation_msa['Essential_Worker_Absolute'] / cbg_occupation_msa['Sum']
columns_of_interest = ['census_block_group','Sum','Essential_Worker_Absolute','Essential_Worker_Ratio']
cbg_occupation_msa = cbg_occupation_msa[columns_of_interest].copy()
# Deal with NaN values
cbg_occupation_msa.fillna(0,inplace=True)

##############################################################################
# Load and scale age-aware CBG-specific attack/death rates (original)

cbg_death_rates_original = np.loadtxt(os.path.join(args.epic_data_root,args.msa_name, 'cbg_death_rates_original_'+args.msa_name))
cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)

# The scaling factors are set according to a grid search
attack_scale = 1 # Fix attack_scale
cbg_attack_rates_scaled = cbg_attack_rates_original * attack_scale
cbg_death_rates_scaled = cbg_death_rates_original * constants.death_scale_dict[args.msa_name]

###############################################################################
# Collect data together

data = pd.DataFrame()

data['Sum'] = cbg_age_msa['Sum'].copy()
data['Elder_Ratio'] = cbg_age_msa['Elder_Ratio'].copy()
data['Mean_Household_Income'] = cbg_income_msa['Mean_Household_Income'].copy()
data['Essential_Worker_Ratio'] = cbg_occupation_msa['Essential_Worker_Ratio'].copy()
# Normalization
data['Elder_Ratio_Normed'] = preprocessing.robust_scale(data['Elder_Ratio'])
data['Mean_Household_Income_Normed'] = preprocessing.robust_scale(data['Mean_Household_Income'])
data['Essential_Worker_Ratio_Normed'] = preprocessing.robust_scale(data['Essential_Worker_Ratio'])

#avg_feat_array = load_vac_results(vac_data_df, data)
#pdb.set_trace()


###############################################################################
if(args.grouping): # Grouping: 按args.num_groups分位数，将全体CBG分为args.num_groups个组，将分割点存储在separators中
    print('Perform CBG grouping.')
    separators = functions.get_separators(data, args.num_groups, 'Elder_Ratio','Sum', normalized=True)
    data['Elder_Ratio_Group'] =  data['Elder_Ratio'].apply(lambda x : functions.assign_group(x, separators))
    separators = functions.get_separators(data, args.num_groups, 'Mean_Household_Income','Sum', normalized=False)
    data['Mean_Household_Income_Group'] =  data['Mean_Household_Income'].apply(lambda x : functions.assign_group(x, separators))
    separators = functions.get_separators(data, args.num_groups, 'Essential_Worker_Ratio','Sum', normalized=True)
    data['Essential_Worker_Ratio_Group'] =  data['Essential_Worker_Ratio'].apply(lambda x : functions.assign_group(x, separators))
    data['Hybrid_Group'] = data.apply(lambda x : assign_hybrid_group(x), axis=1)

    # 分层——采样：
    # 首先检查每组人数，若小于target_num，则与邻组合并。#(若是第一组，则与后一组合并，否则与前一组合并。)
    #(若是最后一组，则与前一组合并，否则与后一组合并。)
    target_pop = data['Sum'].sum() * args.vaccination_ratio
    target_cbg_num = args.NN+10 # at least contains (args.NN+10) CBGs
    count = 0
    max_group_idx = int(args.num_groups*args.num_groups*args.num_groups)
    for i in range(max_group_idx):
        print(len(data[data['Hybrid_Group']==i]))
        if(len(data[data['Hybrid_Group']==i])>0):
            count += 1
        if((data[data['Hybrid_Group']==i]['Sum'].sum()<target_pop) or (len(data[data['Hybrid_Group']==i])<target_cbg_num)):
            if(i==max_group_idx-1): 
                #data[data['Hybrid_Group']==i]['Hybrid_Group'] = 1
                data['Hybrid_Group'] = data['Hybrid_Group'].apply(lambda x : max_group_idx-2 if x==i else x)
            else:
                #data[data['Hybrid_Group']==i]['Hybrid_Group'] = i-1
                data['Hybrid_Group'] = data['Hybrid_Group'].apply(lambda x : i+1 if x==i else x)
    print('Num of groups: ', count)

    # Recheck after merging
    print('Recheck:')
    count = 0
    for i in range(max_group_idx):
        print(len(data[data['Hybrid_Group']==i]))
        if(len(data[data['Hybrid_Group']==i])>0):
            count += 1
    print('Num of groups: ', count)

    # Store filename
    df_filename = os.path.join(args.gt_result_root, args.msa_name, 
                            'demographic_dataframe_%s_%sgroupsperfeat.csv' % (args.msa_name,args.num_groups))
    print('Path to save constructed dataframe: ', df_filename)
    pdb.set_trace()
    #data.to_csv(df_filename)


###############################################################################
# Make dataframe
result_df = pd.DataFrame(columns=['Vaccinated_Idxs','Total_Cases','Case_Rates_STD','Total_Deaths','Death_Rates_STD'])
# Vaccine acceptance
vaccine_acceptance = np.ones(len(cbg_sizes)) # full acceptance

# Baseline:no_vaccination
vaccination_vector_no_vaccination = np.zeros(len(cbg_sizes))
# Run simulations
history_C2_no_vaccination, history_D2_no_vaccination = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                            vaccination_vector=vaccination_vector_no_vaccination,
                                                            vaccine_acceptance=vaccine_acceptance,
                                                            protection_rate = args.protection_rate)
# Average history records across random seeds
cases_cbg_no_vaccination, deaths_cbg_no_vaccination,_,_ = functions.average_across_random_seeds(history_C2_no_vaccination,history_D2_no_vaccination, 
                                                                                        num_cbgs, idxs_msa_all, 
                                                                                        print_results=False,draw_results=False)

final_cases_cbg_no_vaccination = cases_cbg_no_vaccination[-1,:]
case_rates_no_vaccination = final_cases_cbg_no_vaccination/cbg_sizes
case_rates_std_no_vaccination = case_rates_no_vaccination.std()
final_deaths_cbg_no_vaccination = deaths_cbg_no_vaccination[-1,:] #20220118
death_rates_no_vaccination = final_deaths_cbg_no_vaccination/cbg_sizes #20220118
death_rates_std_no_vaccination = death_rates_no_vaccination.std() #20220118
result_df = result_df.append({'Vaccinated_Idxs':[],
                              'Total_Cases':final_cases_cbg_no_vaccination.sum(),
                              'Case_Rates_STD':case_rates_std_no_vaccination,
                              'Total_Deaths':final_deaths_cbg_no_vaccination.sum(),
                              'Death_Rates_STD':death_rates_std_no_vaccination,
                            }, ignore_index=True)
print(result_df)  

###############################################################################
# Randomly choose args.NN CBGs for vaccine distribution
random.seed(args.random_seed)

if(args.grouping): # Grouping: 按args.num_groups分位数，将全体CBG分为args.num_groups个组，将分割点存储在separators中
    start = time.time()
    NUM_GROUPWISE = int(args.num_experiments/count)
    print('Number of samples per group: ', NUM_GROUPWISE)
    for group_idx in range(max_group_idx):
        if(len(data[data['Hybrid_Group']==group_idx])==0): # 跳过0组
            continue
        
        for i in range(NUM_GROUPWISE):
            print('group_idx: ', group_idx, ', sample_idx: ', i)
            # Construct the vaccination vector
            target_idxs = random.sample(list(data[data['Hybrid_Group']==group_idx].index),args.NN) #只在组内采样

            vaccination_vector_randombag = functions.vaccine_distribution_fixed_nn(cbg_table=data, 
                                                                                vaccination_ratio=args.vaccination_ratio, 
                                                                                nn=args.NN, 
                                                                                proportional=args.proportional, #20220117
                                                                                target_idxs=target_idxs
                                                                                )
            #vaccination_vector_randombag = functions.vaccine_distribution_fixed_nn(cbg_table=data, vaccination_ratio=args.vaccination_ratio,nn=args.NN,proportional=args.proportional,target_idxs=target_idxs)

            # Retrieve vaccinated CBGs
            data['Vaccination_Vector'] = vaccination_vector_randombag
            data['Vaccinated'] = data['Vaccination_Vector'].apply(lambda x : 1 if x!=0 else 0)
            vaccinated_idxs = data[data['Vaccinated'] != 0].index.tolist()  
            print('Num of vaccinated CBGs: ', len(vaccinated_idxs))

            # Run simulations
            history_C2_randombag, history_D2_randombag = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                                        vaccination_vector=vaccination_vector_randombag,
                                                                        vaccine_acceptance=vaccine_acceptance,
                                                                        protection_rate = args.protection_rate)
            '''
            # Check significance #20220118
            after_vac = np.sum(np.array(history_D2_randombag), axis=2)[-1,:]
            before_vac = np.sum(np.array(history_D2_no_vaccination), axis=2)[-1,:]
            print('after_vac: ', after_vac)
            print('before_vac: ', before_vac)
            print(stats.ttest_rel(after_vac, before_vac, alternative='less'))
            after_vac = np.sum(np.array(history_C2_randombag), axis=2)[-1,:]
            before_vac = np.sum(np.array(history_C2_no_vaccination), axis=2)[-1,:]
            print(stats.ttest_rel(after_vac, before_vac, alternative='less'))
            '''
            # Average history records across random seeds
            cases_cbg_randombag, deaths_cbg_randombag,_,_ = functions.average_across_random_seeds(history_C2_randombag,history_D2_randombag, 
                                                                                                num_cbgs, idxs_msa_all, 
                                                                                                print_results=False,draw_results=False)
            
            final_cases_cbg_randombag = cases_cbg_randombag[-1,:]
            case_rates = final_cases_cbg_randombag/cbg_sizes
            case_rates_std = case_rates.std()
            final_deaths_cbg_randombag = deaths_cbg_randombag[-1,:] #20220118
            death_rates = final_deaths_cbg_randombag/cbg_sizes #20220118
            death_rates_std = death_rates.std() #20220118
            print('total cases: ', final_cases_cbg_randombag.sum(), 
                #', case_rates.max(): ', case_rates.max(), 
                ', case_rates_std: ', case_rates_std,
                ', total deaths: ', final_deaths_cbg_randombag.sum(), 
                ', death_rates_std: ', death_rates_std,
                )

            # Store in df
            result_df = result_df.append({'Vaccinated_Idxs':vaccinated_idxs,
                                        'Total_Cases':final_cases_cbg_randombag.sum(),
                                        'Case_Rates_STD':case_rates_std,
                                        'Total_Deaths':final_deaths_cbg_randombag.sum(),
                                        'Death_Rates_STD':death_rates_std,
                                        },ignore_index=True)
            #print(result_df)  
            #pdb.set_trace()
            #result_df.to_csv(filename)
else:
    valid_count = 0
    random_idx = 0
    #for random_idx in range(args.num_experiments): #直接全范围随机采样
    while(valid_count<args.num_experiments):
        start1 = time.time()
        print(f'Valid samples till now: {valid_count}')
        print(f'Experiment {random_idx}: ')
        # Construct the vaccination vector
        target_idxs = random.sample(list(np.arange(num_cbgs)), args.NN)    
        vaccination_vector_randombag = functions.vaccine_distribution_fixed_nn(cbg_table=data, 
                                                                                vaccination_ratio=args.vaccination_ratio, 
                                                                                nn=args.NN, 
                                                                                proportional=args.proportional, #20220117
                                                                                target_idxs=target_idxs
                                                                                )
        
        # Retrieve vaccinated CBGs
        data['Vaccination_Vector'] = vaccination_vector_randombag
        data['Vaccinated'] = data['Vaccination_Vector'].apply(lambda x : 1 if x!=0 else 0)
        vaccinated_idxs = data[data['Vaccinated'] != 0].index.tolist()  
        new_point_avg = get_avg_feats(data, vaccinated_idxs);print('new_point_avg.squeeze(): ',new_point_avg.squeeze())
        new_point_std = get_std_feats(data, vaccinated_idxs);print('new_point_std.squeeze(): ',new_point_std.squeeze())
        
        if(random_idx==0):
            avg_feat_array = new_point_avg
            valid_count += 1
            random_idx += 1
            #pdb.set_trace()
        else:
            #if(check_safety(avg_feat_array,new_point_avg,'l2', args.safe_distance)): #2
            if(check_safety(avg_feat_array,new_point_avg,'single-dim', args.safe_distance)): 
                avg_feat_array = np.concatenate((avg_feat_array, new_point_avg),axis=1)
                valid_count += 1
                random_idx += 1
                #pdb.set_trace()
            else:
                random_idx += 1
                #pdb.set_trace()
                continue    

        #print('Num of vaccinated CBGs: ', len(vaccinated_idxs))

        # Run simulations
        history_C2_randombag, history_D2_randombag = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                                    vaccination_vector=vaccination_vector_randombag,
                                                                    vaccine_acceptance=vaccine_acceptance,
                                                                    protection_rate = args.protection_rate)

        # Average history records across random seeds
        cases_cbg_randombag, deaths_cbg_randombag,_,_ = functions.average_across_random_seeds(history_C2_randombag,history_D2_randombag, 
                                                                                            num_cbgs, idxs_msa_all, 
                                                                                            print_results=False,draw_results=False)
        
        final_cases_cbg_randombag = cases_cbg_randombag[-1,:]
        case_rates = final_cases_cbg_randombag/cbg_sizes
        case_rates_std = case_rates.std()
        final_deaths_cbg_randombag = deaths_cbg_randombag[-1,:] #20220118
        death_rates = final_deaths_cbg_randombag/cbg_sizes #20220118
        death_rates_std = death_rates.std() #20220118
        print('total cases: ', final_cases_cbg_randombag.sum(), 
            #', case_rates.max(): ', case_rates.max(), 
            ', case_rates_std: ', case_rates_std,
            ', total deaths: ', final_deaths_cbg_randombag.sum(), 
            ', death_rates_std: ', death_rates_std,
            )

        # Store in df
        result_df = result_df.append({'Vaccinated_Idxs':vaccinated_idxs,
                                    'Total_Cases':final_cases_cbg_randombag.sum(),
                                    'Case_Rates_STD':case_rates_std,
                                    'Total_Deaths':final_deaths_cbg_randombag.sum(),
                                    'Death_Rates_STD':death_rates_std,
                                    },ignore_index=True)
        #print(result_df)  
        #pdb.set_trace()
        result_df.to_csv(filename)    #先不存，观察一下  
        
    
end = time.time()
print('Time: ',(end-start))


print('File name: ', filename)

