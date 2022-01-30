# python gt-gen-vac-fixed-num-cbgs.py MSA_NAME RANDOM_SEED NN NUM_EXPERIMENTS quick_test
# python gt-gen-vac-fixed-num-cbgs.py SanFrancisco 66 70 100 False

import setproctitle
setproctitle.setproctitle("gnn-simu-vac@chenlin")

import sys

import os
import datetime
import pandas as pd
import numpy as np
import pickle
import random
from scipy import stats

import constants
import functions
import disease_model #disease_model_only_modify_attack_rates

import time
import pdb

###############################################################################
# Constants

epic_data_root = '/data/chenlin/COVID-19/Data'
#gt_result_root = os.path.join(os.getcwd(),'../data/safegraph')
gt_result_root = os.path.abspath(os.path.join(os.pardir,'data/safegraph')); print('gt_result_root: ', gt_result_root)

MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
NUM_DAYS = 63
NUM_GROUPS = 3 #5

# Vaccination ratio
VACCINATION_RATIO = 0.02
print('VACCINATION_RATIO: ', VACCINATION_RATIO)

# Vaccination protection rate
PROTECTION_RATE = 1
# Policy execution ratio
EXECUTION_RATIO = 1
# Vaccination time
VACCINATION_TIME = 0 #31
print('VACCINATION_TIME: ', VACCINATION_TIME)

###############################################################################
# Main variable settings

MSA_NAME = sys.argv[1]; print('MSA_NAME: ',MSA_NAME) #MSA_NAME = 'SanFrancisco'
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME] #MSA_NAME_FULL = 'San_Francisco_Oakland_Hayward_CA'

# Random Seed
RANDOM_SEED = int(sys.argv[2])
print('RANDOM_SEED:',RANDOM_SEED)

# Divide the available vaccines to how many CBGs
NN = int(sys.argv[3])
print('Num of CBGs to receive vaccines: ', NN)

# Number of randombag experiments
NUM_EXPERIMENTS = int(sys.argv[4])
print('Number of randombag experiments: ', NUM_EXPERIMENTS)

# Quick Test: prototyping
quick_test = sys.argv[5]; print('Quick testing?', quick_test)
if(quick_test == 'True'): NUM_SEEDS = 2
else: NUM_SEEDS = 40 #30 #60 #30
print('NUM_SEEDS: ', NUM_SEEDS)
STARTING_SEED = range(NUM_SEEDS)

demo_feat_list = ['Age', 'Mean_Household_Income', 'Essential_Worker']

proportional = True # If true, divide vaccines proportional to cbg populations #20220117
if(proportional==True):
    extra_string = 'proportional'
else:
    extra_string = 'identical'

# Store filename
filename = os.path.join(gt_result_root, MSA_NAME, 
                        'vac_results_%s_%s_%s_randomseed%s_%sseeds_%ssamples_%s.csv'
                        %(MSA_NAME,VACCINATION_RATIO,NN,RANDOM_SEED,NUM_SEEDS, NUM_EXPERIMENTS, extra_string))
print('File name: ', filename)


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
                               p_sick_at_t0=constants.parameters_dict[MSA_NAME][0],
                               #vaccination_time=24*31, # when to apply vaccination (which hour)
                               vaccination_time=24*VACCINATION_TIME, # when to apply vaccination (which hour)
                               vaccination_vector = vaccination_vector,
                               vaccine_acceptance=vaccine_acceptance,
                               protection_rate = protection_rate,
                               home_beta=constants.parameters_dict[MSA_NAME][1],
                               cbg_attack_rates_original = cbg_attack_rates_scaled,
                               cbg_death_rates_original = cbg_death_rates_scaled,
                               poi_psi=constants.parameters_dict[MSA_NAME][2],
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
  
###############################################################################
# Load Data

# Load POI-CBG visiting matrices
f = open(os.path.join(epic_data_root, MSA_NAME, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()

# Load precomputed parameters to adjust(clip) POI dwell times
d = pd.read_csv(os.path.join(epic_data_root,MSA_NAME, 'parameters_%s.csv' % MSA_NAME)) 

# No clipping
new_d = d

all_hours = functions.list_hours_in_range(MIN_DATETIME, MAX_DATETIME)
poi_areas = new_d['feet'].values#面积
poi_dwell_times = new_d['median'].values#平均逗留时间
poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
del new_d
del d

# Load ACS Data for MSA-county matching
acs_data = pd.read_csv(os.path.join(epic_data_root,'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
good_list = list(msa_data['FIPS Code'].values);#print('CBG included: ', good_list)
del acs_data

# Load CBG ids for the MSA
cbg_ids_msa = pd.read_csv(os.path.join(epic_data_root,MSA_NAME,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
num_cbgs = len(cbg_ids_msa)
print('Number of CBGs in this metro area:', num_cbgs)

# Mapping from cbg_ids to columns in hourly visiting matrices
cbgs_to_idxs = dict(zip(cbg_ids_msa['census_block_group'].values, range(num_cbgs)))
x = {}
for i in cbgs_to_idxs:
    x[str(i)] = cbgs_to_idxs[i]


# Load SafeGraph data to obtain CBG sizes (i.e., populations)
filepath = os.path.join(epic_data_root,"safegraph_open_census_data/data/cbg_b01.csv")
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
'''
idxs_msa_nyt = y
print('Number of CBGs in this metro area:', len(idxs_msa_all))
print('Number of CBGs in to compare with NYT data:', len(idxs_msa_nyt))

nyt_included = np.zeros(len(idxs_msa_all))
for i in range(len(nyt_included)):
    if(i in idxs_msa_nyt):
        nyt_included[i] = 1
cbg_age_msa['NYT_Included'] = nyt_included.copy()
'''  
# Load other demographic data
# Income
filepath = os.path.join(epic_data_root,"ACS_5years_Income_Filtered_Summary.csv")
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
filepath = os.path.join(epic_data_root,"safegraph_open_census_data/data/cbg_c24.csv")
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

cbg_death_rates_original = np.loadtxt(os.path.join(epic_data_root, MSA_NAME, 'cbg_death_rates_original_'+MSA_NAME))
cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)
#print('Age-aware CBG-specific death rates loaded. Attack rates are irrelevant to age.')

# The scaling factors are set according to a grid search
# Fix attack_scale
attack_scale = 1
cbg_attack_rates_scaled = cbg_attack_rates_original * attack_scale
cbg_death_rates_scaled = cbg_death_rates_original * constants.death_scale_dict[MSA_NAME]
#print('Age-aware CBG-specific death rates scaled.')

###############################################################################
# Collect data together

data = pd.DataFrame()

data['Sum'] = cbg_age_msa['Sum'].copy()
data['Elder_Ratio'] = cbg_age_msa['Elder_Ratio'].copy()
data['Mean_Household_Income'] = cbg_income_msa['Mean_Household_Income'].copy()
data['Essential_Worker_Ratio'] = cbg_occupation_msa['Essential_Worker_Ratio'].copy()

###############################################################################
# Grouping: 按NUM_GROUPS分位数，将全体CBG分为NUM_GROUPS个组，将分割点存储在separators中

separators = functions.get_separators(data, NUM_GROUPS, 'Elder_Ratio','Sum', normalized=True)
data['Elder_Ratio_Group'] =  data['Elder_Ratio'].apply(lambda x : functions.assign_group(x, separators))

separators = functions.get_separators(data, NUM_GROUPS, 'Mean_Household_Income','Sum', normalized=False)
data['Mean_Household_Income_Group'] =  data['Mean_Household_Income'].apply(lambda x : functions.assign_group(x, separators))

separators = functions.get_separators(data, NUM_GROUPS, 'Essential_Worker_Ratio','Sum', normalized=True)
data['Essential_Worker_Ratio_Group'] =  data['Essential_Worker_Ratio'].apply(lambda x : functions.assign_group(x, separators))


# Hybrid Grouping
def assign_hybrid_group(data):
    return (data['Elder_Ratio_Group']*9 + data['Mean_Household_Income_Group']*3 + data['Essential_Worker_Ratio_Group'])
    #return (data['Age_Quantile_FOR_RANDOMBAG']*81 + data['Income_Quantile_FOR_RANDOMBAG']*27 + data['EW_Quantile_FOR_RANDOMBAG']*9 + data['Vulner_Quantile_FOR_RANDOMBAG']*3 + data['Damage_Quantile_FOR_RANDOMBAG'])
    
data['Hybrid_Group'] = data.apply(lambda x : assign_hybrid_group(x), axis=1)

###############################################################################
# 分层——采样：
# 首先检查每组人数，若小于target_num，则与邻组合并。#(若是第一组，则与后一组合并，否则与前一组合并。)
#(若是最后一组，则与前一组合并，否则与后一组合并。)
target_pop = data['Sum'].sum() * VACCINATION_RATIO 
target_cbg_num = NN+10 # at least contains (NN+10) CBGs
count = 0
max_group_idx = int(NUM_GROUPS*NUM_GROUPS*NUM_GROUPS)
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
df_filename = os.path.join(gt_result_root, MSA_NAME, 
                           'demographic_dataframe_%s_%sgroupsperfeat.csv' % (MSA_NAME,NUM_GROUPS))
print('Path to save constructed dataframe: ', df_filename)
pdb.set_trace()
data.to_csv(df_filename)


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
                                                            protection_rate = PROTECTION_RATE)
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
# Randomly choose NN CBGs for vaccine distribution
random.seed(RANDOM_SEED)

#for random_idx in range(NUM_EXPERIMENTS): #直接全范围随机采样
#    start1 = time.time()
#    print(f'Experiment {random_idx}: ')
#    # Construct the vaccination vector
#    target_idxs = random.sample(list(np.arange(num_cbgs)), NN) 

start = time.time()
NUM_GROUPWISE = int(NUM_EXPERIMENTS/count)
print('Number of samples per group: ', NUM_GROUPWISE)
for group_idx in range(max_group_idx):
    if(len(data[data['Hybrid_Group']==group_idx])==0): # 跳过0组
        continue
    
    for i in range(NUM_GROUPWISE):
        print('group_idx: ', group_idx, ', sample_idx: ', i)
        # Construct the vaccination vector
        target_idxs = random.sample(list(data[data['Hybrid_Group']==group_idx].index),NN) #只在组内采样

        vaccination_vector_randombag = functions.vaccine_distribution_fixed_nn(cbg_table=data, 
                                                                            vaccination_ratio=VACCINATION_RATIO, 
                                                                            nn=NN, 
                                                                            proportional=proportional, #20220117
                                                                            target_idxs=target_idxs
                                                                            )
        #vaccination_vector_randombag = functions.vaccine_distribution_fixed_nn(cbg_table=data, vaccination_ratio=VACCINATION_RATIO,nn=NN,proportional=proportional,target_idxs=target_idxs)

        # Retrieve vaccinated CBGs
        data['Vaccination_Vector'] = vaccination_vector_randombag
        data['Vaccinated'] = data['Vaccination_Vector'].apply(lambda x : 1 if x!=0 else 0)
        vaccinated_idxs = data[data['Vaccinated'] != 0].index.tolist()  
        print('Num of vaccinated CBGs: ', len(vaccinated_idxs))

        # Run simulations
        history_C2_randombag, history_D2_randombag = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                                    vaccination_vector=vaccination_vector_randombag,
                                                                    vaccine_acceptance=vaccine_acceptance,
                                                                    protection_rate = PROTECTION_RATE)
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
        result_df.to_csv(filename)
        
    
end = time.time()
print('Time: ',(end-start))


print('File name: ', filename)

