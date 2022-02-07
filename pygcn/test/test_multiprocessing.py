import multiprocessing
import pickle
import argparse
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', default= '/home', 
                    help='Prefix of data root. /home for rl4, /data for dl3, /data4 for dl2.')
parser.add_argument('--simulation_cache_save_filename', default='chenlin/pygcn/pygcn/simulation_cache.pkl',
                    help='File to save traditional_simulate results.')
args = parser.parse_args()

simulation_cache_save_path = os.path.join(args.prefix,args.simulation_cache_save_filename)
print('simulation_cache_save_path: ', simulation_cache_save_path)


# Multiprocessing test for read/write cache_dict
#cache_dict = dict() #不能用普通dict，否则无法实现dict更新！
cache_dict = multiprocessing.Manager().dict()
if(os.path.exists(simulation_cache_save_path)):
    print('Load existing dict.')
    with open(simulation_cache_save_path, 'rb') as f:
        saved_dict = pickle.load(f) 
#cache_dict[1] = 1
cache_dict = saved_dict #可以用普通dict给共享dict赋值
print('cache_dict: ', cache_dict)
pdb.set_trace()

def cache_worker(num,cache_dict):
    print(f'worker {num}')
    if num in cache_dict: print('Found in cache_dict.')
    else:
        cache_dict[num] = num
    #    #cache_dict[num].append(num)
    #    # get the shared list
    #    help_list = cache_dict[num]
    #    help_list.append(num)
    #    # forces the shared list to 
    #    # be serialized back to manager
    #    cache_dict[num] = help_list
        print('cache_dict updated.')
    return

num_list = []
for i in range(10):
    num_list.append(i)

pool = multiprocessing.Pool(processes=6)
for i in range(10):
    pool.apply_async(cache_worker, (num_list[i],cache_dict))
pool.close()
pool.join()
print('Done multiprocessing.')
print(cache_dict)
pdb.set_trace()


# First multiprocessing test
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