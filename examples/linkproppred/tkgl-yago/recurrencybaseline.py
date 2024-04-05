
import sys
sys.path.insert(0, '/home/jgastinger/tgb/TGB2')

## imports

import time
import argparse
import numpy as np
from copy import copy


import ray
from datetime import datetime
from itertools import groupby
from operator import itemgetter

#internal imports 
from tgb_modules.recurrencybaseline_predictor import RecurrencyBaselinePredictor 
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset import LinkPropPredDataset 
from tgb.utils.utils import set_random_seed
from tgb.utils.utils import save_results 


## preprocess: define rules
def create_basis_dict(data):
    ""
    """
    data: concatenated train and vali data, INCLUDING INVERSE QUADRUPLES. we need it for the relation ids.
    """
    rels = list(set(train_val_data[:,1]))
    basis_dict = {}
    for rel in rels:
        basis_id_new = []
        rule_dict = {}
        rule_dict["head_rel"] = int(rel)
        rule_dict["body_rels"] = [int(rel)] #same body and head relation -> what happened before happens again
        rule_dict["conf"] = 1 #same confidence for every rule
        rule_new = rule_dict
        basis_id_new.append(rule_new)
        basis_dict[str(rel)] = basis_id_new
    return basis_dict


## test
def test(best_config, basis_dict, rels, num_nodes, num_rels, test_data_prel, all_data_prel, neg_sampler, num_processes, 
         window, evaluator):
    scores_dict_for_test = {}
    final_logging_dict = {}
    perf_list_all = []
    hits_list_all =[]
    ## loop through relations and apply baselines
    for rel in rels:
        start = time.time()
        if rel in test_data_prel.keys():
            lmbda_psi = best_config[str(rel)]['lmbda_psi'][0]
            alpha = best_config[str(rel)]['alpha'][0]

            # test data for this relation
            test_data_c_rel = test_data_prel[rel]
            timesteps_test = list(set(test_data_c_rel[:,3]))
            timesteps_test.sort()
            all_data_c_rel = all_data_prel[rel]
            
            # queries per process if multiple processes
            num_queries = len(test_data_c_rel) // num_processes
            if num_queries < num_processes: # if we do not have enough queries for all the processes
                num_processes_tmp = 1
                num_queries = len(test_data_c_rel)
            else:
                num_processes_tmp = num_processes      
            
            ## apply baselines for this relation
            neg_samples_batch = neg_sampler.query_batch(np.array(test_data_c_rel[:,0]), np.array(test_data_c_rel[:,2]), 
                                    np.array(test_data_c_rel[:,4]), edge_type=np.array(test_data_c_rel[:,1]), split_mode='test')
            pos_samples_batch = test_data_c_rel[:,2]
            
            ## use this if you wanna use ray:
            if num_processes > 1:
                object_references = [
                    apply_baselines_remote.remote(i, num_queries, test_data_c_rel, all_data_c_rel, window, 
                                        basis_dict, 
                                        num_nodes, num_rels, lmbda_psi, 
                                        alpha, evaluator, neg_samples_batch, pos_samples_batch, mode='test') for i in range(num_processes_tmp)]
                output = ray.get(object_references)

                # batch_data = test_data[test_queries_idx[0]]

                # updates the scores and logging dict for each process
                for proc_loop in range(num_processes_tmp):
                    perf_list_all.extend(output[proc_loop][0])
                    hits_list_all.extend(output[proc_loop][1])

            ## use this if you dont wanna use ray:
            else:
                output = rb_predictor.apply_baselines(0, len(test_data_c_rel), test_data_c_rel, all_data_c_rel, window, 
                                        basis_dict, 
                                        num_nodes, num_rels, 
                                        lmbda_psi, alpha, neg_samples_batch,pos_samples_batch, 
                                        evaluator)
                perf_list, hits_list = output
                perf_list_all.extend(perf_list)
                hits_list_all.extend(hits_list)
            


        end = time.time()
        total_time = round(end - start, 6)  
        print("Relation {} finished in {} seconds.".format(rel, total_time))

    # perf_metrics = 0 #TODO
    return scores_dict_for_test, perf_list_all, hits_list_all


@ray.remote
def apply_baselines_remote(i, num_queries, test_data, all_data, window, basis_dict, num_nodes, 
                num_rels, lmbda_psi, alpha, evaluator, neg_samples_batch, pos_samples_batch, mode):

    return rb_predictor.apply_baselines(i, num_queries, test_data, all_data, window, basis_dict, num_nodes, 
                num_rels, lmbda_psi, alpha, neg_samples_batch, pos_samples_batch, evaluator)

## train
def train(params_dict, basis_dict, rels, num_nodes, num_rels, val_data_prel, trainval_data_prel, neg_sampler, num_processes, 
         window):
    """ optional, find best values for lambda and alpha
    """
    best_config= {}
    for rel in rels: # loop through relations. for each relation, apply rules with selected params, compute valid mrr
        start = time.time()
        rel_key = int(rel)            

        best_config[str(rel_key)] = {}
        best_config[str(rel_key)]['not_trained'] = 'True'    
        best_config[str(rel_key)]['lmbda_psi'] = [default_lmbda_psi,0] #default
        best_config[str(rel_key)]['other_lmbda_mrrs'] = list(np.zeros(len(params_dict['lmbda_psi'])))
        best_config[str(rel_key)]['alpha'] = [default_alpha,0]  #default    
        best_config[str(rel_key)]['other_alpha_mrrs'] = list(np.zeros(len(params_dict['alpha'])))
        
        if rel in val_data_prel.keys():      
            # valid data for this relation  
            val_data_c_rel = copy(val_data_prel[rel])
            timesteps_valid = list(set(val_data_c_rel[:,3]))
            timesteps_valid.sort()
            trainval_data_c_rel = trainval_data_prel[rel]
            neg_samples_batch = neg_sampler.query_batch(np.array(val_data_c_rel[:,0]), np.array(val_data_c_rel[:,2]), 
                                    np.array(val_data_c_rel[:,4]), edge_type=np.array(val_data_c_rel[:,1]), split_mode='val')
            pos_samples_batch = val_data_c_rel[:,2]
            # queries per process if multiple processes
            num_queries = len(val_data_c_rel) // num_processes
            if num_queries < num_processes: # if we do not have enough queries for all the processes
                num_processes_tmp = copy(1)
                num_queries = copy(len(val_data_c_rel))
            else:
                num_processes_tmp = copy(num_processes)

            ######  1) tune lmbda_psi ###############        
            lmbdas_psi = params_dict['lmbda_psi']        

            alpha = 1
            best_lmbda_psi = 0.1
            best_mrr_psi = 0
            lmbda_mrrs = []

            best_config[str(rel_key)]['num_app_valid'] = copy(len(val_data_c_rel))
            best_config[str(rel_key)]['num_app_train_valid'] = copy(len(trainval_data_c_rel))         
            best_config[str(rel_key)]['not_trained'] = 'False'       

            
            for lmbda_psi in lmbdas_psi:   
                perf_list_all = []
                hits_list_all = []
                ## use this if you wanna use ray:

                object_references = [
                    apply_baselines_remote.remote(i, num_queries, val_data_c_rel, trainval_data_c_rel, window, 
                                        basis_dict, 
                                        num_nodes, num_rels, lmbda_psi, 
                                        alpha, evaluator, neg_samples_batch, 
                                        pos_samples_batch, mode='val') for i in range(num_processes_tmp)]
                output = ray.get(object_references)

                # batch_data = test_data[test_queries_idx[0]]

                # updates the scores and logging dict for each process
                for proc_loop in range(num_processes_tmp):
                    perf_list_all.extend(output[proc_loop][0])
                    hits_list_all.extend(output[proc_loop][1])

                # compute mrr
                mrr = np.mean(perf_list_all)
                # # is new mrr better than previous best? if yes: store lmbda
                if mrr > best_mrr_psi:
                    best_mrr_psi = float(mrr)
                    best_lmbda_psi = lmbda_psi

                lmbda_mrrs.append(float(mrr))
            best_config[str(rel_key)]['lmbda_psi'] = [best_lmbda_psi, best_mrr_psi]
            best_config[str(rel_key)]['other_lmbda_mrrs'] = lmbda_mrrs

            ##### 2) tune alphas: ###############
            best_config[str(rel_key)]['not_trained'] = 'False'    
            alphas = params_dict['alpha'] 
            lmbda_psi = best_config[str(rel_key)]['lmbda_psi'][0] # use the best lmbda psi

            alpha_mrrs = []
            # perf_list_all = []
            best_mrr_alpha = 0
            best_alpha=0.99
            for alpha in alphas:
                perf_list_all = []
                hits_list_all = []
                ## use this if you wanna use ray:
                object_references = [
                    apply_baselines_remote.remote(i, num_queries, val_data_c_rel, trainval_data_c_rel, window, 
                                        basis_dict, 
                                        num_nodes, num_rels, lmbda_psi, 
                                        alpha, evaluator, neg_samples_batch, 
                                        pos_samples_batch, mode='val') for i in range(num_processes_tmp)]
                output = ray.get(object_references)


                # updates the scores and logging dict for each process
                for proc_loop in range(num_processes_tmp):
                    # scores_dict_for_eval_lambda.update(output[proc_loop][1])
                    # final_logging_dict.update(output[proc_loop][0])
                    perf_list_all.extend(output[proc_loop][0])
                    hits_list_all.extend(output[proc_loop][1])

                # compute mrr
                mrr_alpha = np.mean(perf_list_all)

                # is new mrr better than previous best? if yes: store alpha
                if mrr_alpha > best_mrr_alpha:
                    best_mrr_alpha = float(mrr_alpha)
                    best_alpha = alpha
                alpha_mrrs.append(float(mrr_alpha))

            best_config[str(rel_key)]['alpha'] = [best_alpha, best_mrr_alpha]
            best_config[str(rel_key)]['other_alpha_mrrs'] = alpha_mrrs

        end = time.time()
        total_time = round(end - start, 6)  
        print("Relation {} finished in {} seconds.".format(rel, total_time))
    return best_config



## todo: move these to dataset.py?
def group_by(data: np.array, key_idx: int) -> dict:
    """
    group data in an np array to dict; where key is specified by key_idx. for example groups elements of array by relations
    :param data: [np.array] data to be grouped
    :param key_idx: [int] index for element of interest
    returns data_dict: dict with key: values of element at index key_idx, values: all elements in data that have that value
    """
    data_dict = {}
    data_sorted = sorted(data, key=itemgetter(key_idx))
    for key, group in groupby(data_sorted, key=itemgetter(key_idx)):
        data_dict[key] = np.array(list(group))
    return data_dict


def reformat_ts(timestamps):
    """ reformat timestamps s.t. they start with 0, and have stepsize 1.
    :param timestamps: np.array() with timestamps
    returns: np.array(ts_new)
    """
    all_ts = list(set(timestamps))
    all_ts.sort()
    ts_min = np.min(all_ts)
    ts_dist = all_ts[1] - all_ts[0]

    ts_new = []
    timestamps2 = timestamps - ts_min
    for timestamp in timestamps2:
        timestamp = int(timestamp/ts_dist)
        ts_new.append(timestamp)
    return np.array(ts_new)


## args
def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="tkgl-yago", type=str) 
    parser.add_argument("--window", "-w", default=0, type=int) # set to e.g. 200 if only the most recent 200 timesteps should be considered. set to -2 if multistep
    parser.add_argument("--num_processes", "-p", default=1, type=int)
    parser.add_argument("--lmbda", "-l",  default=0.1, type=float) # fix lambda. used if trainflag == false
    parser.add_argument("--alpha", "-alpha",  default=0.99, type=float) # fix alpha. used if trainflag == false
    parser.add_argument("--train_flag", "-tr",  default=False) # do we need training, ie selection of lambda and alpha
    parser.add_argument("--save_config", "-c",  default=True) # do we need to save the selection of lambda and alpha in config file?
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parsed = vars(parser.parse_args())
    return parsed

start_o = time.time()

parsed = get_args()
ray.init(num_cpus=parsed["num_processes"], num_gpus=0)

## load dataset and prepare it accordingly
name = parsed["dataset"]
dataset = LinkPropPredDataset(name=name, root="datasets", preprocess=True)
DATA = name
MODEL_NAME = 'RecurrencyBaseline'

SEED = parsed['seed']  # set the random seed for consistency
set_random_seed(SEED)

relations = dataset.edge_type
num_rels = dataset.num_rels
rels = np.arange(0,num_rels)
subjects = dataset.full_data["sources"]
objects= dataset.full_data["destinations"]
num_nodes = dataset.num_nodes 
timestamps_orig = dataset.full_data["timestamps"]

timestamps = reformat_ts(timestamps_orig)

all_quads = np.stack((subjects, relations, objects, timestamps, timestamps_orig), axis=1)
train_data = all_quads[dataset.train_mask]
val_data = all_quads[dataset.val_mask]
test_data = all_quads[dataset.test_mask]

metric = dataset.eval_metric
evaluator = Evaluator(name=name)
neg_sampler = dataset.negative_sampler



train_val_data = np.concatenate([train_data, val_data])
all_data = np.concatenate([train_data, val_data, test_data])

test_data_prel = group_by(test_data, 1)
all_data_prel = group_by(all_data, 1)
val_data_prel = group_by(val_data, 1)
trainval_data_prel = group_by(train_val_data, 1)


#load the ns samples 
if parsed['train_flag']:
    dataset.load_val_ns()
dataset.load_test_ns()

# parameter options
if parsed['train_flag']:
    params_dict = {}
    params_dict['lmbda_psi'] = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.5, 0.9, 1.0001] 
    params_dict['alpha'] = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1]
    default_lmbda_psi = params_dict['lmbda_psi'][-1]
    default_alpha = params_dict['alpha'][-2]

## load rules
basis_dict = create_basis_dict(train_val_data)

## init
rb_predictor = RecurrencyBaselinePredictor(rels)
## train to find best lambda and alpha
start_train = time.time()
if parsed['train_flag']:
    best_config = train(params_dict, basis_dict, rels, num_nodes, num_rels, val_data_prel, trainval_data_prel, neg_sampler, parsed['num_processes'], 
         parsed['window'])
    if parsed['save_config']:
        import json
        with open('best_config.json', 'w') as outfile:
            json.dump(best_config, outfile)
else: # use preset lmbda and alpha; same for all relations
    best_config = {} 
    for rel in rels:
        best_config[str(rel)] = {}
        best_config[str(rel)]['lmbda_psi'] = [parsed['lmbda']]
        best_config[str(rel)]['alpha'] = [parsed['alpha']]

end_train = time.time()
start_test = time.time()
eval_scores, perf_list_all, hits_list_all = test(best_config, basis_dict, rels, num_nodes, num_rels, test_data_prel, 
                                                 all_data_prel, neg_sampler, parsed['num_processes'], 
                                                parsed['window'], evaluator)


print(f"The MRR is {np.mean(perf_list_all)}")
print(f"The Hits@10 is {np.mean(hits_list_all)}")
print(f"We have {len(perf_list_all)} predictions")
print(f"The test set has len {len(test_data)} ")

end_o = time.time()
train_time_o = round(end_train- start_train, 6)  
test_time_o = round(end_o- start_test, 6)  
total_time_o = round(end_o- start_o, 6)  
print("Running Training to find best configs finished in {} seconds.".format(train_time_o))
print("Running testing with best configs finished in {} seconds.".format(test_time_o))
print("Running all steps finished in {} seconds.".format(total_time_o))

# for saving the results...
import os
import os.path as osp
from pathlib import Path
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_NONE_{DATA}_results.json'


metric = dataset.eval_metric
save_results({'model': MODEL_NAME,
              'train_flag': parsed['train_flag'],
              'data': DATA,
              'run': 1,
              'seed': SEED,
              metric: float(np.mean(perf_list_all)),
              'hits10': float(np.mean(hits_list_all)),
              'test_time': test_time_o,
              'tot_train_val_time': total_time_o
              }, 
    results_filename)


ray.shutdown()


    