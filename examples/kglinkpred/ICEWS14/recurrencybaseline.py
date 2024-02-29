
## imports
import json
import time
import argparse
import numpy as np
import pathlib
import os
from copy import copy
import json
import ray
import timeit

#internal imports 
import modules.tkg_utils as utils
from data import data_handler #todo exchange
from modules.recurrencybaseline_predictor import apply_baselines_remote, score_psi
# from tgb.tkglinkpred.evaluate import Evaluator #TODO
# from tgb.tkglinkpred.dataset import LinkPredDataset #TODO
# from tgb.utils.utils import save_results


## preprocess: define rules
def create_basis_dict(data):
    ""
    """
    data: concatenated train and vali data, INCLUDING INVERSE QUADRUPLES. we need it for the relation ids.
    """
    rels = list(set(train_valid_data[:,1]))
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
def test(best_config, basis_dict, rels, num_nodes, num_rels, test_data_prel, all_data_prel, num_processes, 
         score_func_psi, window):
    scores_dict_for_test = {}
    final_logging_dict = {}
    ## loop through relations and apply baselines
    for rel in rels:
        start = time.time()
        rel_key = int(copy(rel))
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
            object_references = [
                apply_baselines_remote.remote(i, num_queries, test_data_c_rel, all_data_c_rel, window, 
                                    basis_dict, score_func_psi, 
                                    num_nodes, 2*num_rels, 
                                    lmbda_psi, alpha) for i in range(num_processes_tmp)]
            output = ray.get(object_references)

            ## updates the scores and logging dict for each process
            for proc_loop in range(num_processes_tmp):
                scores_dict_for_test.update(output[proc_loop][1])
                final_logging_dict.update(output[proc_loop][0])

        end = time.time()
        total_time = round(end - start, 6)  
        print("Relation {} finished in {} seconds.".format(rel, total_time))

    perf_metrics = 0 #TODO
    return perf_metrics


## train
def train():
    """ optional, find best values for lambda and alpha
    """
    pass

## args
def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="ICEWS14", type=str) #ICEWS14, ICEWS18, GDELT, YAGO, WIKI
    parser.add_argument("--window", "-w", default=0, type=int) # set to e.g. 200 if only the most recent 200 timesteps should be considered. set to -2 if multistep
    parser.add_argument("--num_processes", "-p", default=10, type=int)
    parser.add_argument("--lmbda", "-l",  default=0.1, type=float) # fix lambda. used if trainflag == false
    parser.add_argument("--alpha", "-alpha",  default=0.999, type=float) # fix alpha. used if trainflag == false
    parser.add_argument("--trainflag", "-tr",  default=False) # do we need training, ie selection of lambda and alpha

    parsed = vars(parser.parse_args())
    return parsed

start_o = time.time()

parsed = get_args()
ray.init(num_cpus=parsed["num_processes"], num_gpus=0)
score_func_psi = score_psi #TODO


## load dataset - TODO: change
dataset = (parsed["dataset"], 3) # identifier, timestamp_column_idx
train_data, valid_data, test_data, stat = data_handler.load(dataset[0])
num_nodes, num_rels = int(stat[0]), int(stat[1])
train_data = data_handler.add_inverse_quadruples(train_data, num_rels)
valid_data = data_handler.add_inverse_quadruples(valid_data, num_rels)
test_data = data_handler.add_inverse_quadruples(test_data, num_rels)
train_valid_data = np.concatenate((train_data, valid_data))
all_data = np.concatenate((train_data, valid_data, test_data))
rels = np.arange(0,2*num_rels)
test_data_prel = data_handler.group_by(test_data, 1, rels)
all_data_prel = data_handler.group_by(all_data, 1, rels)

## load rules
basis_dict = create_basis_dict(train_valid_data)

## train to find best lambda and alpha
start_train = time.time()
if parsed['train_flag']:
    best_config = train()
else: # use preset lmbda and alpha; same for all relations
    best_config = {} 
    for rel in rels:
        best_config[str(rel)]['lmbda_psi'] = [parsed['lmbda']]
        best_config[str(rel)]['alpha'] = [parsed['alpha']]

end_train = time.time()
start_test = time.time()
eval_scores = test(best_config)

# print some infos:
end_o = time.time()
train_time_o = round(end_train- start_train, 6)  
test_time_o = round(end_o- start_test, 6)  
total_time_o = round(end_o- start_o, 6)  
print("Running Training to find best configs finished in {} seconds.".format(train_time_o))
print("Running testing with best configs finished in {} seconds.".format(test_time_o))
print("Running all steps finished in {} seconds.".format(total_time_o))

# compute mrr
#TODO
# print("Now computing the test MRR")
# timesteps_test = list(set(test_data[:,3]))
# timesteps_test.sort()
# mrr_and_friends = utils.compute_mrr(scores_dict_for_test, test_data, timesteps_test)
# mrr = mrr_and_friends[1]




ray.shutdown()


    