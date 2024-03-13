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
import modules2.tkg_utils as utils

from modules2.recurrencybaseline_predictor import RecurrencyBaselinePredictor #apply_baselines_remote, score_psi
# from tgb.tkglinkpred.evaluate import Evaluator #TODO
from tgb.linkproppred.dataset import LinkPropPredDataset 
# from tgb.utils.utils import save_results #TODO


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
         window):
    scores_dict_for_test = {}
    final_logging_dict = {}
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
            ## use this if you wanna use ray:
            object_references = [
                apply_baselines_remote.remote(i, num_queries, test_data_c_rel, all_data_c_rel, window, 
                                    basis_dict, 
                                    num_nodes, 2*num_rels, 
                                    lmbda_psi, alpha) for i in range(num_processes_tmp)]
            output = ray.get(object_references)

            ## use this if you dont wanna use ray:
            # output = rb_predictor.apply_baselines(0, len(test_data_c_rel), test_data_c_rel, all_data_c_rel, window, 
            #                         basis_dict, 
            #                         num_nodes, 2*num_rels, 
            #                         lmbda_psi, alpha)
            

            ## updates the scores and logging dict for each process
            for proc_loop in range(num_processes_tmp):
                scores_dict_for_test.update(output[proc_loop][1])
                final_logging_dict.update(output[proc_loop][0])

        end = time.time()
        total_time = round(end - start, 6)  
        print("Relation {} finished in {} seconds.".format(rel, total_time))

    # perf_metrics = 0 #TODO
    return scores_dict_for_test


@ray.remote
def apply_baselines_remote(i, num_queries, test_data, all_data, window, basis_dict, num_nodes, 
                num_rels, lmbda_psi, alpha):
    return rb_predictor.apply_baselines(i, num_queries, test_data, all_data, window, basis_dict, num_nodes, 
                num_rels, lmbda_psi, alpha)

## train
def train(params_dict, basis_dict, rels, num_nodes, num_rels, valid_data_prel, trainvalid_data_prel, num_processes, 
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
        
        if rel in valid_data_prel.keys():      
            # valid data for this relation  
            valid_data_c_rel = copy(valid_data_prel[rel])
            timesteps_valid = list(set(valid_data_c_rel[:,3]))
            timesteps_valid.sort()
            trainvalid_data_c_rel = trainvalid_data_prel[rel]
            
            # queries per process if multiple processes
            num_queries = len(valid_data_c_rel) // num_processes
            if num_queries < num_processes: # if we do not have enough queries for all the processes
                num_processes_tmp = copy(1)
                num_queries = copy(len(valid_data_c_rel))
            else:
                num_processes_tmp = copy(num_processes)

            ######  1) tune lmbda_psi ###############        
            lmbdas_psi = params_dict['lmbda_psi']        

            alpha = 1
            best_lmbda_psi = 0
            best_mrr_psi = 0
            lmbda_mrrs = []

            best_config[str(rel_key)]['num_app_valid'] = copy(len(valid_data_c_rel))
            best_config[str(rel_key)]['num_app_train_valid'] = copy(len(trainvalid_data_c_rel))         
            best_config[str(rel_key)]['not_trained'] = 'False'       

            for lmbda_psi in lmbdas_psi:    
                object_references = [
                        apply_baselines_remote.remote(i, num_queries, valid_data_c_rel, trainvalid_data_c_rel, window, 
                                        basis_dict, num_nodes, 2*num_rels, 
                                        lmbda_psi, alpha) for i in range(num_processes_tmp)]
                output = ray.get(object_references)

                scores_dict_for_eval = {}
                for proc_loop in range(num_processes_tmp):
                    scores_dict_for_eval.update(output[proc_loop][1])

                # compute mrr
                mrr_and_friends = utils.compute_mrr(scores_dict_for_eval, valid_data_c_rel, timesteps_valid) #TODO
                mrr = mrr_and_friends[1]

                # # is new mrr better than previous best? if yes: store lmbda
                if mrr > best_mrr_psi:
                    best_mrr_psi = mrr
                    best_lmbda_psi = lmbda_psi

                lmbda_mrrs.append(mrr)
            best_config[str(rel_key)]['lmbda_psi'] = [best_lmbda_psi, best_mrr_psi]
            best_config[str(rel_key)]['other_lmbda_mrrs'] = lmbda_mrrs

            ##### 2) tune alphas: ###############
            best_config[str(rel_key)]['not_trained'] = 'False'    
            alphas = params_dict['alpha'] 
            lmbda_psi = best_config[str(rel_key)]['lmbda_psi'][0] # use the best lmbda psi

            alpha_mrrs = []
            best_mrr_alpha = 0
            for alpha in alphas:
                object_references = [
                        apply_baselines_remote.remote(i, num_queries, valid_data_c_rel, trainvalid_data_c_rel, window, 
                                        basis_dict, num_nodes, 2*num_rels,                                         
                                        lmbda_psi, alpha) for i in range(num_processes_tmp)]
                output_alpha = ray.get(object_references)

                scores_dict_for_eval_alpha = {}
                for proc_loop in range(num_processes_tmp):
                    scores_dict_for_eval_alpha.update(output_alpha[proc_loop][1])

                # compute mrr
                mrr_and_friends = utils.compute_mrr(scores_dict_for_eval_alpha, valid_data_c_rel, timesteps_valid)
                mrr_alpha = mrr_and_friends[1]

                # is new mrr better than previous best? if yes: store alpha
                if mrr_alpha > best_mrr_alpha:
                    best_mrr_alpha = mrr_alpha
                    best_alpha = alpha
                alpha_mrrs.append(mrr_alpha)

            best_config[str(rel_key)]['alpha'] = [best_alpha, best_mrr_alpha]
            best_config[str(rel_key)]['other_alpha_mrrs'] = alpha_mrrs

        end = time.time()
        total_time = round(end - start, 6)  
        print("Relation {} finished in {} seconds.".format(rel, total_time))
    return best_config



## todo: move these to dataset.py?
def group_by(data: np.array, key_idx: int, rels: list) -> dict:
    data_dict = {}
    data_sorted = sorted(data, key=itemgetter(key_idx))
    for key, group in groupby(data_sorted, key=itemgetter(key_idx)):
        data_dict[key] = np.array(list(group))
    return data_dict

def add_inverse_quadruples(triples: np.array, num_rels:int) -> np.array:
    inverse_triples = triples[:, [2, 1, 0, 3]]
    inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels  # we also need inverse triples
    all_triples = np.concatenate((triples[:,0:4], inverse_triples))

    return all_triples

def reformat_ts(timestamps):
    """ reformat timestamps s.t. they start with 0, and have stepsize 1.
    :param timestamps: np.array() with timestamps
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
    parser.add_argument("--dataset", "-d", default="tkgl-polecat", type=str) #ICEWS14, ICEWS18, GDELT, YAGO, WIKI
    parser.add_argument("--window", "-w", default=0, type=int) # set to e.g. 200 if only the most recent 200 timesteps should be considered. set to -2 if multistep
    parser.add_argument("--num_processes", "-p", default=10, type=int)
    parser.add_argument("--lmbda", "-l",  default=0.1, type=float) # fix lambda. used if trainflag == false
    parser.add_argument("--alpha", "-alpha",  default=0.999, type=float) # fix alpha. used if trainflag == false
    parser.add_argument("--train_flag", "-tr",  default=True) # do we need training, ie selection of lambda and alpha

    parsed = vars(parser.parse_args())
    return parsed

start_o = time.time()

parsed = get_args()
ray.init(num_cpus=parsed["num_processes"], num_gpus=0)

## load dataset and prepare it accordingly
name = parsed["dataset"]
dataset = LinkPropPredDataset(name=name, root="datasets", preprocess=True)

relations = dataset.edge_type.astype(int)
num_rels = len(set(relations))
rels = np.arange(0,2*num_rels)
subjects = dataset.full_data["sources"].astype(int)
objects= dataset.full_data["destinations"].astype(int)
num_nodes = max(np.concatenate((dataset.full_data['sources'].astype(int), dataset.full_data['destinations'].astype(int))))
timestamps = dataset.full_data["timestamps"].astype(int)

timestamps = reformat_ts(timestamps)

all_quads = np.stack((subjects, relations, objects, timestamps), axis=1)
train_data = all_quads[dataset.train_mask]
valid_data = all_quads[dataset.val_mask]
test_data = all_quads[dataset.test_mask]

train_data = add_inverse_quadruples(train_data, num_rels)
valid_data = add_inverse_quadruples(valid_data, num_rels)
test_data = add_inverse_quadruples(test_data, num_rels)
train_valid_data = np.concatenate([train_data, valid_data])
all_data = np.concatenate([train_data, valid_data, test_data])

test_data_prel = group_by(test_data, 1, rels)
all_data_prel = group_by(all_data, 1, rels)
valid_data_prel = group_by(valid_data, 1, rels)
trainvalid_data_prel = group_by(train_valid_data, 1, rels)

#
if parsed['train_flag']:
    params_dict = {}
    params_dict['lmbda_psi'] = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.5, 0.9, 1.0001] 
    params_dict['alpha'] = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1]
    default_lmbda_psi = params_dict['lmbda_psi'][-1]
    default_alpha = params_dict['alpha'][-2]

## load rules
basis_dict = create_basis_dict(train_valid_data)

## init

rb_predictor = RecurrencyBaselinePredictor(rels)
## train to find best lambda and alpha
start_train = time.time()
if parsed['train_flag']:
    best_config = train(params_dict, basis_dict, rels, num_nodes, num_rels, valid_data_prel, trainvalid_data_prel, parsed['num_processes'], 
         parsed['window'])
else: # use preset lmbda and alpha; same for all relations
    best_config = {} 
    for rel in rels:
        best_config[str(rel)] = {}
        best_config[str(rel)]['lmbda_psi'] = [parsed['lmbda']]
        best_config[str(rel)]['alpha'] = [parsed['alpha']]

end_train = time.time()
start_test = time.time()
eval_scores = test(best_config, basis_dict, rels, num_nodes, num_rels, test_data_prel, all_data_prel, parsed['num_processes'], 
         parsed['window'])

# compute mrr
#TODO
# print("Now computing the test MRR")
# timesteps_test = list(set(test_data[:,3]))
# timesteps_test.sort()
# mrr_and_friends = utils.compute_mrr(scores_dict_for_test, test_data, timesteps_test)
# mrr = mrr_and_friends[1]

# print some infos:
end_o = time.time()
train_time_o = round(end_train- start_train, 6)  
test_time_o = round(end_o- start_test, 6)  
total_time_o = round(end_o- start_o, 6)  
print("Running Training to find best configs finished in {} seconds.".format(train_time_o))
print("Running testing with best configs finished in {} seconds.".format(test_time_o))
print("Running all steps finished in {} seconds.".format(total_time_o))


# Lines to write to the file
lines = [
    name + "\n",
    datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n",
    "Running Training to find best configs finished in {} seconds.\n".format(train_time_o),
    "Running testing with best configs finished in {} seconds.\n".format(test_time_o),
    "Running all steps finished in {} seconds.\n".format(total_time_o)
]

# Write lines to the file
with open("runtimes.txt", "w") as file:
    file.writelines(lines)






ray.shutdown()


    