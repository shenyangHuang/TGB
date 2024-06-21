"""
https://github.com/liu-yushan/TLogic/tree/main/mycode
TLogic: Temporal Logical Rules for Explainable Link Forecasting on Temporal Knowledge Graphs.
Yushan Liu, Yunpu Ma, Marcel Hildebrandt, Mitchell Joblin, Volker Tresp
"""

# imports
import sys
import os
import os.path as osp
from pathlib import Path
modules_path = osp.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(modules_path)
import timeit
import argparse
import numpy as np
import json
from joblib import Parallel, delayed
import itertools

from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset import LinkPropPredDataset 
from modules.tlogic_learn_modules import Temporal_Walk, Rule_Learner, store_edges
import modules.tlogic_apply_modules as ra
from tgb.utils.utils import set_random_seed,  save_results
from modules.tkg_utils import reformat_ts, get_inv_relation_id, create_scores_array

def learn_rules(i, num_relations):
    """
    Learn rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_relations (int): minimum number of relations for each process

    Returns:
        rl.rules_dict (dict): rules dictionary
    """

    # if seed:
    #     np.random.seed(seed)

    num_rest_relations = len(all_relations) - (i + 1) * num_relations
    if num_rest_relations >= num_relations:
        relations_idx = range(i * num_relations, (i + 1) * num_relations)
    else:
        relations_idx = range(i * num_relations, len(all_relations))

    num_rules = [0]
    for k in relations_idx:
        rel = all_relations[k]
        for length in rule_lengths:
            it_start =  timeit.default_timer()
            for _ in range(num_walks):
                walk_successful, walk = temporal_walk.sample_walk(length + 1, rel)
                if walk_successful:
                    rl.create_rule(walk)
            it_end =  timeit.default_timer()
            it_time = round(it_end - it_start, 6)
            num_rules.append(sum([len(v) for k, v in rl.rules_dict.items()]) // 2)
            num_new_rules = num_rules[-1] - num_rules[-2]
            print(
                "Process {0}: relation {1}/{2}, length {3}: {4} sec, {5} rules".format(
                    i,
                    k - relations_idx[0] + 1,
                    len(relations_idx),
                    length,
                    it_time,
                    num_new_rules,
                )
            )

    return rl.rules_dict

def apply_rules(i, num_queries, rules_dict, neg_sampler, data, window, learn_edges, all_quads, args, split_mode, 
                log_per_rel=False, num_rels=0):
    """
    Apply rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_queries (int): minimum number of queries for each process

    Returns:
        hits_list (list): hits list (hits@10 per sample)
        perf_list (list): performance list (mrr per sample)
    """
    perf_per_rel = {}
    for rel in range(num_rels):
            perf_per_rel[rel] = []
    print("Start process", i, "...")
    all_candidates = [dict() for _ in range(len(args))]
    no_cands_counter = 0

    num_rest_queries = len(data) - (i + 1) * num_queries
    if num_rest_queries >= num_queries:
        test_queries_idx = range(i * num_queries, (i + 1) * num_queries)
    else:
        test_queries_idx = range(i * num_queries, len(data))

    cur_ts = data[test_queries_idx[0]][3]
    edges = ra.get_window_edges(all_quads[:,0:4], cur_ts, learn_edges, window)

    it_start =  timeit.default_timer()
    hits_list = [0] * len(test_queries_idx)
    perf_list = [0] * len(test_queries_idx)
    for index, j in enumerate(test_queries_idx):
        neg_sample_el =  neg_sampler.query_batch(np.expand_dims(np.array(data[j,0]), axis=0), 
                                                np.expand_dims(np.array(data[j,2]), axis=0), 
                                                np.expand_dims(np.array(data[j,4]), axis=0), 
                                                np.expand_dims(np.array(data[j,1]), axis=0), 
                                                split_mode=split_mode)[0]        
        
        # neg_samples_batch[j]
        pos_sample_el =  data[j,2]
        test_query = data[j]
        assert pos_sample_el == test_query[2]
        cands_dict = [dict() for _ in range(len(args))]

        if test_query[3] != cur_ts:
            cur_ts = test_query[3]
            edges = ra.get_window_edges(all_quads[:,0:4], cur_ts, learn_edges, window)

        if test_query[1] in rules_dict:
            dicts_idx = list(range(len(args)))
            for rule in rules_dict[test_query[1]]:
                walk_edges = ra.match_body_relations(rule, edges, test_query[0])

                if 0 not in [len(x) for x in walk_edges]:
                    rule_walks = ra.get_walks(rule, walk_edges)
                    if rule["var_constraints"]:
                        rule_walks = ra.check_var_constraints(
                            rule["var_constraints"], rule_walks
                        )

                    if not rule_walks.empty:
                        cands_dict = ra.get_candidates(
                            rule,
                            rule_walks,
                            cur_ts,
                            cands_dict,
                            score_func,
                            args,
                            dicts_idx,
                        )
                        for s in dicts_idx:
                            cands_dict[s] = {
                                x: sorted(cands_dict[s][x], reverse=True)
                                for x in cands_dict[s].keys()
                            }
                            cands_dict[s] = dict(
                                sorted(
                                    cands_dict[s].items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )
                            )
                            top_k_scores = [v for _, v in cands_dict[s].items()][:top_k]
                            unique_scores = list(
                                scores for scores, _ in itertools.groupby(top_k_scores)
                            )
                            if len(unique_scores) >= top_k:
                                dicts_idx.remove(s)
                        if not dicts_idx:
                            break

            if cands_dict[0]:
                for s in range(len(args)):
                    # Calculate noisy-or scores
                    scores = list(
                        map(
                            lambda x: 1 - np.product(1 - np.array(x)),
                            cands_dict[s].values(),
                        )
                    )
                    cands_scores = dict(zip(cands_dict[s].keys(), scores))
                    noisy_or_cands = dict(
                        sorted(cands_scores.items(), key=lambda x: x[1], reverse=True)
                    )
                    all_candidates[s][j] = noisy_or_cands
            else:  # No candidates found by applying rules
                no_cands_counter += 1
                for s in range(len(args)):
                    all_candidates[s][j] = dict()

        else:  # No rules exist for this relation
            no_cands_counter += 1
            for s in range(len(args)):
                all_candidates[s][j] = dict()

        if not (j - test_queries_idx[0] + 1) % 100:
            it_end =  timeit.default_timer()
            it_time = round(it_end - it_start, 6)
            print(
                "Process {0}: test samples finished: {1}/{2}, {3} sec".format(
                    i, j - test_queries_idx[0] + 1, len(test_queries_idx), it_time
                )
            )
            it_start =  timeit.default_timer()

        predictions = create_scores_array(all_candidates[s][j], num_nodes)  
        predictions_of_interest_pos = np.array(predictions[pos_sample_el])
        predictions_of_interest_neg = predictions[neg_sample_el]
        input_dict = {
            "y_pred_pos": predictions_of_interest_pos,
            "y_pred_neg": predictions_of_interest_neg,
            "eval_metric": ['mrr'], 
        }

        predictions = evaluator.eval(input_dict)
        perf_list[index] = predictions['mrr']
        hits_list[index] = predictions['hits@10']
        if split_mode == "test":
            if log_per_rel:
                perf_per_rel[test_query[1]].append(perf_list[index]) #test_query[1] is the relation index

    if split_mode == "test":
        if log_per_rel:   
            for rel in range(num_rels):
                if len(perf_per_rel[rel]) > 0:
                    perf_per_rel[rel] = float(np.mean(perf_per_rel[rel]))
                else:
                    perf_per_rel.pop(rel)       
               

    return perf_list, hits_list, perf_per_rel


## args
def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="tkgl-smallpedia", type=str) 
    parser.add_argument("--rule_lengths", "-l", default="1", type=int, nargs="+")
    parser.add_argument("--num_walks", "-n", default="100", type=int)
    parser.add_argument("--transition_distr", default="exp", type=str)
    parser.add_argument("--window", "-w", default=0, type=int) # set to e.g. 200 if only the most recent 200 timesteps should be considered. set to -2 if multistep
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--num_processes", "-p", default=1, type=int)
    parser.add_argument("--alpha", "-alpha",  default=0.99, type=float) # fix alpha. used if trainflag == false
    # parser.add_argument("--train_flag", "-tr",  default=True) # do we need training, ie selection of lambda and alpha
    parser.add_argument("--save_config", "-c",  default=True) # do we need to save the selection of lambda and alpha in config file?
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--run_nr', type=int, help='Run Number', default=1)
    parser.add_argument('--learn_rules_flag', type=bool, help='Do we want to learn the rules', default=True)
    parser.add_argument('--rule_filename', type=str, help='if rules not learned: where are they stored', default='0_r[3]_n100_exp_s1_rules.json')
    parser.add_argument('--log_per_rel', type=bool, help='Do we want to log mrr per relation', default=False)
    parser.add_argument('--compute_valid_mrr', type=bool, help='Do we want to compute mrr for valid set', default=True)
    parsed = vars(parser.parse_args())
    return parsed

start_o =  timeit.default_timer()

## get args
parsed = get_args()
dataset = parsed["dataset"]
rule_lengths = parsed["rule_lengths"]
rule_lengths = [rule_lengths] if (type(rule_lengths) == int) else rule_lengths
print('rule_lengths', rule_lengths)
num_walks = parsed["num_walks"]
transition_distr = parsed["transition_distr"]
num_processes = parsed["num_processes"]
window = parsed["window"]
top_k = parsed["top_k"]
log_per_rel = parsed['log_per_rel']

MODEL_NAME = 'TLogic'
SEED = parsed['seed']  # set the random seed for consistency
set_random_seed(SEED)

## load dataset and prepare it accordingly
name = parsed["dataset"]
compute_valid_mrr = parsed["compute_valid_mrr"]
dataset = LinkPropPredDataset(name=name, root="datasets", preprocess=True)
DATA = name

relations = dataset.edge_type
num_rels = dataset.num_rels

subjects = dataset.full_data["sources"]
objects= dataset.full_data["destinations"]
num_nodes = dataset.num_nodes 
timestamps_orig = dataset.full_data["timestamps"]
timestamps = reformat_ts(timestamps_orig, DATA) # stepsize:1

all_quads = np.stack((subjects, relations, objects, timestamps, timestamps_orig), axis=1)
train_data = all_quads[dataset.train_mask,0:4] # we do not need the original timestamps
val_data = all_quads[dataset.val_mask,0:5]
test_data = all_quads[dataset.test_mask,0:5]
all_data = all_quads[:,0:4]

metric = dataset.eval_metric
evaluator = Evaluator(name=name)
neg_sampler = dataset.negative_sampler

inv_relation_id = get_inv_relation_id(num_rels)

#load the ns samples 

dataset.load_val_ns()
dataset.load_test_ns()
output_dir =  f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
learn_rules_flag = parsed['learn_rules_flag']
## 1. learn rules
start_train =  timeit.default_timer()
if learn_rules_flag:
    print("start learning rules")
    # edges (dict): edges for each relation
    # inv_relation_id (dict): mapping of relation to inverse relation
    
    temporal_walk = Temporal_Walk(train_data, inv_relation_id, transition_distr)
    rl = Rule_Learner(edges=temporal_walk.edges, id2relation=None, inv_relation_id=inv_relation_id,  
                        output_dir=output_dir)
    all_relations = sorted(temporal_walk.edges)  # Learn for all relations

    start =  timeit.default_timer()
    num_relations = len(all_relations) // num_processes
    output = Parallel(n_jobs=num_processes)(
        delayed(learn_rules)(i, num_relations) for i in range(num_processes)
    )
    end =  timeit.default_timer()

    all_rules = output[0]
    for i in range(1, num_processes):
        all_rules.update(output[i])

    total_time = round(end - start, 6)
    print("Learning finished in {} seconds.".format(total_time))

    rl.rules_dict = all_rules
    rl.sort_rules_dict()

    rule_filename = rl.save_rules(0, rule_lengths, num_walks, transition_distr, SEED)
    # rl.save_rules_verbalized(0, rule_lengths, num_walks, transition_distr, seed)
    # rules_statistics(rl.rules_dict)
else:
    rule_filename = parsed['rule_filename']
    print("Loading rules from file {}".format(parsed['rule_filename']))

end_train =  timeit.default_timer()

## 2. Apply rules

rules_dict = json.load(open(output_dir + rule_filename))
rules_dict = {int(k): v for k, v in rules_dict.items()}

rules_dict = ra.filter_rules(
    rules_dict, min_conf=0.01, min_body_supp=2, rule_lengths=rule_lengths
) # filter rules for minimum confidence, body support and rule length

learn_edges = store_edges(train_data)
score_func = ra.score_12
# It is possible to specify a list of list of arguments for tuning
args = [[0.1, 0.5]]

# compute valid mrr
start_valid =  timeit.default_timer()
if compute_valid_mrr:
    print('Computing valid MRR')

    num_queries = len(val_data) // num_processes

    output = Parallel(n_jobs=num_processes)(
        delayed(apply_rules)(i, num_queries,rules_dict, neg_sampler, val_data, window, learn_edges, 
                            all_quads, args, split_mode='val') for i in range(num_processes))
    end =  timeit.default_timer()

    perf_list_val = []
    hits_list_val = []

    for i in range(num_processes):
        perf_list_val.extend(output[i][0])
        hits_list_val.extend(output[i][1])
else:
    perf_list_val = [0]
    hits_list_val = [0]
    

end_valid =  timeit.default_timer()

# compute test mrr
if log_per_rel ==True:
    num_processes = 1 #otherwise logging per rel does not work for our implementation
start_test =  timeit.default_timer()
print('Computing test MRR')
start =  timeit.default_timer()
num_queries = len(test_data) // num_processes

output = Parallel(n_jobs=num_processes)(
    delayed(apply_rules)(i, num_queries,rules_dict, neg_sampler, test_data, window, learn_edges, 
                         all_quads, args, split_mode='test', log_per_rel=log_per_rel, num_rels=num_rels) for i in range(num_processes))
end =  timeit.default_timer()

perf_list_all = []
hits_list_all = []


for i in range(num_processes):
    perf_list_all.extend(output[i][0])
    hits_list_all.extend(output[i][1])
if log_per_rel == True:
    perf_per_rel = output[0][2]


total_time = round(end - start, 6)
total_valid_time = round(end_valid - start_valid, 6)
print("Application finished in {} seconds.".format(total_time))

print(f"The valid MRR is {np.mean(perf_list_val)}")
print(f"The MRR is {np.mean(perf_list_all)}")
print(f"The Hits@10 is {np.mean(hits_list_all)}")
print(f"We have {len(perf_list_all)} predictions")
print(f"The test set has len {len(test_data)} ")

end_o =  timeit.default_timer()
train_time_o = round(end_train- start_train, 6)  
test_time_o = round(end_o- start_test, 6)  
total_time_o = round(end_o- start_o, 6)  
print("Running Training to find best configs finished in {} seconds.".format(train_time_o))
print("Running testing with best configs finished in {} seconds.".format(test_time_o))
print("Running all steps finished in {} seconds.".format(total_time_o))

results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)

if log_per_rel == True:
    results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results_per_rel.json'
    with open(results_filename, 'w') as json_file:
        json.dump(perf_per_rel, json_file)

results_filename = f'{results_path}/{MODEL_NAME}_NONE_{DATA}_results.json'
metric = dataset.eval_metric
save_results({'model': MODEL_NAME,
              'train_flag': None,
              'rule_len': rule_lengths,
              'window': window,
              'data': DATA,
              'run': 1,
              'seed': SEED,
              metric: float(np.mean(perf_list_all)),
              'hits10': float(np.mean(hits_list_all)),
              'val_mrr': float(np.mean(perf_list_val)),
              'test_time': test_time_o,
              'tot_train_val_time': total_time_o,
              'valid_time': total_valid_time
              }, 
    results_filename)
