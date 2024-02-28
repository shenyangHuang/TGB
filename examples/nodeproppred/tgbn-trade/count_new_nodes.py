
import timeit
import numpy as np
from tqdm import tqdm
import math
import os
import os.path as osp
from pathlib import Path
import sys
import argparse

# internal imports
from modules.nodebank import NodeBank
from tgb.linkproppred.evaluate import Evaluator
from modules.edgebank_predictor import EdgeBankPredictor
from tgb.utils.utils import set_random_seed
from tgb.nodeproppred.dataset import NodePropPredDataset

# ==================
# ==================
# ==================

def count_nodes(data, test_mask, nodebank):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        data: a dataset object
        test_mask: required masks to load the test set edges
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluation
    """
    node_dict_new = {}
    node_dict = {}
    num_batches = math.ceil(len(data['sources'][test_mask]) / BATCH_SIZE)
    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(data['sources'][test_mask]))
        pos_src, pos_dst, pos_t = (
            data['sources'][test_mask][start_idx: end_idx],
            data['destinations'][test_mask][start_idx: end_idx],
            data['timestamps'][test_mask][start_idx: end_idx],
        )

        for node in pos_src:
            if (not nodebank.query_node(node)):
                if (node not in node_dict_new):
                    node_dict_new[node] = 1

            if (node not in node_dict):
                node_dict[node] = 1
        
        for node in pos_dst:
            if (not nodebank.query_node(node)):
                if (node not in node_dict_new):
                    node_dict_new[node] = 1

            if (node not in node_dict):
                node_dict[node] = 1

    return len(node_dict_new), len(node_dict)






def get_args():
    parser = argparse.ArgumentParser('*** TGB: EdgeBank ***')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='tgbl-wiki')
    parser.add_argument('--bs', type=int, help='Batch size', default=200)
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--mem_mode', type=str, help='Memory mode', default='unlimited', choices=['unlimited', 'fixed_time_window'])
    parser.add_argument('--time_window_ratio', type=float, help='Test window ratio', default=0.15)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv 

# ==================
# ==================
# ==================

start_overall = timeit.default_timer()

# set hyperparameters
args, _ = get_args()

SEED = args.seed  # set the random seed for consistency
set_random_seed(SEED)
MEMORY_MODE = args.mem_mode # `unlimited` or `fixed_time_window`
BATCH_SIZE = 10000
K_VALUE = args.k_value
TIME_WINDOW_RATIO = args.time_window_ratio
DATA = "tgbn-token" #"tgbl-wiki"

MODEL_NAME = 'EdgeBank'

# data loading with `numpy`
dataset = NodePropPredDataset(name=DATA, root="datasets", preprocess=True)
data = dataset.full_data  
metric = dataset.eval_metric

# get masks
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask

train_src = data['sources'][train_mask]
train_dst = data['destinations'][train_mask]

#data for memory in edgebank
hist_src = np.concatenate([data['sources'][train_mask]])
hist_dst = np.concatenate([data['destinations'][train_mask]])
hist_ts = np.concatenate([data['timestamps'][train_mask]])

# Set EdgeBank with memory updater
edgebank = EdgeBankPredictor(
        hist_src,
        hist_dst,
        hist_ts,
        memory_mode=MEMORY_MODE,
        time_window_ratio=TIME_WINDOW_RATIO)

print("==========================================================")
print(f"============*** {MODEL_NAME}: {MEMORY_MODE}: {DATA} ***==============")
print("==========================================================")

evaluator = Evaluator(name=DATA)


nodebank = NodeBank(train_src, train_dst)

new_val_num, val_total = count_nodes(data, val_mask, nodebank)
print ()
print ("-------------------------------------------------------")
print ("there are ", new_val_num, " new nodes in the validation set")
print ("there are ", val_total, " total nodes in the validation set")
print (" the percentage of new nodes in the validation set is ", (new_val_num/val_total))


new_test_num, test_total = count_nodes(data, test_mask, nodebank)
print ()
print ("-------------------------------------------------------")
print ("there are ", new_test_num, " new nodes in the test set")
print ("there are ", test_total, " total nodes in the test set") 
print (" the percentage of new nodes in the test set is ", (new_test_num/test_total))
