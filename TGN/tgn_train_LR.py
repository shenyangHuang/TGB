"""
Dynamic Link Regression (Edge Regression) with TGN 

Date:
    - Mar. 13, 2023: 
        Edge weights are processed by normalizing by the sum of the weighted temporal degree of the source nodes.
        Test phase only evaluates on the positive edges

"""
import math
import logging
import time
import sys
import torch
import numpy as np
import pandas as np
import pickle
from pathlib import Path

from evaluation.evaluation_LR import eval_link_reg_only_pos_e
from model.tgn import TGN
from utils.utils import *
from utils.data_load import get_data_LR
from utils.neg_edge_sampler import NegativeEdgeSampler
from utils.log import *
from train.train_LR import train_val_LR
from utils.test_data_generate import generate_test_edge_for_one_snapshot, generate_pos_graph_snapshots
from utils.arg_parser import get_args

# # Hyper-parameters based on the characteristics of the datasets
NUM_SNAPSHOTS_TEST = {
                    'UNtrade': 3,
                      }

# parse the arguments
args, sys_argv = get_args()

# set parameters
DATA = args.data
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
NUM_NEIGHBORS = args.n_degree
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
NUM_LAYER = args.n_layer
GPU = args.gpu
BATCH_SIZE = args.bs
NUM_EPOCH = args.n_epoch
LEARNING_RATE = args.lr
BACKPROP_EVERY = args.backprop_every
SEED = args.seed
EGO_SNAP = args.ego_snap
TR_NEG_SAMPLE = args.tr_neg_sample
TS_NEG_SAMPLE = args.ts_neg_sample
TR_RND_NE_RATIO = args.tr_rnd_ne_ratio
TS_RND_NE_RATIO = args.ts_rnd_ne_ratio

# for saving the results...
meta_info = {'model': args.prefix,
            'data': DATA,
            'tr_neg_sample': TR_NEG_SAMPLE,
            'ts_neg_sample': TS_NEG_SAMPLE,
            'tr_rnd_ne_ratio': TR_RND_NE_RATIO,
            'ts_rnd_ne_ratio': TS_RND_NE_RATIO,
            }

# ===================================== Set the seed, logger, and file paths
set_random_seed(SEED)
logger, get_checkpoint_path, get_best_model_path = set_up_log_path(args, sys_argv)

# ===================================== load the data

node_features, edge_features, full_data, train_data, val_data, \
 test_data = get_data_LR(DATA, seed=SEED, use_validation=True, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

# ===================================== Create neighbor samplers
# create two neighbor finders to handle graph extraction.
# the train and validation use partial ones, while test phase always uses the full one
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)
partial_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# ===================================== Negative Edge Samplers
# create random samplers to generate train/val/test instances for the negative edges
last_ts_before_test = val_data.timestamps[-1]
# train
train_rand_sampler = NegativeEdgeSampler(train_data.sources, train_data.destinations, train_data.timestamps,
                                        last_ts_before_test, NS=TR_NEG_SAMPLE, rnd_sample_ratio=TR_RND_NE_RATIO)
# validation
val_rand_sampler = NegativeEdgeSampler(full_data.sources, full_data.destinations, full_data.timestamps,
                                       last_ts_before_test, NS=TR_NEG_SAMPLE, rnd_sample_ratio=TR_RND_NE_RATIO, seed=0)
# nn_val_rand_sampler = NegativeEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, new_node_val_data.timestamps,
#                                           last_ts_before_test, NS=TR_NEG_SAMPLE, rnd_sample_ratio=TR_RND_NE_RATIO, seed=1)
# test
# test_rand_sampler = NegativeEdgeSampler(full_data.sources, full_data.destinations, full_data.timestamps,
                                        # last_ts_before_test, NS=TS_NEG_SAMPLE, seed=2, rnd_sample_ratio=TS_RND_NE_RATIO)
# nn_test_rand_sampler = NegativeEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, new_node_test_data.timestamps, 
#                                            last_ts_before_test, NS=TS_NEG_SAMPLE,
#                                            seed=3, rnd_sample_ratio=TS_RND_NE_RATIO)

logger.info(f"INFO: Number of unique timestamps in FULL_DATA: {get_no_unique_values(full_data.timestamps)}")
logger.info(f"INFO: Number of unique timestamps in TRAIN_DATA: {get_no_unique_values(train_data.timestamps)}")
logger.info(f"INFO: Number of unique timestamps in VAL_DATA: {get_no_unique_values(val_data.timestamps)}")
logger.info(f"INFO: Number of unique timestamps in TEST_DATA: {get_no_unique_values(test_data.timestamps)}")

# ===================================== set up the device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# ===================================== The main flow ...
# compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

Path("./LR_stats/").mkdir(parents=True, exist_ok=True)

for i_run in range(args.n_runs):

    start_time_run = time.time()
    logger.info("="*50)
    logger.info("********** Run {} starts. **********".format(i_run))
    
    ts_STD_pred_trans = "LR_stats/STD_pred_TRANS.csv"

    train_rand_sampler.reset_random_state(new_seed=i_run)

    # model initialization
    tgn = TGN(neighbor_finder=partial_ngh_finder, node_features=node_features,
                  edge_features=edge_features, device=device,
                  n_layers=NUM_LAYER,
                  n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                  message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                  memory_update_at_start=not args.memory_update_at_end,
                  embedding_module_type=args.embedding_module,
                  message_function=args.message_function,
                  aggregator_type=args.aggregator,
                  memory_updater_type=args.memory_updater,
                  n_neighbors=NUM_NEIGHBORS,
                  mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                  mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                  use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                  use_source_embedding_in_message=args.use_source_embedding_in_message,
                  dyrep=args.dyrep, decoder='LR')
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
    tgn = tgn.to(device)

    early_stopper = EarlyStopMonitor(max_round=args.patience)

    # ===================================== Training & Validation
    train_val_data = (train_data, val_data, None)
    train_val_sampler = (train_rand_sampler, val_rand_sampler, None)

    train_val_LR(tgn, train_val_data, train_val_sampler, NUM_NEIGHBORS, partial_ngh_finder, full_ngh_finder, 
              USE_MEMORY, BATCH_SIZE, BACKPROP_EVERY,
              NUM_EPOCH, criterion, optimizer, early_stopper, logger, i_run, device,
              get_checkpoint_path)

    # ===================================== TEST
    tgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
        val_memory_backup = tgn.memory.backup_memory()

    # ========= Transductive
    logger.info("TEST: Standard Setting: Transductive")
    if USE_MEMORY:
        tgn.memory.restore_memory(val_memory_backup)

    test_perf_dict = eval_link_reg_only_pos_e(model=tgn, data=test_data, logger=logger, batch_size=BATCH_SIZE, n_neighbors=NUM_NEIGHBORS)
    for metric_name, metric_value in test_perf_dict.items():
        logger.info('INFO: Test statistics: Old nodes -- {}: {}'.format(metric_name, metric_value))
    # write the summary of prediction results to a csv file (instead of scrapping the log file!)
    dict_list = [meta_info, test_perf_dict]
    write_dicts_to_csv(dict_list, ts_STD_pred_trans)
   
   # ===================================== Save the model
    if USE_MEMORY:
        tgn.memory.restore_memory(val_memory_backup)
    logger.info('Run {}: Saving TGN model ...'.format(i_run))
    torch.save(tgn.state_dict(), get_best_model_path(i_run))
    logger.info('TGN model saved at {}.'.format(get_best_model_path(i_run)))

    logger.info('Run {} elapsed time: {} seconds.'.format(i_run, (time.time() - start_time_run)))
    logger.info("="*50)

