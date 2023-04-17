"""
Test the performance of EdgeRegress: Simple baselines for EdgeRegression task
Task: Link Regression

Date:
    - March 16, 2023


python EdgeRegress_test_LR.py -d "UNtrade" --prefix "EdgeRegress" --n_runs 5 --gpu 0 --seed 123 --tr_neg_sample "rnd" --ts_neg_sample "rnd"
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


from utils.utils import *
from utils.data_load import get_data_LR
from utils.log import *
from utils.test_data_generate import generate_test_edge_for_one_snapshot, generate_pos_graph_snapshots
from utils.arg_parser import get_args
from heuristic_baselines.edge_regress import *

# # Hyper-parameters based on the characteristics of the datasets
NUM_SNAPSHOTS_TEST = {
                    'UNtrade': 3,
                      }

# parse the arguments
args, sys_argv = get_args()

# set parameters
DATA = args.data
GPU = args.gpu
BATCH_SIZE = args.bs
NUM_EPOCH = args.n_epoch
SEED = args.seed
EGO_SNAP = args.ego_snap
TR_NEG_SAMPLE = args.tr_neg_sample
TS_NEG_SAMPLE = args.ts_neg_sample
TR_RND_NE_RATIO = args.tr_rnd_ne_ratio
TS_RND_NE_RATIO = args.ts_rnd_ne_ratio
PREFIX = args.prefix

# ===================================== Set the seed, logger, and file paths
set_random_seed(SEED)
logger, get_checkpoint_path, get_best_model_path = set_up_log_path(args, sys_argv)

# ===================================== load the data

node_features, edge_features, full_data, train_data, val_data, \
 test_data = get_data_LR(DATA, seed=SEED, use_validation=True, val_ratio=args.val_ratio, test_ratio=args.test_ratio)


logger.info(f"INFO: Number of unique timestamps in FULL_DATA: {get_no_unique_values(full_data.timestamps)}")
logger.info(f"INFO: Number of unique timestamps in TRAIN_DATA: {get_no_unique_values(train_data.timestamps)}")
logger.info(f"INFO: Number of unique timestamps in VAL_DATA: {get_no_unique_values(val_data.timestamps)}")
logger.info(f"INFO: Number of unique timestamps in TEST_DATA: {get_no_unique_values(test_data.timestamps)}")

# ===================================== The main flow ...

Path("./LR_stats/").mkdir(parents=True, exist_ok=True)
ts_STD_pred_trans = "LR_stats/STD_pred_TRANS.csv"

for i_run in range(args.n_runs):

    start_time_run = time.time()
    logger.info("="*50)
    logger.info("********** Run {} starts. **********".format(i_run))

    # model initialization

    # # PersistentForcaster
    # baseline_model = PersistenPersistentForcasterceForcaster(train_data, val_data)

    # # HistoricalMeanTeller
    # baseline_model = HistoricalMeanTeller(train_data, val_data)

    # # AbsoluteMeanTeller
    # baseline_model = AbsoluteMeanTeller(train_data, val_data)

    # SnapshotMeanTeller
    baseline_model = SnapshotMeanTeller(full_data)

    # for saving the results...
    meta_info = {'model': baseline_model.model_name,
                'data': DATA,
                'tr_neg_sample': TR_NEG_SAMPLE,
                'ts_neg_sample': TS_NEG_SAMPLE,
                'tr_rnd_ne_ratio': TR_RND_NE_RATIO,
                'ts_rnd_ne_ratio': TS_RND_NE_RATIO,
            }

    # ===================================== TEST

    # ========= Transductive

    # test_perf_dict = eval_link_reg_only_pos_e_baseline(baseline_model, test_data, logger)
    test_perf_dict = eval_link_reg_only_pos_e_baseline(baseline_model, test_data, logger, snapshot_ts=True)  # only for SnapshotMeanTeller
    dict_list = [meta_info, test_perf_dict]
    write_dicts_to_csv(dict_list, ts_STD_pred_trans)


   # ===================================== 

    logger.info('TEST: Run {} elapsed time: {} seconds.'.format(i_run, (time.time() - start_time_run)))
    logger.info("="*50)


