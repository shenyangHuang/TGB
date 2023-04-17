"""
Test the performance of EdgeRegress: Simple baselines for EdgeRegression task
Task: Link Regression

Date:
    - March 16, 2023


python EdgeRegress_test_LR.py -d "UNtrade" --prefix "EdgeRegress" --n_runs 5 --gpu 0 --seed 123 --tr_neg_sample "rnd" --ts_neg_sample "rnd"
"""

import time
#import torch


#local dependencies
from tgb.utils import utils
from edge_regress import *
from tgb.edgeregression.dataset import EdgeRegressionDataset
from tgb.edgeregression.evaluate import Evaluator

SEED = 42
name = "un_trade"
N_RUNS = 1


# ===================================== Set the seed, logger, and file paths
utils.set_random_seed(SEED)

# ===================================== load the data
# TODO work with tgb code
dataset = EdgeRegressionDataset(name=name, root="datasets")

train_data = dataset.train_data
val_data = dataset.val_data
test_data = dataset.test_data



# ===================================== The main flow .
for i_run in range(N_RUNS):

    start_time_run = time.time()

    # model initialization

    # # PersistentForcaster
    baseline_model = PersistentForcaster(train_data, val_data)

    # # HistoricalMeanTeller
    # baseline_model = HistoricalMeanTeller(train_data, val_data)

    # # AbsoluteMeanTeller
    # baseline_model = AbsoluteMeanTeller(train_data, val_data)

    # SnapshotMeanTeller
    # baseline_model = SnapshotMeanTeller(full_data)
    # ===================================== TEST

    # ========= Transductive

    # TODO change to the evaluator code
    y_pred = baseline_model.compute_edge_weights(test_data["sources"], test_data["destinations"])

    evaluator = Evaluator(name=name)
    print(evaluator.expected_input_format) 
    print(evaluator.expected_output_format) 
    input_dict = {"y_true": test_data["y"], "y_pred": y_pred, 'eval_metric': ['mse']}

    result_dict = evaluator.eval(input_dict) 
    print (result_dict)


    # test_perf_dict = eval_link_reg_only_pos_e_baseline(baseline_model, test_data, logger, snapshot_ts=True)  # only for SnapshotMeanTeller
    # dict_list = [meta_info, test_perf_dict]


