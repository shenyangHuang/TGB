"""
Test the performance of EdgeRegress: Simple baselines for EdgeRegression task
Task: Link Regression

Date:
    - March 16, 2023
"""

import time
#import torch


#local dependencies
from tgb.utils import utils
from edge_regress_baselines import *
from tgb.edgeregression.evaluate import Evaluator
from tgb.edgeregression.dataset_pyg import PyGEdgeRegressDataset

SEED = 42
N_RUNS = 1


# data loading
name = "un_trade"
dataset = PyGEdgeRegressDataset(name=name, root="datasets")
data = dataset.data[0]

train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)

# ===================================== Set the seed
utils.set_random_seed(SEED)

# ===================================== The main flow .
for i_run in range(N_RUNS):

    start_time_run = time.time()

    # model initialization
    model_name = 'SnapshotMeanTeller'
    memory_update_enable = False

    if model_name == 'PersistentForcaster':
        baseline_model = PersistentForcaster(train_data, val_data, memory_update_enable)

    elif model_name == 'HistoricalMeanTeller':
        baseline_model = HistoricalMeanTeller(train_data, val_data, memory_update_enable)

    elif model_name == 'AbsoluteMeanTeller':
        baseline_model = AbsoluteMeanTeller(train_data, val_data, memory_update_enable)

    elif model_name == 'SnapshotMeanTeller':
        baseline_model = SnapshotMeanTeller(data)

    else:
        raise ValueError("Undefined model requested!")

    # ===================================== TEST
    # ========= transductive dynamic link regression

    if model_name == 'SnapshotMeanTeller':
        y_pred = baseline_model.compute_edge_weights(test_data.t)
    else:
        # memoy is ONLY updated if model.memory_update_enable = True
        y_pred = baseline_model.compute_edge_weights(test_data.src, test_data.dst, pos_e=True, pos_edge_weights=data.y)


    evaluator = Evaluator(name=name)
    print("DEBUG: Evaluator expected input format: ", evaluator.expected_input_format) 
    print("DEBUG: Evaluator expected output format: ", evaluator.expected_output_format) 
    input_dict = {"y_true": test_data["y"], "y_pred": y_pred, 'eval_metric': ['mse', 'rmse']}

    result_dict = evaluator.eval(input_dict) 
    print (f"INFO: {model_name}: Test MSE: {result_dict['mse']:.4f}, Test RMSE: {result_dict['rmse']:.4f}")



