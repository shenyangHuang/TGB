"""
Dynamic Link Prediction with EdgeBank
NOTE: This implementation works only based on `numpy`

Reference: 
    - https://github.com/fpour/DGB/tree/main
"""

import time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm
import math

# internal imports
from tgb.linkproppred.evaluate import Evaluator
from modules.edgebank_predictor import EdgeBankPredictor
from tgb.utils.utils import set_random_seed
from tgb.linkproppred.dataset import LinkPropPredDataset


# ==================
start_overall = time.time()

# set the random seed for consistency
seed = 1
set_random_seed(seed)

# set hyperparameters
memory_mode = 'fixed_time_window' # `unlimited` or `fixed_time_window`
batch_size = 200
k_value = 10
time_window_ratio = 0.15 

# data loading with `numpy`
dataset_name = "wikipedia"
dataset = LinkPropPredDataset(name=dataset_name, root="datasets", preprocess=True)
data = dataset.full_data  
metric = dataset.eval_metric

# get masks
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask

#data for memory in edgebank
hist_src = np.concatenate([data['sources'][train_mask], data['sources'][val_mask]])
hist_dst = np.concatenate([data['destinations'][train_mask], data['destinations'][val_mask]])
hist_ts = np.concatenate([data['timestamps'][train_mask], data['timestamps'][val_mask]])

# Set EdgeBank with memory updater
edgebank = EdgeBankPredictor(
        hist_src,
        hist_dst,
        hist_ts,
        memory_mode=memory_mode,
        time_window_ratio=time_window_ratio)


def test_one_vs_many(data, test_mask, neg_sampler, split_mode):
    r"""
    evaluate the dynamic link prediction 
    """
    num_batches = math.ceil(len(data['sources'][test_mask]) / batch_size)
    mrr_list = []
    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(data['sources'][test_mask]))
        pos_src, pos_dst, pos_t = (
            data['sources'][test_mask][start_idx: end_idx],
            data['destinations'][test_mask][start_idx: end_idx],
            data['timestamps'][test_mask][start_idx: end_idx],
        )
        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)
        
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = np.array([int(pos_src[idx]) for _ in range(len(neg_batch) + 1)])
            query_dst = np.concatenate([np.array([int(pos_dst[idx])]), neg_batch])

            y_pred = edgebank.predict_link(query_src, query_dst)
            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0]]),
                "y_pred_neg": np.array(y_pred[1:]),
                "eval_metric": [metric],
            }
            metrics_mrr_rnk = evaluator.eval(input_dict)
            mrr_list.append(metrics_mrr_rnk[metric])
            
        # update edgebank memory after each positive batch
        edgebank.update_memory(pos_src, pos_dst, pos_t)

    perf_metrics = {metric: float(np.mean(mrr_list)),
                    }
    return perf_metrics


print("==========================================================")
print(f"============*** EdgeBank: {memory_mode} ***==============")
print("==========================================================")

evaluator = Evaluator(name=dataset_name)
neg_sampler = dataset.negative_sampler
NEG_SAMPLE_MODE = "hist_rnd"

# ==================================================== Test
# loading the test negative samples
dataset.load_test_ns()

# testing ...
start_test = time.time()
perf_metrics_test = test_one_vs_many(data, test_mask, neg_sampler, split_mode='test')
end_test = time.time()

print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY --- NS-Mode: {NEG_SAMPLE_MODE} <<< ")
for perf_name, perf_value in perf_metrics_test.items():
    print(f"\tTest: {perf_name}: {perf_value: .4f}")
print(f"\tTest: Elapsed Time (s): {end_test - start_test: .4f}")

print(f'Overall Elapsed Time (s): {time.time() - start_overall: .4f}')
print("==============================================================")
