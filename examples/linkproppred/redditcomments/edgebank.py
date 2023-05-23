"""
Dynamic Link Prediction with EdgeBank

Reference: 
    - https://github.com/fpour/DGB/tree/main
"""

import timeit
import numpy as np
import torch
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm

# internal imports
from tgb.linkproppred.evaluate import Evaluator
from modules.edgebank_predictor import EdgeBankPredictor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.utils.utils import set_random_seed


# ==================
start_overall = timeit.default_timer()

# set the random seed for consistency
seed = 1
torch.manual_seed(seed)
set_random_seed(seed)

# set hyperparameters
memory_mode = 'fixed_time_window' #'unlimited' # `unlimited` or `fixed_time_window`
batch_size = 200
k_value = 10
time_window_ratio = 0.5 #0.15

# data loading
dataset_name = "redditcomments"
dataset = PyGLinkPropPredDataset(name=dataset_name, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
metric = dataset.eval_metric
print (metric)

# split the data
train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)

#data for memory in edgebank
hist_src = np.concatenate([train_data.src, val_data.src])
hist_dst = np.concatenate([train_data.dst, val_data.dst])
hist_ts = np.concatenate([train_data.t, val_data.t])

# Set EdgeBank with memory update
edgebank = EdgeBankPredictor(
        hist_src,
        hist_dst,
        hist_ts,
        memory_mode=memory_mode,
        time_window_ratio=time_window_ratio,
        pos_prob=1.0,)


def test_one_vs_many(loader, neg_sampler, split_mode):
    r"""
    evaluate the dynamic link prediction 
    """
    hist_at_k_list, mrr_list = [], []
    for pos_batch in tqdm(loader):
        pos_src, pos_dst, pos_t = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
        )
        neg_batch_list = neg_sampler.query_batch(pos_batch, split_mode=split_mode)
        
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = np.array([int(pos_src[idx]) for _ in range(len(neg_batch) + 1)])
            query_dst = np.concatenate([np.array([int(pos_dst[idx])]), neg_batch])

            y_pred = edgebank.predict_link(query_src, query_dst)
            # compute hist@k & MRR
            metrics_mrr_rnk = evaluator.eval_rnk_metrics(y_pred_pos=y_pred[0],  # first element is a positive edge
                                                         y_pred_neg=y_pred[1:],  # the rests are the negative edges
                                                         type_info='numpy', k=k_value)
            hist_at_k_list.append(metrics_mrr_rnk[f'hits@{k_value}'])
            mrr_list.append(metrics_mrr_rnk['mrr'])
            
            #* update edgebank memory after each batch
            edgebank.update_memory(pos_src, pos_dst, pos_t)

    perf_metrics = {f'hits@{k_value}': float(np.mean(hist_at_k_list)),
                    'mrr': float(np.mean(mrr_list)),
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
start_test = timeit.default_timer()
perf_metrics_test = test_one_vs_many(test_loader, neg_sampler, split_mode='test')
end_test = timeit.default_timer()

print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY --- NS-Mode: {NEG_SAMPLE_MODE} <<< ")
for perf_name, perf_value in perf_metrics_test.items():
    print(f"\tTest: {perf_name}: {perf_value: .4f}")
print(f"\tTest: Elapsed Time (s): {end_test - start_test: .4f}")


print(f'Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}')
print("==============================================================")
