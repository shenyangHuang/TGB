"""
Dynamic Link Prediction with EdgeBank

Reference: 
    - https://github.com/fpour/DGB/tree/main
"""

import time
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm

# internal imports
from tgb.linkproppred.evaluate import Evaluator
from modules.edgebank_predictor import EdgeBankPredictor_MemoryUpdate
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.utils.utils import set_random_seed


# ==================
start_overall = time.time()

# set the random seed for consistency
seed = 1
torch.manual_seed(seed)
set_random_seed(seed)

# set parameters
batch_size = 200
k_value = 10
val_ratio = test_ratio = 0.15
edgebank_memory_mode = 'fixed_time_window'  # `unlimited` or `fixed_time_window`
time_window_ratio = val_ratio

# data loading
dataset_name = "wikipedia"
dataset = PyGLinkPropPredDataset(name=dataset_name, root="datasets")
data = dataset.get_TemporalData()

# split the data
train_data, val_data, test_data = data.train_val_test_split(val_ratio=val_ratio, test_ratio=test_ratio)

# train_loader = TemporalDataLoader(train_data, batch_size=batch_size)  # we do not need the train-loader for EdgeBank
# val_loader = TemporalDataLoader(val_data, batch_size=batch_size)  # we do not need the validation-loader for EdgeBank
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)


# Set EdgeBank with memory update
edgebank = EdgeBankPredictor_MemoryUpdate(memory_mode=edgebank_memory_mode, time_window_ratio=time_window_ratio)


def test_one_vs_many(loader, neg_sampler, split_mode):
    r"""
    evaluate the dynamic link prediction 
    """

    tr_val_hist_src = np.concatenate([train_data.src, val_data.src])
    tr_val_hist_dst = np.concatenate([train_data.dst, val_data.dst])
    tr_val_hist_ts = np.concatenate([train_data.t, val_data.t])

    hist_at_k_list, mrr_list = [], []
    for pos_batch in tqdm(loader):
        start_batch = time.time()
        pos_src, pos_dst, pos_t = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
        )
        neg_batch_list = neg_sampler.query_batch(pos_batch, split_mode=split_mode)

        # retrieve already seen test edges
        min_t_pos_batch = np.min(np.array(pos_t))
        test_mask = test_data.t < min_t_pos_batch
        seen_test_hist_src = test_data.src[test_mask]
        seen_test_hist_dst = test_data.dst[test_mask]
        seen_test_hist_ts = test_data.t[test_mask]

        # generate the set of already seen edges
        history_src = np.concatenate([tr_val_hist_src, seen_test_hist_src])
        history_dst = np.concatenate([tr_val_hist_dst, seen_test_hist_dst])
        history_ts = np.concatenate([tr_val_hist_ts, seen_test_hist_ts])
        
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = np.array([int(pos_src[idx]) for _ in range(len(neg_batch) + 1)])
            query_dst = np.concatenate([np.array([int(pos_dst[idx])]), neg_batch])

            y_pred = edgebank.predict_link_proba(history_src, history_dst, history_ts, query_src, query_dst)
            # compute hist@k & MRR
            metrics_mrr_rnk = evaluator.eval_rnk_metrics(y_pred_pos=y_pred[0],  # first element is a positive edge
                                                         y_pred_neg=y_pred[1:],  # the rests are the negative edges
                                                         type_info='numpy', k=k_value)
            hist_at_k_list.append(metrics_mrr_rnk[f'hits@{k_value}'])
            mrr_list.append(metrics_mrr_rnk['mrr'])

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
start_test = time.time()
perf_metrics_test = test_one_vs_many(test_loader, neg_sampler, split_mode='test')
end_test = time.time()

print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY --- NS-Mode: {NEG_SAMPLE_MODE} <<< ")
for perf_name, perf_value in perf_metrics_test.items():
    print(f"\tTest: {perf_name}: {perf_value: .4f}")
print(f"\tTest: Elapsed Time (s): {end_test - start_test: .4f}")


print(f'Overall Elapsed Time (s): {time.time() - start_overall: .4f}')
print("==============================================================")





