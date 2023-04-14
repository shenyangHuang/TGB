"""
implement persistant forecast as baseline for the node prop pred task
simply predict last seen label for the node
"""

import time
import os.path as osp
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import ndcg_score
from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader


#local imports
from tgb.nodeproppred.dataset_pyg import PyGNodePropertyDataset

device = 'cpu'

#! first need to provide pyg dataset support for lastfm dataset

name = "lastfmgenre"
dataset = PyGNodePropertyDataset(name=name, root="datasets")
num_classes = dataset.num_classes
data = dataset.data[0]
data.t = data.t.long()
data = data.to(device)
print ("finished setting up dataset")

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)

batch_size = 200

train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)

class PersistantForecaster:
    def __init__(self, num_class):
        self.dict = {}
        self.num_class = num_class

    def update_dict(self, node_id, label):
        self.dict[node_id] = label

    def query_dict(self, node_id):
        r"""
        Parameters:
            node_id: the node to query
        Returns:
            returns the last seen label of the node if it exists, if not return zero vector
        """
        if node_id in self.dict:
            return self.dict[node_id]
        else:
            return np.zeros(self.num_class)

forecaster = PersistantForecaster(num_classes)


def test_n_upate(loader):
    total_ncdg = 0
    label_t = dataset.get_label_time() #check when does the first label start
    TOP_K = 10
    num_labels = 0

    print ("training starts")
    for batch in tqdm(loader):
        batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        query_t = batch.t[-1]
        if (query_t > label_t):
            label_tuple = dataset.get_node_label(query_t)
            if (label_tuple is None):
                break
            label_ts, label_srcs, labels = label_tuple[0], label_tuple[1], label_tuple[2]
            label_ts = label_ts.numpy()
            label_srcs = label_srcs.numpy()
            labels = labels.numpy()
            label_t = dataset.get_label_time()

            preds = []

            for i in range(0, label_srcs.shape[0]):
                node_id = label_srcs[i]
                pred_vec = forecaster.query_dict(node_id)
                preds.append(pred_vec)
                forecaster.update_dict(node_id, labels[i])

            np_pred = np.stack(preds, axis=0)
            np_true = labels
            ncdg_score = ndcg_score(np_true, np_pred, k=TOP_K)
            num_labels += label_ts.shape[0]
            total_ncdg += ncdg_score * label_ts.shape[0]

    metric_dict = {
    "ndcg": total_ncdg / num_labels,
    }
    return metric_dict


"""
train, val and test for one epoch only
"""

start_time = time.time()
metric_dict = test_n_upate(train_loader)
ncdg = metric_dict["ndcg"]
print ("testing persistant forecaster")
print(f'training ncdg: {ncdg:.4f}')
print("Persistant forecast on Training takes--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
val_dict = test_n_upate(val_loader)
print(f'Val ncdg: {val_dict["ndcg"]:.4f}')
print("Persistant forecast on validation takes--- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
test_dict = test_n_upate(test_loader)
print(f'Test ncdg: {test_dict["ndcg"]:.4f}')
print("Persistant forecast on Test takes--- %s seconds ---" % (time.time() - start_time))
dataset.reset_label_time()


