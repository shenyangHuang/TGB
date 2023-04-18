"""
TGN pyG: Link Prediction

source: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
"""

import os.path as osp
import numpy as np

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
# from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
import math
import time

# internal imports
from models.tgn import TGNMemory
from edgepred_utils import *



overall_start = time.time()

LR = 0.0001
batch_size = 200
K = 10  # for computing metrics@k
n_epoch = 50

memory_dim = time_dim = embedding_dim = 100

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# data loading
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'JODIE')
dataset = JODIEDataset(path, name='wikipedia')
data = dataset[0]

# For small datasets, we can put the whole dataset on GPU and thus avoid
# expensive memory transfer costs for mini-batches:
data = data.to(device)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# split the data
train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)

train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)

# neighhorhood sampler
neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
    memory_updater_type='gru'
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()), lr=LR)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def train():
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
                                dtype=torch.long, device=device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs = [], []
    for batch in loader:
        batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
                                dtype=torch.long, device=device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())



@torch.no_grad()
def test_exh(loader):
    """
    Evaluated the dynamic link prediction in an exhaustive manner
    """
    memory.eval()
    gnn.eval()
    link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs = [], []
    for batch in loader:
        batch = batch.to(device)
        src_orig, pos_dst_orig, t_orig, msg_orig = batch.src, batch.dst, batch.t, batch.msg
        batch_size = src_orig.size(0)
        
        edges_per_node = gen_eval_set_for_batch(src_orig, pos_dst_orig)

        pos_out_agg, neg_out_agg = [], []
        prec_at_k_list, rec_at_k_list, f1_at_k_list = [], [], []
        for pos_s in src_orig:
            pos_s = pos_s.item()
            pos_dst = torch.tensor(edges_per_node[pos_s]['pos'], device=device)
            pos_src = torch.tensor([pos_s for _ in range(len(edges_per_node[pos_s]['pos']))], device=device)

            neg_dst = torch.tensor(edges_per_node[pos_s]['neg'], device=device)
            neg_src = torch.tensor([pos_s for _ in range(len(edges_per_node[pos_s]['neg']))], device=device)

            # positive edges 
            pos_n_id = torch.cat([pos_src, pos_dst]).unique()
            pos_n_id, pos_edge_index, pos_e_id = neighbor_loader(pos_n_id)
            assoc[pos_n_id] = torch.arange(pos_n_id.size(0), device=device)

            pos_z, pos_last_update = memory(pos_n_id)
            pos_z = gnn(pos_z, pos_last_update, pos_edge_index, data.t[pos_e_id].to(device),
                    data.msg[pos_e_id].to(device))

            pos_out = link_pred(pos_z[assoc[pos_src]], pos_z[assoc[pos_dst]])
            pos_out_agg.append(pos_out)

            # negative edges
            n_neg_iter = math.ceil(len(neg_dst) / batch_size)
            for n_iter_idx in range(n_neg_iter):
                n_start_idx = n_iter_idx * batch_size
                n_end_idx = min(n_start_idx + batch_size, len(neg_dst))

                neg_src_iter = neg_src[n_start_idx: n_end_idx]
                neg_dst_iter = neg_dst[n_start_idx: n_end_idx]

                neg_n_id = torch.cat([neg_src_iter, neg_dst_iter]).unique()
                neg_n_id, neg_edge_index, neg_e_id = neighbor_loader(neg_n_id)
                assoc[neg_n_id] = torch.arange(neg_n_id.size(0), device=device)
                neg_z, neg_last_update = memory(neg_n_id)
                neg_z = gnn(neg_z, neg_last_update, neg_edge_index, data.t[neg_e_id].to(device),
                        data.msg[neg_e_id].to(device))

                neg_out = link_pred(neg_z[assoc[neg_src_iter]], neg_z[assoc[neg_dst_iter]])
                neg_out_agg.append(neg_out)

            # precision@k & recall@k should be calculated for each positive source node separately
            # y_true, y_pred_proba
            y_pred_proba_src = torch.cat([pos_out.squeeze(dim=-1), neg_out.squeeze(dim=-1)], dim=0).sigmoid().cpu()
            y_true_src = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)
            prec_at_k, rec_at_k, f1_at_k = metric_at_k_score(y_true_src, y_pred_proba_src, k=K, pos_label=1)
            prec_at_k_list.append(prec_at_k)
            rec_at_k_list.append(rec_at_k)
            f1_at_k_list.append(f1_at_k)

        # metrics per processing each batch of positive edges
        pos_out_agg = torch.cat([pos_out.squeeze(dim=-1) for pos_out in pos_out_agg], dim=0)
        neg_out_agg = torch.cat([neg_out.squeeze(dim=-1) for neg_out in neg_out_agg], dim=0)
        y_pred = torch.cat([pos_out_agg, neg_out_agg], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out_agg.size(0)),
             torch.zeros(neg_out_agg.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src_orig, pos_dst_orig, t_orig, msg_orig)
        neighbor_loader.insert(src_orig, pos_dst_orig)

    perf_metrics = {'ap': float(torch.tensor(aps).mean()),
                    'auc': float(torch.tensor(aucs).mean()),
                    'prec@k': float(torch.tensor(prec_at_k_list).mean()),
                    'rec@k': float(torch.tensor(rec_at_k_list).mean()),
                    'f1@k': float(torch.tensor(f1_at_k_list).mean()),
                    }

    return perf_metrics


# Train & Validation
print("INFO: =======================================")
print("INFO: ==========*** TGN model ***==========")
print("INFO: =======================================")

for epoch in range(n_epoch):
    start_epoch_train = time.time()
    loss = train()
    end_epoch_train = time.time()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Elapsed Time (s): {end_epoch_train - start_epoch_train: .4f}')
    val_ap, val_auc = test(val_loader)
    print(f'\tVal AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')

# ===========
# Final Test: Since the exhaustive test takes more time, we only do it at the end of training procedure
start_test = time.time()
perf_metrics_test = test_exh(test_loader)
end_test = time.time()
for perf_name, perf_value in perf_metrics_test.items():
    print(f"\tTest: {perf_name}: {perf_value: .4f}")
print(f'Test: Elapsed Time (s): {end_test - start_test: .4f}')

overall_end = time.time()
print(f'Overal Elapsed Time (s): {overall_end - overall_start: .4f}')
print("INFO: =======================================")