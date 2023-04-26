"""
JODIE
    This has been implemented with intuitions from the following sources:
    - https://github.com/twitter-research/tgn
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

    Spec.:
        - Memory Updater: RNN
        - Embedding Module: time

"""

import os.path as osp
import numpy as np

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
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


# set the clock for tracking the time
overall_start = time.time()

# set the global parameters
LR = 0.0001
batch_size = 200
n_epoch = 20
K = 10  # for computing metrics@k

memory_dim = time_dim = embedding_dim = 100

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class TimeEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        class NormalLinear(torch.nn.Linear):
            # From TGN code: From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.out_channels)

    def forward(self, x, last_update, t): 
        rel_t = t - last_update
        embeddings = x * (1 + self.embedding_layer(rel_t.to(x.dtype).unsqueeze(1)))

        return embeddings


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

# Memory Module
memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
    memory_updater_type='rnn'  # TGN: 'gru', JODIE & DyRep: 'rnn'
).to(device)

# Embedding Module
emb_module = TimeEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim
).to(device)

# Decoder: Link Predictor
link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(emb_module.parameters())
    | set(link_pred.parameters()), lr=LR)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def train():
    memory.train()
    emb_module.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
                                dtype=torch.long, device=device)

        all_nodes = torch.cat([src, pos_dst, neg_dst])
        n_id, n_idx, n_counts = torch.unique(all_nodes, return_inverse=True, return_counts=True)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get the current time for unique nodes
        all_times = torch.cat([t, t, t])
        _, idx_sorted = torch.sort(n_idx, stable=True)
        cum_sum = n_counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=device), cum_sum[:-1]))
        first_idices = idx_sorted[cum_sum]
        n_times = all_times[first_idices]

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = emb_module(z, last_update, n_times)

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]]).sigmoid()
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]]).sigmoid()

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader):
    memory.eval()
    emb_module.eval()
    link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs = [], []
    for batch in loader:
        batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
                                dtype=torch.long, device=device)

        all_nodes = torch.cat([src, pos_dst, neg_dst])
        n_id, n_idx, n_counts = torch.unique(all_nodes, return_inverse=True, return_counts=True)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get the time of the current edge for unique nodes
        all_times = torch.cat([t, t, t])
        _, idx_sorted = torch.sort(n_idx, stable=True)
        cum_sum = n_counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=device), cum_sum[:-1]))
        first_idices = idx_sorted[cum_sum]
        n_times = all_times[first_idices]

        z, last_update = memory(n_id)
        z = emb_module(z, last_update, n_times)

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        memory.update_state(src, pos_dst, t, msg)

    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())




# Train & Validation
print("=============================================")
print("=============*** JODIE model ***=============")
print("=============================================")

for epoch in range(1, n_epoch + 1):
    start_epoch_train = time.time()
    loss = train()
    end_epoch_train = time.time()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Elapsed Time (s): {end_epoch_train - start_epoch_train: .4f}')
    val_ap, val_auc = test(val_loader)
    print(f'\tVal AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')

# =========== Test
start_test_time = time.time()
test_ap, test_auc = test(test_loader)
end_test_time = time.time()
print("INFO: Final TEST Performance:")
print(f'\tTest AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, Elapsed Time (s): {end_test_time - start_test_time: .4f}')
print("=============================================")