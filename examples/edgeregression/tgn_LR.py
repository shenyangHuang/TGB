"""
TGN pyG: Link Regression

source: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
"""

import os.path as osp
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, mean_squared_error
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
from tgb.edgeregression.dataset_pyg import PyGEdgeRegressDataset
import math
import time
import numpy as np



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


class LinkRegressor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


# tracking the time
end_to_end_start = time.time()

# set parameters
batch_size = 200
memory_dim = time_dim = embedding_dim = 100
LEARNING_RATE = 0.0001
n_epochs = 50

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data loading
name = "un_trade"
dataset = PyGEdgeRegressDataset(name=name, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.data[0]
data.t = data.t.long()
data.y = data.y.float()
data = data.to(device)


train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]



# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)

neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

# model initialization
# memory module
memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)
# GAT
gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)
# decoder: link regressor
link_regress = LinkRegressor(in_channels=embedding_dim).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_regress.parameters()), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def train():
    memory.train()
    gnn.train()
    link_regress.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg, y = batch.src, batch.dst, batch.t, batch.msg, batch.y

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))

        pos_out = link_regress(z[assoc[src]], z[assoc[pos_dst]])
        loss = criterion(pos_out.squeeze(), y)

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
    link_regress.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    mse, rmse = [], []
    for batch in loader:
        batch = batch.to(device)
        src, pos_dst, t, msg, y = batch.src, batch.dst, batch.t, batch.msg, batch.y

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))

        pos_out = link_regress(z[assoc[src]], z[assoc[pos_dst]])

        y_pred = pos_out.cpu().numpy()
        y_true = y.cpu().numpy()

        mse.append(mean_squared_error(y_true, y_pred))
        rmse.append(math.sqrt(mean_squared_error(y_true, y_pred)))

        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

    return float(torch.tensor(mse).mean()), float(torch.tensor(rmse).mean())


for epoch in range(n_epochs):
    epoch_start_time = time.time()
    loss = train()
    epoch_elapsed_time = time.time() - epoch_start_time
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Elapsed Time (sec.): {epoch_elapsed_time: .4f}')
    val_mse, val_rmse = test(val_loader)
    test_mse, test_rmse = test(test_loader)
    print(f'\tVal MSE: {val_mse:.4f}, Val RMSE: {val_rmse:.4f}')
    print(f'\tTest MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}')

print(f'End to end elapsed time (sec.): {time.time() - end_to_end_start: .4f}')