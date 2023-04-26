"""
TGAT

    This has been implemented with intuitions from the following sources:
    - https://github.com/twitter-research/tgn
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

    Spec.:
        - No Memory Module
        - Embedding Module: Two-Layer Attention with Number of Neighbors = 20
        - Uniform Sampling of Neighbors (the default is sampling the most recent neighbors)
"""

import os.path as osp

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
# from torch_geometric.nn import TGNMemory
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    # LastNeighborLoader,
    TimeEncoder,
)
import time

# internal imports
from models.tgn import LastNeighborLoader

# set the global parameters
LR = 0.0001
batch_size = 200
n_epoch = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'JODIE')
dataset = JODIEDataset(path, name='wikipedia')
data = dataset[0]

# For small datasets, we can put the whole dataset on GPU and thus avoid
# expensive memory transfer costs for mini-batches:
data = data.to(device)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)

train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)

num_neighbors = 20
neighbor_loader = LastNeighborLoader(data.num_nodes, size=num_neighbors, device=device, sample_uniform=True)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv_1 = TransformerConv(in_channels, in_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)
        self.conv_2 = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)

        x = self.conv_1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv_2(x, edge_index, edge_attr)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h).sigmoid()


time_dim = embedding_dim = 100

gnn = GraphAttentionEmbedding(
    in_channels=embedding_dim,
    out_channels=embedding_dim,
    msg_dim=data.msg.size(-1),
    time_enc=TimeEncoder(time_dim),
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

optimizer = torch.optim.Adam(
    set(gnn.parameters()) | set(link_pred.parameters()), lr=LR)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def train():
    gnn.train()
    link_pred.train()

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

        z = torch.rand((n_id.size(0), embedding_dim), device=device)  # since the datasets mostly do not have node features
        z = gnn(z, data.t[edge_index[0]].to(device), edge_index, data.t[e_id].to(device), data.msg[e_id].to(device))

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader):
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

        z = torch.rand((n_id.size(0), embedding_dim), device=device)  # since the datasets mostly do not have node features
        z = gnn(z, data.t[edge_index[0]].to(device), edge_index, data.t[e_id].to(device), data.msg[e_id].to(device))

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        y_pred = torch.cat([pos_out, neg_out], dim=0).cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        neighbor_loader.insert(src, pos_dst)

    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())



print("=============================================")
print("==============*** TGAT model ***=============")
print("=============================================")

# =========== Train & Validation
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