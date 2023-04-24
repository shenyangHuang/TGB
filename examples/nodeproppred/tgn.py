# This code achieves a performance of around 96.60%. However, it is not
# directly comparable to the results reported by the TGN paper since a
# slightly different evaluation setup is used here.
# In particular, predictions in the same batch are made in parallel, i.e.
# predictions for interactions later in the batch have no access to any
# information whatsoever about previous interactions in the same batch.
# On the contrary, when sampling node neighborhoods for interactions later in
# the batch, the TGN paper code has access to previous interactions in the
# batch.
# While both approaches are correct, together with the authors of the paper we
# decided to present this version here as it is more realsitic and a better
# test bed for future methods.

import os.path as osp
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import ndcg_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

from tgb.nodeproppred.dataset_pyg import PyGNodePropertyDataset
import torch.nn.functional as F
import time



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

name = "lastfmgenre"
dataset = PyGNodePropertyDataset(name=name, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask

num_classes = dataset.num_classes
data = dataset.data[0]
data = data.to(device)


train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

batch_size = 200

train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)

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

class NodePredictor(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_node = Linear(in_dim, in_dim)
        self.out = Linear(in_dim, out_dim)

    def forward(self, node_embed):
        h = self.lin_node(node_embed)
        h = h.relu()
        h = self.out(h)
        output = F.log_softmax(h, dim=-1)
        return output



memory_dim = time_dim = embedding_dim = 100

memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

node_pred = NodePredictor(in_dim=embedding_dim, out_dim=num_classes).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(node_pred.parameters()), lr=0.0001)

criterion = torch.nn.CrossEntropyLoss()
# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def plot_curve(scores, out_name):
    plt.plot(scores, color="#e34a33")
    plt.ylabel("score")
    plt.savefig(out_name + ".pdf")
    plt.close()



def train(plotting=True):
    """
    also want to track the training curve now
    """
    train_ndcg = []
    memory.train()
    gnn.train()
    node_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    label_t = dataset.get_label_time() #check when does the first label start
    TOP_Ks = [5,10,20]
    total_ncdg = np.zeros(len(TOP_Ks)) 
    track_ncdg = []
    num_labels = 0

    print ("training starts")
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg


        query_t = batch.t[-1]
        if (query_t > label_t):
            label_tuple = dataset.get_node_label(query_t)
            label_ts, label_srcs, labels = label_tuple[0], label_tuple[1], label_tuple[2]
            label_t = dataset.get_label_time()
            label_srcs = label_srcs.to(device)


            #! problem with current implementation
            '''
            to feed into gnn requires edge index and last update
            however the nodes might not have been upated recently
            rel_t = last_update[edge_index[0]] - t
            IndexError: index 533 is out of bounds for dimension 0 with size 46
            '''

            # must require edges for time dependent embedding
            # apply edge index as self-loops
            self_msg = torch.ones(label_ts.shape[0],1).float().to(device)

            #.unique()
            n_id = label_srcs
            n_id, mem_edge_index, _ = neighbor_loader(n_id)
            z, last_update = memory(n_id)

            #! use self loop if there is no memory update yet
            if (mem_edge_index.shape[1] != last_update.shape[0]):
                self_loop = torch.zeros(2,label_ts.shape[0], dtype=int).to(device)
                mem_edge_index = self_loop
                
            z = gnn(z, last_update, mem_edge_index, label_ts.to(device),
                self_msg)

            z = z[0:labels.shape[0]]

            #loss and metric computation

            pred = node_pred(z)
            loss = criterion(pred, labels.to(device))
            np_pred = pred.cpu().detach().numpy()
            np_true = labels.cpu().detach().numpy()

            for i in range(len(TOP_Ks)):
                ncdg_score = ndcg_score(np_true, np_pred, k=TOP_Ks[i])
                total_ncdg[i] += ncdg_score * label_ts.shape[0]
                if (TOP_Ks[i] == 10):
                    track_ncdg.append(ncdg_score)

            num_labels += label_ts.shape[0]

            loss.backward()
            optimizer.step()
            total_loss += float(loss)

        #! only memory update here
        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        memory.detach()

    if (plotting):
        plot_curve(track_ncdg, "training_curve_ncdg10")

    metric_dict = {
    "ce":total_loss / num_labels,
    }

    for i in range(len(TOP_Ks)):
        k = TOP_Ks[i]
        metric_dict["ndcg_" + str(k)] = total_ncdg[i] / num_labels
    return metric_dict


@torch.no_grad()
def test(loader):
    memory.eval()
    gnn.eval()
    node_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.
    total_ncdg = 0
    label_t = dataset.get_label_time() #check when does the first label start
    #TOP_K = 10
    TOP_Ks = [5,10,20]
    total_ncdg = np.zeros(len(TOP_Ks)) 
    num_labels = 0

    print ("testing starts")
    for batch in tqdm(loader):
        batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        query_t = batch.t[-1]
        if (query_t > label_t):
            label_tuple = dataset.get_node_label(query_t)
            if (label_tuple is None):
                break
            label_ts, label_srcs, labels = label_tuple[0], label_tuple[1], label_tuple[2]
            label_t = dataset.get_label_time()
            label_srcs = label_srcs.to(device)

            self_msg = torch.ones(label_ts.shape[0],1).float().to(device)

            n_id = label_srcs
            n_id, mem_edge_index, _ = neighbor_loader(n_id)
            z, last_update = memory(n_id)
            z = gnn(z, last_update, mem_edge_index, label_ts.to(device),
                self_msg)

            z = z[0:labels.shape[0]]
            #loss and metric computation
            pred = node_pred(z)
            np_pred = pred.cpu().detach().numpy()
            np_true = labels.cpu().detach().numpy()

            for i in range(len(TOP_Ks)):
                ncdg_score = ndcg_score(np_true, np_pred, k=TOP_Ks[i])
                total_ncdg[i] += ncdg_score * label_ts.shape[0]

            num_labels += label_ts.shape[0]

        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

    metric_dict = {}

    for i in range(len(TOP_Ks)):
        k = TOP_Ks[i]
        metric_dict["ndcg_" + str(k)] = total_ncdg[i] / num_labels
    return metric_dict

for epoch in range(1, 51):
    start_time = time.time()
    train_dict = train()
    print ("------------------------------------")
    print(f'training Epoch: {epoch:02d}')
    print (train_dict)
    print("Training takes--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    val_dict = test(val_loader)
    print (val_dict)
    print("Validation takes--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    test_dict = test(test_loader)
    print (test_dict)
    dataset.reset_label_time()
    print("Test takes--- %s seconds ---" % (time.time() - start_time))
    print ("------------------------------------")




