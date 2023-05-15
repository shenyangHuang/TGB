"""
JODIE
    This has been implemented with intuitions from the following sources:
    - https://github.com/twitter-research/tgn
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

    Spec.:
        - Memory Updater: RNN
        - Embedding Module: time
        - (No time encoder)

"""

import os.path as osp
import numpy as np

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TransformerConv
import math
import time

# internal imports
from tgb.linkproppred.negative_sampler import *
from tgb.linkproppred.evaluate import Evaluator
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.tgn_memory import TGNMemory
from modules.emb_module import TimeEmbedding
from modules.decoder import LinkPredictor


overall_start = time.time()

LR = 0.0001
batch_size = 200
k = 10  # for computing metrics@k
n_epoch = 50
rnd_seed = 1234
dataset_name = "wikipedia"

memory_dim = time_dim = embedding_dim = 100

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "JODIE")
dataset = JODIEDataset(path, name=dataset_name)
data = dataset[0]

# For small datasets, we can put the whole dataset on GPU and thus avoid
# expensive memory transfer costs for mini-batches:
data = data.to(device)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# split the data
train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15
)

train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)


# Memory Module
memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
    memory_updater_cell="rnn",  # TGN: 'gru', JODIE & DyRep: 'rnn'
).to(device)

# Embedding Module
emb_module = TimeEmbedding(in_channels=memory_dim, out_channels=embedding_dim).to(
    device
)

# Decoder: Link Predictor
link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters())
    | set(emb_module.parameters())
    | set(link_pred.parameters()),
    lr=LR,
)
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
        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

        all_nodes = torch.cat([src, pos_dst, neg_dst])
        n_id, n_idx, n_counts = torch.unique(
            all_nodes, return_inverse=True, return_counts=True
        )
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get the current time for unique nodes
        all_times = t.repeat(3)
        _, idx_sorted = torch.sort(n_idx, stable=True)
        cum_sum = n_counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=device), cum_sum[:-1]))
        first_idices = idx_sorted[cum_sum]
        n_times = all_times[first_idices]

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = emb_module(z, last_update, n_times)

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

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
def test_one_vs_many(loader, neg_sampler):
    """
    Evaluated the dynamic link prediction in an exhaustive manner
    """
    memory.eval()
    emb_module.eval()
    link_pred.eval()

    torch.manual_seed(rnd_seed)  # Ensure deterministic sampling across epochs.

    hist_at_k_list, mrr_list = [], []

    for pos_batch in loader:
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        batch_list = neg_sampler.sample(pos_batch)
        for pos_batch_idx, batch in enumerate(batch_list):
            src, dst, y_true = batch["src"], batch["dst"], batch["y"]
            all_nodes = torch.cat([src, dst])
            n_id, n_idx, n_counts = torch.unique(
                all_nodes, return_inverse=True, return_counts=True
            )
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get the current time for unique nodes
            all_times = pos_t[pos_batch_idx].repeat(all_nodes.size(0))
            _, idx_sorted = torch.sort(n_idx, stable=True)
            cum_sum = n_counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0], device=device), cum_sum[:-1]))
            first_idices = idx_sorted[cum_sum]
            n_times = all_times[first_idices]

            # Get updated memory of all nodes involved in the computation.
            z, last_update = memory(n_id)
            z = emb_module(z, last_update, n_times)

            y_pred = link_pred(z[assoc[src]], z[assoc[dst]])

            # hist@k & MRR
            metrics_mrr_rnk = evaluator.eval_rnk_metrics(
                y_pred_pos=y_pred[y_true == 1].squeeze(dim=-1).cpu(),
                y_pred_neg=y_pred[y_true == 0].squeeze(dim=-1).cpu(),
                type_info="torch",
                k=k,
            )
            hist_at_k_list.append(metrics_mrr_rnk[f"hits@{k}"])
            mrr_list.append(metrics_mrr_rnk["mrr"])

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(pos_src, pos_dst, pos_t, pos_msg)

    perf_metrics = {
        f"hits@{k}": float(torch.tensor(hist_at_k_list).mean()),
        "mrr": float(torch.tensor(mrr_list).mean()),
    }
    return perf_metrics


@torch.no_grad()
def test_one_vs_one(loader):
    memory.eval()
    emb_module.eval()
    link_pred.eval()

    torch.manual_seed(rnd_seed)  # Ensure deterministic sampling across epochs.

    aps, aucs = [], []
    for batch in loader:
        batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

        all_nodes = torch.cat([src, pos_dst, neg_dst])
        n_id, n_idx, n_counts = torch.unique(
            all_nodes, return_inverse=True, return_counts=True
        )
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get the current time for unique nodes
        all_times = t.repeat(3)
        _, idx_sorted = torch.sort(n_idx, stable=True)
        cum_sum = n_counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=device), cum_sum[:-1]))
        first_idices = idx_sorted[cum_sum]
        n_times = all_times[first_idices]

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = emb_module(z, last_update, n_times)

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        y_pred = torch.cat([pos_out, neg_out], dim=0).cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0
        )

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        memory.update_state(src, pos_dst, t, msg)

    perf_metrics = {
        "ap": float(torch.tensor(aps).mean()),
        "auc": float(torch.tensor(aucs).mean()),
    }

    return perf_metrics


print("==========================================================")
print("=================*** JODIE model: ONE-VS-MANY ***===========")
print("==========================================================")

evaluator = Evaluator(name=dataset_name)

# ==================================================== Train & Validation
start_train_val = time.time()
for epoch in range(1, n_epoch + 1):
    start_epoch_train = time.time()
    loss = train()
    end_epoch_train = time.time()
    print(
        f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Elapsed Time (s): {end_epoch_train - start_epoch_train: .4f}"
    )
    val_perf_metrics = test_one_vs_one(val_loader)  # used for validation only
    val_ap, val_auc = val_perf_metrics["ap"], val_perf_metrics["auc"]
    print(f"\tVal AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}")
end_train_val = time.time()
print(f"Train & Validation: Elapsed Time (s): {end_train_val - start_train_val: .4f}")

# ==================================================== Test
# negative sampler
num_neg_e_per_pos = 200
NEG_SAMPLE_MODE = "RND"  # ['RND', 'HIST_RND']
if NEG_SAMPLE_MODE == "RND":
    # Negative Sampler: RANDOM
    neg_sampler = NegativeEdgeSampler_RND(
        first_dst_id=min_dst_idx,
        last_dst_id=max_dst_idx,
        num_neg_e=num_neg_e_per_pos,
        device=device,
        rnd_seed=rnd_seed,
    )
elif NEG_SAMPLE_MODE == "HIST_RND":
    # Negative Sampler: HISTORICAL-RANDOM
    neg_sampler = NegativeEdgeSampler_HIST_RND(
        first_dst_id=min_dst_idx,
        last_dst_id=max_dst_idx,
        train_data=train_data,
        val_data=val_data,
        num_neg_e=num_neg_e_per_pos,
        device=device,
        rnd_seed=rnd_seed,
    )
else:
    raise ValueError("Undefined Negative Sampling Strategy!")

# testing ...
start_test = time.time()
perf_metrics_test = test_one_vs_many(test_loader, neg_sampler)
end_test = time.time()

print(
    f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY --- NS-Mode: {NEG_SAMPLE_MODE} <<< "
)
for perf_name, perf_value in perf_metrics_test.items():
    print(f"\tTest: {perf_name}: {perf_value: .4f}")
print(f"Test: Elapsed Time (s): {end_test - start_test: .4f}")


overall_end = time.time()
print(f"Overall Elapsed Time (s): {overall_end - overall_start: .4f}")
print("==============================================================")
