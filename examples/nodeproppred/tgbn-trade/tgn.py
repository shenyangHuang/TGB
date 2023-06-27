import timeit
import argparse
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

from modules.decoder import NodePredictor
from modules.emb_module import GraphAttentionEmbedding
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from tgb.nodeproppred.evaluate import Evaluator
from tgb.utils.utils import set_random_seed
from tgb.utils.stats import plot_curve

parser = argparse.ArgumentParser(description='parsing command line arguments as hyperparameters')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='random seed to use')
parser.parse_args()
args = parser.parse_args()
# setting random seed
seed = int(args.seed) #1,2,3,4,5
torch.manual_seed(seed)
set_random_seed(seed)

# hyperparameters
lr = 0.0001
epochs = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name = "tgbn-trade"
dataset = PyGNodePropPredDataset(name=name, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask

eval_metric = dataset.eval_metric
num_classes = dataset.num_classes
data = dataset.get_TemporalData()
data = data.to(device)

evaluator = Evaluator(name=name)


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

memory_dim = time_dim = embedding_dim = 100

memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

gnn = (
    GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    )
    .to(device)
    .float()
)

node_pred = NodePredictor(in_dim=embedding_dim, out_dim=num_classes).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters()) | set(node_pred.parameters()),
    lr=lr,
)

criterion = torch.nn.CrossEntropyLoss()
# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def plot_curve(scores, out_name):
    plt.plot(scores, color="#e34a33")
    plt.ylabel("score")
    plt.savefig(out_name + ".pdf")
    plt.close()


def process_edges(src, dst, t, msg):
    if src.nelement() > 0:
        # msg = msg.to(torch.float32)
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)


def train():
    memory.train()
    gnn.train()
    node_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    label_t = dataset.get_label_time()  # check when does the first label start
    num_labels = 0
    total_score = 0

    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        query_t = batch.t[-1]
        # check if this batch moves to the next day
        if query_t > label_t:
            # find the node labels from the past day
            label_tuple = dataset.get_node_label(query_t)
            label_ts, label_srcs, labels = (
                label_tuple[0],
                label_tuple[1],
                label_tuple[2],
            )
            label_t = dataset.get_label_time()
            label_srcs = label_srcs.to(device)

            # Process all edges that are still in the past day
            previous_day_mask = batch.t < label_t
            process_edges(
                src[previous_day_mask],
                dst[previous_day_mask],
                t[previous_day_mask],
                msg[previous_day_mask],
            )
            # Reset edges to be the edges from tomorrow so they can be used later
            src, dst, t, msg = (
                src[~previous_day_mask],
                dst[~previous_day_mask],
                t[~previous_day_mask],
                msg[~previous_day_mask],
            )

            """
            modified for node property prediction
            1. sample neighbors from the neighbor loader for all nodes to be predicted
            2. extract memory from the sampled neighbors and the nodes
            3. run gnn with the extracted memory embeddings and the corresponding time and message
            """
            n_id = label_srcs
            n_id_neighbors, mem_edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id_neighbors] = torch.arange(n_id_neighbors.size(0), device=device)

            z, last_update = memory(n_id_neighbors)

            z = gnn(
                z,
                last_update,
                mem_edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )
            z = z[assoc[n_id]]

            # loss and metric computation
            pred = node_pred(z)

            loss = criterion(pred, labels.to(device))
            np_pred = pred.cpu().detach().numpy()
            np_true = labels.cpu().detach().numpy()

            input_dict = {
                "y_true": np_true,
                "y_pred": np_pred,
                "eval_metric": [eval_metric],
            }
            result_dict = evaluator.eval(input_dict)
            score = result_dict[eval_metric]
            total_score += score
            num_labels += label_ts.shape[0]

            loss.backward()
            optimizer.step()
            total_loss += float(loss)

        # Update memory and neighbor loader with ground-truth state.
        process_edges(src, dst, t, msg)
        memory.detach()

    metric_dict = {
        "ce": total_loss / num_labels,
    }
    metric_dict[eval_metric] = total_score / num_labels
    return metric_dict


@torch.no_grad()
def test(loader):
    memory.eval()
    gnn.eval()
    node_pred.eval()

    total_score = 0
    label_t = dataset.get_label_time()  # check when does the first label start
    num_labels = 0

    for batch in tqdm(loader):
        batch = batch.to(device)
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        query_t = batch.t[-1]
        if query_t > label_t:
            label_tuple = dataset.get_node_label(query_t)
            if label_tuple is None:
                break
            label_ts, label_srcs, labels = (
                label_tuple[0],
                label_tuple[1],
                label_tuple[2],
            )
            label_t = dataset.get_label_time()
            label_srcs = label_srcs.to(device)

            # Process all edges that are still in the past day
            previous_day_mask = batch.t < label_t
            process_edges(
                src[previous_day_mask],
                dst[previous_day_mask],
                t[previous_day_mask],
                msg[previous_day_mask],
            )
            # Reset edges to be the edges from tomorrow so they can be used later
            src, dst, t, msg = (
                src[~previous_day_mask],
                dst[~previous_day_mask],
                t[~previous_day_mask],
                msg[~previous_day_mask],
            )

            """
            modified for node property prediction
            1. sample neighbors from the neighbor loader for all nodes to be predicted
            2. extract memory from the sampled neighbors and the nodes
            3. run gnn with the extracted memory embeddings and the corresponding time and message
            """
            n_id = label_srcs
            n_id_neighbors, mem_edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id_neighbors] = torch.arange(n_id_neighbors.size(0), device=device)

            z, last_update = memory(n_id_neighbors)
            z = gnn(
                z,
                last_update,
                mem_edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )
            z = z[assoc[n_id]]

            # loss and metric computation
            pred = node_pred(z)
            np_pred = pred.cpu().detach().numpy()
            np_true = labels.cpu().detach().numpy()

            input_dict = {
                "y_true": np_true,
                "y_pred": np_pred,
                "eval_metric": [eval_metric],
            }
            result_dict = evaluator.eval(input_dict)
            score = result_dict[eval_metric]
            total_score += score
            num_labels += label_ts.shape[0]

        process_edges(src, dst, t, msg)

    metric_dict = {}
    metric_dict[eval_metric] = total_score / num_labels
    return metric_dict


train_curve = []
val_curve = []
test_curve = []
max_val_score = 0  #find the best test score based on validation score
best_test_idx = 0
for epoch in range(1, epochs + 1):
    start_time = timeit.default_timer()
    train_dict = train()
    print("------------------------------------")
    print(f"training Epoch: {epoch:02d}")
    print(train_dict)
    train_curve.append(train_dict[eval_metric])
    print("Training takes--- %s seconds ---" % (timeit.default_timer() - start_time))
    
    start_time = timeit.default_timer()
    val_dict = test(val_loader)
    print(val_dict)
    val_curve.append(val_dict[eval_metric])
    if (val_dict[eval_metric] > max_val_score):
        max_val_score = val_dict[eval_metric]
        best_test_idx = epoch - 1
    print("Validation takes--- %s seconds ---" % (timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    test_dict = test(test_loader)
    print(test_dict)
    test_curve.append(test_dict[eval_metric])
    print("Test takes--- %s seconds ---" % (timeit.default_timer() - start_time))
    print("------------------------------------")
    dataset.reset_label_time()


# code for plotting
plot_curve(train_curve, "train_curve")
plot_curve(val_curve, "val_curve")
plot_curve(test_curve, "test_curve")

max_test_score = test_curve[best_test_idx]
print("------------------------------------")
print("------------------------------------")
print ("best val score: ", max_val_score)
print ("best validation epoch   : ", best_test_idx + 1)
print ("best test score: ", max_test_score)
