import timeit
import argparse
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.utils.utils import set_random_seed
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding

parser = argparse.ArgumentParser(description='parsing command line arguments as hyperparameters')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='random seed to use')
parser.parse_args()
args = parser.parse_args()
# setting random seed
seed = int(args.seed) #1,2,3,4,5
print ("setting random seed to be", seed)
torch.manual_seed(seed)
set_random_seed(seed)
n_epoch = 50


name = "amazonreview"
dataset = PyGLinkPropPredDataset(name=name, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())


train_loader = TemporalDataLoader(train_data, batch_size=200)
val_loader = TemporalDataLoader(val_data, batch_size=200)
test_loader = TemporalDataLoader(test_data, batch_size=200)


@torch.no_grad()
def test_one_vs_many(loader, neg_sampler, split_mode):
    for pos_batch in loader:
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)


print("==========================================================")
print("=================*** TGN model: ONE-VS-MANY ***===========")
print("==========================================================")

evaluator = Evaluator(name=name)

# negative sampler
NEG_SAMPLE_MODE = "hist_rnd"
neg_sampler = dataset.negative_sampler

# ==================================================== Train & Validation
# loading the validation negative samples
dataset.load_val_ns()

test_one_vs_many(val_loader,neg_sampler, split_mode="val")

# ==================================================== Test
# loading the test negative samples
dataset.load_test_ns()


test_one_vs_many(test_loader,neg_sampler, split_mode="test")