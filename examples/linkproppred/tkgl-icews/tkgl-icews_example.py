import numpy as np
import timeit
from tqdm import tqdm
import sys
import os
import os.path as osp

#internal imports 
tgb_modules_path = osp.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(tgb_modules_path)

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from torch_geometric.loader import TemporalDataLoader
from tgb.linkproppred.evaluate import Evaluator

DATA = "tkgl-icews"

# data loading
dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
metric = dataset.eval_metric

print ("there are {} nodes and {} edges".format(dataset.num_nodes, dataset.num_edges))
print ("there are {} relation types".format(dataset.num_rels))


timestamp = data.t
head = data.src
tail = data.dst
edge_type = data.edge_type #relation
neg_sampler = dataset.negative_sampler

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]


metric = dataset.eval_metric
evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler

BATCH_SIZE = 200
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

start_time = timeit.default_timer()
#load the ns samples first
dataset.load_val_ns()
for batch in tqdm(val_loader):
    src, pos_dst, t, msg, rel = batch.src, batch.dst, batch.t, batch.msg, batch.edge_type
    neg_batch_list = neg_sampler.query_batch(src.detach().cpu().numpy(), pos_dst.detach().cpu().numpy(), t.detach().cpu().numpy(), rel.detach().cpu().numpy(), split_mode='val')
print ("loading ns samples from validation", timeit.default_timer() - start_time)
# for i, (src, dst, t, rel) in enumerate(zip(val_data.src, val_data.dst, val_data.t, val_data.edge_type)):
#     #must use np array to query
#     neg_batch_list = neg_sampler.query_batch(np.array([src]), np.array([dst]), np.array([t]), edge_type=np.array([rel]), split_mode='val')

start_time = timeit.default_timer()
dataset.load_test_ns()
for batch in test_loader:
    src, pos_dst, t, msg, rel = batch.src, batch.dst, batch.t, batch.msg, batch.edge_type
    neg_batch_list = neg_sampler.query_batch(src.detach().cpu().numpy(), pos_dst.detach().cpu().numpy(), t.detach().cpu().numpy(), rel.detach().cpu().numpy(), split_mode='test')
print ("loading ns samples from test", timeit.default_timer() - start_time)
# for i, (src, dst, t, rel) in enumerate(zip(test_data.src, test_data.dst, test_data.t, test_data.edge_type)):
#     #must use np array to query
#     neg_batch_list = neg_sampler.query_batch(np.array([src]), np.array([dst]), np.array([t]), edge_type=np.array([rel]), split_mode='test')
print ("retrieved all negative samples")


# #* load numpy arrays instead
# from tgb.linkproppred.dataset import LinkPropPredDataset

# # data loading
# dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)
# data = dataset.full_data  
# metric = dataset.eval_metric
# sources = dataset.full_data['sources']
# print (sources.dtype)

