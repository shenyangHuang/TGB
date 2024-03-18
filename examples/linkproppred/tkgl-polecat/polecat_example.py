import numpy as np
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator

DATA = "tkgl-polecat"

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


#load the ns samples first
dataset.load_val_ns()
for i, (src, dst, t, rel) in enumerate(zip(val_data.src, val_data.dst, val_data.t, val_data.edge_type)):
    #must use np array to query
    neg_batch_list = neg_sampler.query_batch(np.array([src]), np.array([dst]), np.array([t]), edge_type=np.array([rel]), split_mode='val')

print ("retrieved all negative samples")


# #* load numpy arrays instead
# from tgb.linkproppred.dataset import LinkPropPredDataset

# # data loading
# dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)
# data = dataset.full_data  
# metric = dataset.eval_metric
# sources = dataset.full_data['sources']
# print (sources.dtype)

