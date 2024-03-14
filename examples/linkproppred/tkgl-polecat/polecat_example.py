from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset


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

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]


from tgb.linkproppred.dataset import LinkPropPredDataset

# data loading
dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)
data = dataset.full_data  
metric = dataset.eval_metric
sources = dataset.full_data['sources']
print (sources.dtype)

