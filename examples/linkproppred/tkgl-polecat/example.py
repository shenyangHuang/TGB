from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset


DATA = "tkgl-polecat"

# data loading
dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
metric = dataset.eval_metric


timestamp = data.t
head = data.src
tail = data.dst
edge_type = data.edge_type #relation

print (edge_type)

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]