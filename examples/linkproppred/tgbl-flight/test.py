from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from torch_geometric.loader import TemporalDataLoader

dataset = PyGLinkPropPredDataset(name='tgbl-flight', root="datasets")
val_mask = dataset.val_mask

data = dataset.get_TemporalData()
val_data = data[val_mask]

neg_sampler = dataset.negative_sampler
val_loader = TemporalDataLoader(val_data, batch_size=200)

neg_sampler = dataset.negative_sampler
dataset.load_val_ns()

batch = next(iter(val_loader))
print (batch.src[0], " , ", batch.dst[0], " , ", batch.t[0])
for i in range(len(batch.t)):
    if (batch.src[i].item() == 12 and batch.dst[i].item() == 4):
        print (batch.src[i], " , ", batch.dst[i], " , ", batch.t[i])


# for batch in iter(val_loader):
#     for i in range(len(batch.t)):
#         if (batch.t[i].item() == 1638144000):
#             print (batch.src[i], " , ", batch.dst[i], " , ", batch.t[i])
#             neg_batch_list = neg_sampler.query_batch(batch.src[i], batch.dst[i], batch.t[i], split_mode="val")

# neg_batch_list = neg_sampler.query_batch(batch.src, batch.dst, batch.t, split_mode="val")
# print (neg_batch_list)