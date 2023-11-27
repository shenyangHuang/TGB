from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm
import datetime
import csv


dataset = PyGLinkPropPredDataset(name='tgbl-flight', root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask

data = dataset.get_TemporalData()
train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

neg_sampler = dataset.negative_sampler
train_loader = TemporalDataLoader(train_data, batch_size=200)
val_loader = TemporalDataLoader(val_data, batch_size=200)
test_loader = TemporalDataLoader(test_data, batch_size=200)

# print ("save to unix timestamp")
# # code to convert from date time string in to unix timestamp
# fname = "../../../tgb/datasets/tgbl_flight/tgbl-flight_edgelist_v2.csv"
# outname = "../../../tgb/datasets/tgbl_flight/tgbl-flight_edgelist_ts.csv"
# ctr = 0
# with open(outname, "w") as outf:
#     write = csv.writer(outf)
#     fields = ["timestamp", "src", "dst", "callsign", "typecode"]
#     write.writerow(fields)
#     with open(fname, "r") as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=",")
#         idx = 0
#         for row in tqdm(csv_reader):
#             if idx == 0:
#                 idx += 1
#                 continue
#             else:
#                 ts = row[0]
#                 TIME_FORMAT = "%Y-%m-%d"
#                 date_cur = datetime.datetime.strptime(ts, TIME_FORMAT)
#                 ts = int(date_cur.timestamp())
#                 row[0] = ts
#                 write.writerow(row)
#                 ctr += 1
# print ("there are {} edges".format(ctr))



neg_sampler = dataset.negative_sampler
dataset.load_val_ns()

for batch in tqdm(val_loader):
    neg_batch_list = neg_sampler.query_batch(batch.src, batch.dst, batch.t, split_mode="val")


print ("passed val query")

dataset.load_test_ns()

for batch in tqdm(test_loader):
    neg_batch_list = neg_sampler.query_batch(batch.src, batch.dst, batch.t, split_mode="test")

print ("passed test query")