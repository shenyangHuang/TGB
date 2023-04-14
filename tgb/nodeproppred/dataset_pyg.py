import os.path as osp
from typing import Callable, Optional

import torch

from torch_geometric.data import InMemoryDataset, TemporalData, download_url
from tgb.nodeproppred.dataset import NodePropertyDataset
import warnings


#TODO check https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/in_memory_dataset.html
#avoid any overlapping properties
class PyGNodePropertyDataset(InMemoryDataset):
    r"""
    PyG wrapper for the NodePropertyDataset
    Parameters:
        name: name of the dataset
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
        pre_transform (callable, optional): A function/transform that takes in
    """
    def __init__(
        self,
        name: str,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.dataset = NodePropertyDataset(name=name, root=root)
        self.__num_classes__ = self.dataset.num_classes
        super().__init__(root, transform, pre_transform)
        self.process_data()


    @property
    def num_classes(self):
        return self.__num_classes__

    def process_data(self):
        """
        collate on the data from the NodePropertyDataset
        """

        #! check a few tensor typing constraints first, forcing the conversion if needed
        src = torch.from_numpy(self.dataset.full_data["sources"])
        dst = torch.from_numpy(self.dataset.full_data["destinations"])
        t = torch.from_numpy(self.dataset.full_data["timestamps"])
        y = torch.from_numpy(self.dataset.full_data["y"])
        msg = torch.from_numpy(self.dataset.full_data['edge_idxs']).reshape([-1,1])  #use edge features here if available

        
        if (src.dtype != torch.int64 and src.dtype != torch.int32):
            warnings.warn("sources tensor is not of type int64 or int32, forcing conversion")
            src = src.long()
        
        if (dst.dtype != torch.int64 and dst.dtype != torch.int32):
            warnings.warn("destinations tensor is not of type int64 or int32, forcing conversion")
            dst = dst.long()
        
        if (t.dtype != torch.int64 and t.dtype != torch.int32):
            warnings.warn("time tensor is not of type int64 or int32, forcing conversion")
            t = t.long()

        if (msg.dtype != torch.float32 and msg.dtype != torch.float64):
            warnings.warn("msg tensor is not of type float64 or float32, forcing conversion")
            msg = msg.float()
        data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self._data = self.collate([data])

    def reset_label_time(self):
        self.dataset.reset_ctr()

    def get_node_label(self, cur_t):
        label_tuple = self.dataset.find_next_labels_batch(cur_t)
        if (label_tuple is None):
            return None
        label_ts, label_srcs, labels = label_tuple[0], label_tuple[1], label_tuple[2]
        label_ts = torch.from_numpy(label_ts).long()
        label_srcs = torch.from_numpy(label_srcs).long()
        labels = torch.from_numpy(labels).float()
        return label_ts, label_srcs, labels

    def get_label_time(self):
        return self.dataset.get_nearest_label_ctr()
    


    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'
