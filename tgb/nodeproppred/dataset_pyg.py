import os.path as osp
from typing import Optional, Dict, Any, Optional, Callable


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
        self._train_mask = torch.from_numpy(self.dataset.train_mask)
        self._val_mask = torch.from_numpy(self.dataset.val_mask)
        self._test_mask = torch.from_numpy(self.dataset.test_mask)
        self.__num_classes__ = self.dataset.num_classes
        super().__init__(root, transform, pre_transform)
        self.process_data()


    @property
    def num_classes(self):
        return self.__num_classes__
    
    
    @property
    def train_mask(self) -> Dict[str, Any]:
        r"""
        Returns the train mask of the dataset 
        Returns:
            train_mask: Dict[str, Any]
        """
        if (self._train_mask is None):
            raise ValueError("training split hasn't been loaded")
        return self._train_mask
    
    @property
    def val_mask(self) -> Dict[str, Any]:
        r"""
        Returns the validation mask of the dataset 
        Returns:
            val_mask: Dict[str, Any]
        """
        if (self._val_mask is None):
            raise ValueError("validation split hasn't been loaded")
        
        return self._val_mask
    
    @property
    def test_mask(self) -> Dict[str, Any]:
        r"""
        Returns the test mask of the dataset:
        Returns:
            test_mask: Dict[str, Any]
        """
        if (self._test_mask is None):
            raise ValueError("test split hasn't been loaded")
        
        return self._test_mask
    
    
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

        
        if (src.dtype != torch.int64):
            #warnings.warn("sources tensor is not of type int64 or int32, forcing conversion")
            src = src.long()
        
        if (dst.dtype != torch.int64):
            #warnings.warn("destinations tensor is not of type int64 or int32, forcing conversion")
            dst = dst.long()
        
        if (t.dtype != torch.int64):
            #warnings.warn("time tensor is not of type int64 or int32, forcing conversion")
            t = t.long()

        if (msg.dtype != torch.float32):
            #warnings.warn("msg tensor is not of type float64 or float32, forcing conversion")
            msg = msg.to(torch.float32)
            
        data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self._data = self.collate([data])

    def reset_label_time(self):
        self.dataset.reset_label_time()

    def get_node_label(self, cur_t):
        label_tuple = self.dataset.find_next_labels_batch(cur_t)
        if (label_tuple is None):
            return None
        label_ts, label_srcs, labels = label_tuple[0], label_tuple[1], label_tuple[2]
        label_ts = torch.from_numpy(label_ts).long()
        label_srcs = torch.from_numpy(label_srcs).long()
        labels = torch.from_numpy(labels).to(torch.float32)
        return label_ts, label_srcs, labels

    def get_label_time(self):
        return self.dataset.return_label_ts()




    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'
