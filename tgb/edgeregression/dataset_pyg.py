import os.path as osp
from typing import Callable, Optional

import torch

from torch_geometric.data import InMemoryDataset, TemporalData, download_url
from tgb.edgeregression.dataset import EdgeRegressionDataset
import warnings



class PyGEdgeRegressDataset(InMemoryDataset):
    r"""
    PyG wrapper for the EdgeRegressionDataset
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

        super().__init__(root, transform, pre_transform)
        self.dataset = EdgeRegressionDataset(name=name, root=root)
        self._data = None
        self.process_data()

    def process_data(self):
        """
        collate on the data from the EdgeRegressionDataset
        """

        #! check a few tensor typing constraints first, forcing the conversion if needed
        src = torch.from_numpy(self.dataset.full_data["sources"])
        dst = torch.from_numpy(self.dataset.full_data["destinations"])
        t = torch.from_numpy(self.dataset.full_data["timestamps"])
        
        if (src.dtype != torch.int64 and src.dtype != torch.int32):
            warnings.warn("sources tensor is not of type int64 or int32, forcing conversion")
            src = src.long()
        
        if (dst.dtype != torch.int64 and dst.dtype != torch.int32):
            warnings.warn("destinations tensor is not of type int64 or int32, forcing conversion")
            dst = dst.long()
        
        if (t.dtype != torch.int64 and t.dtype != torch.int32):
            warnings.warn("time tensor is not of type int64 or int32, forcing conversion")
            t = t.long()
        y = torch.from_numpy(self.dataset.full_data["y"]).long()
        msg = torch.from_numpy(self.dataset.full_data['edge_idxs']).reshape([-1,1]).float()  #use edge features here if available
        data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self._data = self.collate([data])

    
    @property
    def data(self):
        if (self._data is None):
            raise RuntimeError('Dataset not initialized. Call process_data()')
        else:
            return self._data
    

    

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'
