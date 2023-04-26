import torch
import os.path as osp
from typing import Optional, Dict, Any, Optional, Callable

from torch_geometric.data import InMemoryDataset, TemporalData, download_url
from tgb.linkproppred.dataset import LinkPropPredDataset
import warnings



class PyGLinkPropPredDataset(InMemoryDataset):
    r"""
    PyG wrapper for the LinkPropPredDataset
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
        self.dataset = LinkPropPredDataset(name=name, root=root)
        self._train_mask = torch.from_numpy(self.dataset.train_mask)
        self._val_mask = torch.from_numpy(self.dataset.val_mask)
        self._test_mask = torch.from_numpy(self.dataset.test_mask)
        super().__init__(root, transform, pre_transform)
        self.node_feat = self.dataset.node_feat
        self.process_data()


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
        collate on the data from the EdgeRegressionDataset
        """

        #! check a few tensor typing constraints first, forcing the conversion if needed
        src = torch.from_numpy(self.dataset.full_data["sources"])
        dst = torch.from_numpy(self.dataset.full_data["destinations"])
        t = torch.from_numpy(self.dataset.full_data["timestamps"])
        y = torch.from_numpy(self.dataset.full_data["y"])
        msg = torch.from_numpy(self.dataset.full_data['edge_feat'])  #use edge features here if available

        
        if (src.dtype != torch.int64 and src.dtype != torch.int32):
            warnings.warn("sources tensor is not of type int64 or int32, forcing conversion")
            src = src.long()
        
        if (dst.dtype != torch.int64 and dst.dtype != torch.int32):
            warnings.warn("destinations tensor is not of type int64 or int32, forcing conversion")
            dst = dst.long()
        
        if (t.dtype != torch.int64 and t.dtype != torch.int32):
            warnings.warn("time tensor is not of type int64 or int32, forcing conversion")
            t = t.long()

        #! this is required for some reason, investigate more in the future
        msg = msg.float()

        data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self._data = self.collate([data])

    def get_node_feat(self,srcs):
        return torch.from_numpy(self.node_feat[srcs.cpu()])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'
    

