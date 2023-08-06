import os.path as osp
from typing import Optional, Dict, Any, Optional, Callable


import torch

from torch_geometric.data import InMemoryDataset, TemporalData, download_url
from tgb.nodeproppred.dataset import NodePropPredDataset
import warnings


# TODO check https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/in_memory_dataset.html
# avoid any overlapping properties
class PyGNodePropPredDataset(InMemoryDataset):
    r"""
    PyG wrapper for the NodePropPredDataset
    can return pytorch tensors for src,dst,t,msg,label
    can return Temporal Data object
    also query the node labels corresponding to a timestamp from edge batch
    Parameters:
        name: name of the dataset, passed to `NodePropPredDataset`
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
        self.dataset = NodePropPredDataset(name=name, root=root)
        self._train_mask = torch.from_numpy(self.dataset.train_mask)
        self._val_mask = torch.from_numpy(self.dataset.val_mask)
        self._test_mask = torch.from_numpy(self.dataset.test_mask)
        self.__num_classes = self.dataset.num_classes
        super().__init__(root, transform, pre_transform)
        self.process_data()

    @property
    def num_classes(self) -> int:
        """
        how many classes are in the node label
        Returns:
            num_classes: int
        """
        return self.__num_classes

    @property
    def eval_metric(self) -> str:
        """
        the official evaluation metric for the dataset, loaded from info.py
        Returns:
            eval_metric: str, the evaluation metric
        """
        return self.dataset.eval_metric

    @property
    def train_mask(self) -> torch.Tensor:
        r"""
        Returns the train mask of the dataset
        Returns:
            train_mask: the mask for edges in the training set
        """
        if self._train_mask is None:
            raise ValueError("training split hasn't been loaded")
        return self._train_mask

    @property
    def val_mask(self) -> torch.Tensor:
        r"""
        Returns the validation mask of the dataset
        Returns:
            val_mask: the mask for edges in the validation set
        """
        if self._val_mask is None:
            raise ValueError("validation split hasn't been loaded")
        return self._val_mask

    @property
    def test_mask(self) -> torch.Tensor:
        r"""
        Returns the test mask of the dataset:
        Returns:
            test_mask: the mask for edges in the test set
        """
        if self._test_mask is None:
            raise ValueError("test split hasn't been loaded")
        return self._test_mask

    @property
    def src(self) -> torch.Tensor:
        r"""
        Returns the source nodes of the dataset
        Returns:
            src: the idx of the source nodes
        """
        return self._src

    @property
    def dst(self) -> torch.Tensor:
        r"""
        Returns the destination nodes of the dataset
        Returns:
            dst: the idx of the destination nodes
        """
        return self._dst

    @property
    def ts(self) -> torch.Tensor:
        r"""
        Returns the timestamps of the dataset
        Returns:
            ts: the timestamps of the edges
        """
        return self._ts

    @property
    def edge_feat(self) -> torch.Tensor:
        r"""
        Returns the edge features of the dataset
        Returns:
            edge_feat: the edge features
        """
        return self._edge_feat

    @property
    def edge_label(self) -> torch.Tensor:
        r"""
        Returns the edge labels of the dataset
        Returns:
            edge_label: the labels of the edges (all one tensor)
        """
        return self._edge_label

    def process_data(self):
        """
        convert data to pytorch tensors
        """
        src = torch.from_numpy(self.dataset.full_data["sources"])
        dst = torch.from_numpy(self.dataset.full_data["destinations"])
        t = torch.from_numpy(self.dataset.full_data["timestamps"])
        edge_label = torch.from_numpy(self.dataset.full_data["edge_label"])
        msg = torch.from_numpy(self.dataset.full_data["edge_feat"])
        # msg = torch.from_numpy(self.dataset.full_data["edge_feat"]).reshape(
        #     [-1, 1]
        # ) 
        # * check typing
        if src.dtype != torch.int64:
            src = src.long()

        if dst.dtype != torch.int64:
            dst = dst.long()

        if t.dtype != torch.int64:
            t = t.long()

        if msg.dtype != torch.float32:
            msg = msg.float()

        self._src = src
        self._dst = dst
        self._ts = t
        self._edge_label = edge_label
        self._edge_feat = msg

    def get_TemporalData(
        self,
    ) -> TemporalData:
        """
        return the TemporalData object for the entire dataset
        Returns:
            data: TemporalData object storing the edgelist
        """
        data = TemporalData(
            src=self._src,
            dst=self._dst,
            t=self._ts,
            msg=self._edge_feat,
            y=self._edge_label,
        )
        return data

    def reset_label_time(self) -> None:
        """
        reset the pointer for the node labels, should be done per epoch
        """
        self.dataset.reset_label_time()

    def get_node_label(self, cur_t):
        """
        return the node labels for the current timestamp
        """
        label_tuple = self.dataset.find_next_labels_batch(cur_t)
        if label_tuple is None:
            return None
        label_ts, label_srcs, labels = label_tuple[0], label_tuple[1], label_tuple[2]
        label_ts = torch.from_numpy(label_ts).long()
        label_srcs = torch.from_numpy(label_srcs).long()
        labels = torch.from_numpy(labels).to(torch.float32)
        return label_ts, label_srcs, labels

    def get_label_time(self) -> int:
        """
        return the timestamps of the current node labels
        Returns:
            t: time of the current node labels
        """
        return self.dataset.return_label_ts()

    def len(self) -> int:
        """
        size of the dataset
        Returns:
            size: int
        """
        return self._src.shape[0]

    def get(self, idx: int) -> TemporalData:
        """
        construct temporal data object for a single edge
        Parameters:
            idx: index of the edge
        Returns:
            data: TemporalData object
        """
        data = TemporalData(
            src=self._src[idx],
            dst=self._dst[idx],
            t=self._ts[idx],
            msg=self._edge_feat[idx],
            y=self._edge_label[idx],
        )
        return data

    def __repr__(self) -> str:
        return f"{self.name.capitalize()}()"
