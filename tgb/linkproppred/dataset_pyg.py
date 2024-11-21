import torch
from typing import Optional, Optional, Callable

from torch_geometric.data import Dataset, TemporalData
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler


class PyGLinkPropPredDataset(Dataset):
    def __init__(
        self,
        name: str,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        r"""
        PyG wrapper for the LinkPropPredDataset
        can return pytorch tensors for src,dst,t,msg,label
        can return Temporal Data object
        Parameters:
            name: name of the dataset, passed to `LinkPropPredDataset`
            root (string): Root directory where the dataset should be saved, passed to `LinkPropPredDataset`
            transform (callable, optional): A function/transform that takes in an, not used in this case
            pre_transform (callable, optional): A function/transform that takes in, not used in this case
        """
        self.name = name
        self.root = root
        self.dataset = LinkPropPredDataset(name=name, root=root)
        self._train_mask = torch.from_numpy(self.dataset.train_mask)
        self._val_mask = torch.from_numpy(self.dataset.val_mask)
        self._test_mask = torch.from_numpy(self.dataset.test_mask)
        super().__init__(root, transform, pre_transform)
        self._node_feat = self.dataset.node_feat
        self._edge_type = None
        self._static_data = None

        if self._node_feat is None:
            self._node_feat = None
        else:
            self._node_feat = torch.from_numpy(self._node_feat).float()
        
        self._node_type = self.dataset.node_type
        if self.node_type is not None:
            self._node_type = torch.from_numpy(self.dataset.node_type).long()
        
        self.process_data()

        self._ns_sampler = self.dataset.negative_sampler

    @property
    def eval_metric(self) -> str:
        """
        the official evaluation metric for the dataset, loaded from info.py
        Returns:
            eval_metric: str, the evaluation metric
        """
        return self.dataset.eval_metric

    @property
    def negative_sampler(self) -> NegativeEdgeSampler:
        r"""
        Returns the negative sampler of the dataset, will load negative samples from disc
        Returns:
            negative_sampler: NegativeEdgeSampler
        """
        return self._ns_sampler
    
    @property
    def num_nodes(self) -> int:
        r"""
        Returns the total number of unique nodes in the dataset 
        Returns:
            num_nodes: int, the number of unique nodes
        """
        return self.dataset.num_nodes
    
    @property
    def num_rels(self) -> int:
        r"""
        Returns the total number of unique relations in the dataset 
        Returns:
            num_rels: int, the number of unique relations
        """
        return self.dataset.num_rels
    
    @property
    def num_edges(self) -> int:
        r"""
        Returns the total number of edges in the dataset 
        Returns:
            num_edges: int, the number of edges
        """
        return self.dataset.num_edges

    def load_val_ns(self) -> None:
        r"""
        load the negative samples for the validation set
        """
        self.dataset.load_val_ns()

    def load_test_ns(self) -> None:
        r"""
        load the negative samples for the test set
        """
        self.dataset.load_test_ns()

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
    def node_feat(self) -> torch.Tensor:
        r"""
        Returns the node features of the dataset
        Returns:
            node_feat: the node features
        """
        return self._node_feat
    
    @property
    def node_type(self) -> torch.Tensor:
        r"""
        Returns the node types of the dataset
        Returns:
            node_type: the node types [N]
        """
        return self._node_type

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
    def static_data(self) -> torch.Tensor:
        r"""
        Returns the static data of the dataset for tkgl-wikidata and tkgl-smallpedia
        Returns:
            static_data: the static data of the dataset
        """
        if (self._static_data is None):
            static_dict = {}
            static_dict["head"] = torch.from_numpy(self.dataset.static_data["head"]).long()
            static_dict["tail"] = torch.from_numpy(self.dataset.static_data["tail"]).long()
            static_dict["edge_type"] = torch.from_numpy(self.dataset.static_data["edge_type"]).long()
            self._static_data = static_dict
            return self._static_data
        else:
            return self._static_data 
    
    @property
    def edge_type(self) -> torch.Tensor:
        r"""
        Returns the edge types for each edge
        Returns:
            edge_type: edge type tensor (int)
        """
        return self._edge_type

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
            edge_label: the labels of the edges
        """
        return self._edge_label

    def process_data(self) -> None:
        r"""
        convert the numpy arrays from dataset to pytorch tensors
        """
        src = torch.from_numpy(self.dataset.full_data["sources"])
        dst = torch.from_numpy(self.dataset.full_data["destinations"])
        ts = torch.from_numpy(self.dataset.full_data["timestamps"])
        msg = torch.from_numpy(
            self.dataset.full_data["edge_feat"]
        )  # use edge features here if available
        edge_label = torch.from_numpy(
            self.dataset.full_data["edge_label"]
        )  # this is the label indicating if an edge is a true edge, always 1 for true edges
        w = torch.from_numpy(
            self.dataset.full_data["w"]
        )


        # * first check typing for all tensors
        # source tensor must be of type int64
        # warnings.warn("sources tensor is not of type int64 or int32, forcing conversion")
        if src.dtype != torch.int64:
            src = src.long()

        # destination tensor must be of type int64
        if dst.dtype != torch.int64:
            dst = dst.long()

        # timestamp tensor must be of type int64
        if ts.dtype != torch.int64:
            ts = ts.long()

        # message tensor must be of type float32
        if msg.dtype != torch.float32:
            msg = msg.float()

        # weight tensor must be of type float32
        if w.dtype != torch.float32:
            w = w.float()

        #* for tkg
        if ("edge_type" in self.dataset.full_data):
            edge_type = torch.from_numpy(self.dataset.full_data["edge_type"])
            if edge_type.dtype != torch.int64:
                edge_type = edge_type.long()
            self._edge_type = edge_type

        self._src = src
        self._dst = dst
        self._ts = ts
        self._edge_label = edge_label
        self._edge_feat = msg
        self._w = w

    def get_TemporalData(self) -> TemporalData:
        """
        return the TemporalData object for the entire dataset
        """
        if (self._edge_type is not None):
            data = TemporalData(
                src=self._src,
                dst=self._dst,
                t=self._ts,
                msg=self._edge_feat,
                y=self._edge_label,
                edge_type=self._edge_type,
                w=self._w,
            )
        else:
            data = TemporalData(
                src=self._src,
                dst=self._dst,
                t=self._ts,
                msg=self._edge_feat,
                y=self._edge_label,
                w=self._w,
            )
        return data

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
        if (self._edge_type is not None):
            data = TemporalData(
                src=self._src[idx],
                dst=self._dst[idx],
                t=self._ts[idx],
                msg=self._edge_feat[idx],
                y=self._edge_label[idx],
                edge_type=self._edge_type[idx]
            )
        else:
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
