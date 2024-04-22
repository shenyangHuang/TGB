"""
Sample negative edges for evaluation of dynamic link prediction
Load already generated negative edges from file, batch them based on the positive edge, and return the evaluation set
"""

import torch
from torch import Tensor
import numpy as np
from torch_geometric.data import TemporalData
from tgb.utils.utils import save_pkl, load_pkl
from tgb.utils.info import PROJ_DIR
from typing import Union
import os
import time


class THGNegativeEdgeSampler(object):
    def __init__(
        self,
        dataset_name: str,
        first_node_id: int,
        last_node_id: int,
        node_type: np.ndarray,
        strategy: str = "node-type-filtered",
    ) -> None:
        r"""
        Negative Edge Sampler
            Loads and query the negative batches based on the positive batches provided.
        constructor for the negative edge sampler class

        Parameters:
            dataset_name: name of the dataset
            first_node_id: identity of the first node
            last_node_id: indentity of the last destination node
            node_type: the node type of each node
            strategy: will always load the pre-generated negatives
        
        Returns:
            None
        """
        self.dataset_name = dataset_name
        self.eval_set = {}
        self.first_node_id = first_node_id
        self.last_node_id = last_node_id
        self.node_type = node_type
        assert isinstance(self.node_type, np.ndarray), "node_type should be a numpy array"
        self.node_type_dict = self.get_destinations_based_on_node_type(first_node_id, last_node_id, self.node_type)

    def get_destinations_based_on_node_type(self, 
                                            first_node_id: int,
                                            last_node_id: int,
                                            node_type: np.ndarray) -> dict:
        r"""
        get the destination node id arrays based on the node type

        Parameters:
            first_node_id: the first node id
            last_node_id: the last node id
            node_type: the node type of each node

        Returns:
            node_type_dict: a dictionary containing the destination node ids for each node type
        """
        node_type_store = {}
        assert first_node_id <= last_node_id, "Invalid destination node ids!"
        assert len(node_type) == (last_node_id - first_node_id + 1), "node type array must match the indices"
        for k in range(len(node_type)):
            nt = int(node_type[k]) #node type must be ints
            nid = k + first_node_id
            if nt not in node_type_store:
                node_type_store[nt] = {nid:1}
            else:
                node_type_store[nt][nid] = 1
        
        node_type_dict = {}
        for ntype in node_type_store:
            node_type_dict[ntype] = np.array(list(node_type_store[ntype].keys()))
            assert np.all(np.diff(node_type_dict[ntype]) >= 0), "Destination node ids for a given type must be sorted"
            assert np.all(node_type_dict[ntype] <= last_node_id), "Destination node ids must be less than or equal to the last destination id"
        return node_type_dict

    def load_eval_set(
        self,
        fname: str,
        split_mode: str = "val",
    ) -> None:
        r"""
        Load the evaluation set from disk, can be either val or test set ns samples
        Parameters:
            fname: the file name of the evaluation ns on disk
            split_mode: the split mode of the evaluation set, can be either `val` or `test`
        
        Returns:
            None
        """
        assert split_mode in [
            "val",
            "test",
        ], "Invalid split-mode! It should be `val`, `test`"
        if not os.path.exists(fname):
            raise FileNotFoundError(f"File not found at {fname}")
        self.eval_set[split_mode] = load_pkl(fname)

    def query_batch(self, 
                    pos_src: Union[Tensor, np.ndarray], 
                    pos_dst: Union[Tensor, np.ndarray], 
                    pos_timestamp: Union[Tensor, np.ndarray], 
                    edge_type: Union[Tensor, np.ndarray],
                    split_mode: str = "test") -> list:
        r"""
        For each positive edge in the `pos_batch`, return a list of negative edges
        `split_mode` specifies whether the valiation or test evaluation set should be retrieved.
        modify now to include edge type argument

        Parameters:
            pos_src: list of positive source nodes
            pos_dst: list of positive destination nodes
            pos_timestamp: list of timestamps of the positive edges
            split_mode: specifies whether to generate negative edges for 'validation' or 'test' splits

        Returns:
            neg_samples: list of numpy array; each array contains the set of negative edges that
                        should be evaluated against each positive edge.
        """
        assert split_mode in [
            "val",
            "test",
        ], "Invalid split-mode! It should be `val`, `test`!"
        if self.eval_set[split_mode] == None:
            raise ValueError(
                f"Evaluation set is None! You should load the {split_mode} evaluation set first!"
            )
        
        # check the argument types...
        if torch is not None and isinstance(pos_src, torch.Tensor):
            pos_src = pos_src.detach().cpu().numpy()
        if torch is not None and isinstance(pos_dst, torch.Tensor):
            pos_dst = pos_dst.detach().cpu().numpy()
        if torch is not None and isinstance(pos_timestamp, torch.Tensor):
            pos_timestamp = pos_timestamp.detach().cpu().numpy()
        if torch is not None and isinstance(edge_type, torch.Tensor):
            edge_type = edge_type.detach().cpu().numpy()
        
        if not isinstance(pos_src, np.ndarray) or not isinstance(pos_dst, np.ndarray) or not(pos_timestamp, np.ndarray) or not(edge_type, np.ndarray):
            raise RuntimeError(
                "pos_src, pos_dst, and pos_timestamp need to be either numpy ndarray or torch tensor!"
                )

        neg_samples = []
        for pos_s, pos_d, pos_t, e_type in zip(pos_src, pos_dst, pos_timestamp, edge_type):
            if (pos_t, pos_s, e_type) not in self.eval_set[split_mode]:
                raise ValueError(
                    f"The edge ({pos_s}, {pos_d}, {pos_t}, {e_type}) is not in the '{split_mode}' evaluation set! Please check the implementation."
                )
            else:
                conflict_dict = self.eval_set[split_mode]
                conflict_set, d_node_type = conflict_dict[(pos_t, pos_s, e_type)]
                all_dst = self.node_type_dict[d_node_type]
                filtered_all_dst = np.delete(all_dst, conflict_set, axis=0)
                neg_d_arr = filtered_all_dst
                neg_samples.append(
                        neg_d_arr
                    )
        
        #? can't convert to numpy array due to different lengths of negative samples
        return neg_samples
