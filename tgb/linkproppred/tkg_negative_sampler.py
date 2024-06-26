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


class TKGNegativeEdgeSampler(object):
    def __init__(
        self,
        dataset_name: str,
        first_dst_id: int,
        last_dst_id: int,
        strategy: str = "time-filtered",
        partial_path: str = PROJ_DIR + "/data/processed",
    ) -> None:
        r"""
        Negative Edge Sampler
            Loads and query the negative batches based on the positive batches provided.
        constructor for the negative edge sampler class

        Parameters:
            dataset_name: name of the dataset
            first_dst_id: identity of the first destination node
            last_dst_id: indentity of the last destination node
            strategy: will always load the pre-generated negatives
            partial_path: the path to the directory where the negative edges are stored
        
        Returns:
            None
        """
        self.dataset_name = dataset_name
        self.eval_set = {}
        self.first_dst_id = first_dst_id
        self.last_dst_id = last_dst_id
        self.strategy = strategy
        self.dst_dict = None
      
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
        
        if self.strategy == "time-filtered":
            neg_samples = []
            for pos_s, pos_d, pos_t, e_type in zip(pos_src, pos_dst, pos_timestamp, edge_type):
                if (pos_t, pos_s, e_type) not in self.eval_set[split_mode]:
                    raise ValueError(
                        f"The edge ({pos_s}, {pos_d}, {pos_t}, {e_type}) is not in the '{split_mode}' evaluation set! Please check the implementation."
                    )
                else:
                    conflict_dict = self.eval_set[split_mode]
                    conflict_set = conflict_dict[(pos_t, pos_s, e_type)]
                    all_dst = np.arange(self.first_dst_id, self.last_dst_id + 1)
                    filtered_all_dst = np.delete(all_dst, conflict_set, axis=0)

                    #! always using all possible destinations for evaluation
                    neg_d_arr = filtered_all_dst

                    #! this is very slow
                    neg_samples.append(
                            neg_d_arr
                        )
        elif self.strategy == "dst-time-filtered":
            neg_samples = []
            for pos_s, pos_d, pos_t, e_type in zip(pos_src, pos_dst, pos_timestamp, edge_type):
                if (pos_t, pos_s, e_type) not in self.eval_set[split_mode]:
                    raise ValueError(
                        f"The edge ({pos_s}, {pos_d}, {pos_t}, {e_type}) is not in the '{split_mode}' evaluation set! Please check the implementation."
                    )
                else:
                    filtered_dst = self.eval_set[split_mode]
                    neg_d_arr = filtered_dst[(pos_t, pos_s, e_type)]
                    neg_samples.append(
                            neg_d_arr
                        )
        #? can't convert to numpy array due to different lengths of negative samples
        return neg_samples


        