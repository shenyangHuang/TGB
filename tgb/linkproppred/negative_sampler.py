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
import os
import time


class NegativeEdgeSampler(object):
    def __init__(
        self,
        dataset_name: str,
        strategy: str = "hist_rnd",
    ):
        r"""
        Negative Edge Sampler
            Loads and query the negative batches based on the positive batches provided.
        """
        self.dataset_name = dataset_name
        assert strategy in [
            "rnd",
            "hist_rnd",
        ], "The supported strategies are `rnd` or `hist_rnd`!"
        self.strategy = strategy
        self.eval_set = {}

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
        """
        assert split_mode in [
            "val",
            "test",
        ], "Invalid split-mode! It should be `val`, `test`"
        if not os.path.exists(fname):
            raise FileNotFoundError(f"File not found at {fname}")
        self.eval_set[split_mode] = load_pkl(fname)

    def reset_eval_set(self, split_mode):
        r"""
        Reset evaluation set
        """
        assert split_mode in [
            "val",
            "test",
        ], "Invalid split-mode! It should be `val`, `test`!"
        self.eval_set[split_mode] = None

    def query_batch(self, pos_batch, split_mode):
        r"""
        For each positive edge in the `pos_batch`, return a list of negative edges
        `split_mode` specifies whether the valiation or test evaluation set should be retrieved.
        """
        assert split_mode in [
            "val",
            "test",
        ], "Invalid split-mode! It should be `val`, `test`!"
        if self.eval_set[split_mode] == None:
            raise ValueError(
                f"Evaluation set is None! You should load the {split_mode} evaluation set first!"
            )

        # retrieve the negative sample lists for each positive edge in the `pos_batch`
        # first, get the information from the batch
        pos_src, pos_dst, pos_timestamp = (
            pos_batch.src.cpu().numpy(),
            pos_batch.dst.cpu().numpy(),
            pos_batch.t.cpu().numpy(),
        )
        neg_samples = []
        for pos_s, pos_d, pos_t in zip(pos_src, pos_dst, pos_timestamp):
            if (pos_s, pos_d, pos_t) not in self.eval_set[split_mode]:
                raise ValueError(
                    f"The edge ({pos_s}, {pos_d}, {pos_t}) is not in the '{split_mode}' evaluation set! Please check the implementation."
                )
            else:
                neg_samples.append(
                    [
                        int(neg_dst)
                        for neg_dst in self.eval_set[split_mode][(pos_s, pos_d, pos_t)]
                    ]
                )

        return neg_samples
