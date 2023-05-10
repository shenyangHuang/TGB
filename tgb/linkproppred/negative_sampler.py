"""
Sample negative edges for evaluation of dynamic link prediction
"""

import torch
from torch import Tensor
import numpy as np
from torch_geometric.data import TemporalData
from tgb.utils.utils import save_pkl, load_pkl
from tgb.utils.info import PROJ_DIR
import os
import time


class NegativeEdgeSampler_RND(object):
    def __init__(
        self,
        dataset_name: str,
        first_dst_id: int,
        last_dst_id: int,
        device: str,
        num_neg_e: int = 200,  # number of negative edges sampled per positive edges --> make it constant => 1000
        rnd_seed: int = 123
    ):
        r"""
        Negative Edge Sampler class
        it is used for sampling negative edges for the task of link prediction.
        negative edges are sampled with 'oen_vs_many' strategy.
        it is assumed that the destination nodes are indexed sequentially with 'first_dst_id' and 'last_dst_id' being the first and last index, respectively.
        """
        self.device = device
        self.rnd_seed = rnd_seed
        np.random.seed(self.rnd_seed)
        self.dataset_name = dataset_name

        self.first_dst_id = first_dst_id
        self.last_dst_id = last_dst_id
        self.num_neg_e = num_neg_e

        self.val_eval_set = None 
        self.test_eval_set = None

    def generate_negative_samples(self, data, split_mode, partial_path):
        r"""
        For each positive edge, sample a batch of negative edges
        """
        assert split_mode in ['val', 'test'], 'Invalid split-mode! It should be `val` or `test`!'
        # file name for saving or loading...
        filename = partial_path + "/" + self.dataset_name + '_' + split_mode + '.pkl'

        if os.path.exists(filename):
            print(f"INFO: negative samples for '{split_mode}' evaluation are already generated!")
        else:
            print(f"INFO: Generating negative samples for '{split_mode}' evaluation!")
            # retrieve the information from the batch
            pos_src, pos_dst, pos_timestamp = data.src.cpu().numpy(), data.dst.cpu().numpy(), data.t.cpu().numpy()

            # all possible destinations
            all_dst = np.arange(self.first_dst_id, self.last_dst_id + 1)

            evaluation_set = {}
            # generate a list of negative destinations for each positive edge
            for pos_s, pos_d, pos_t, in zip(pos_src, pos_dst, pos_timestamp):
                t_mask = pos_timestamp == pos_t
                src_mask = pos_src == pos_s
                fn_mask = np.logical_and(t_mask, src_mask)
                pos_e_dst_same_src = pos_dst[fn_mask]
                filtered_all_dst = [dst for dst in all_dst if dst not in pos_e_dst_same_src]

                replace = True if self.num_neg_e > len(filtered_all_dst) else False
                neg_d_arr = np.random.choice(filtered_all_dst, self.num_neg_e, replace=replace)

                evaluation_set[(pos_s, pos_d, pos_t)] = neg_d_arr

            # save the generated evaluation set to disk            
            save_pkl(evaluation_set, filename)

    def load_eval_set(self, split_mode, partial_path):
        r"""
        Load the evaluation set from disk
        """
        assert split_mode in ['val', 'test'], 'Invalid split-mode! It should be `val`, `test`!'

        filename = partial_path + "/" + self.dataset_name + '_' + split_mode + '.pkl'
        if not os.path.exists(filename):
            raise ValueError(f"Please generate the negative samples for '{split_mode}' evaluation first!")

        if split_mode == 'val':
            self.val_eval_set = load_pkl(filename)
        elif split_mode == 'test':
            self.test_eval_set = load_pkl(filename)
        else:
            raise ValueError("Invalid split-mode!")
    
    def reset_eval_set(self, split_mode, partial_path):
        r"""
        Reset evaluation set
        """
        assert split_mode in ['val', 'test'], 'Invalid split-mode! It should be `val`, `test`!'

        filename = partial_path + "/" + self.dataset_name + '_' + split_mode + '.pkl'
        if split_mode == 'val':
            self.val_eval_set = None
        elif split_mode == 'test':
            self.test_eval_set = None
        else:
            raise ValueError("Invalid split-mode!")

    def query_batch(self, pos_batch, split_mode):
        r"""
        For each positive edge in the `pos_batch`, return a list of negative edges
        `split_mode` specifies whether the valiation or test evaluation set should be retrieved.
        """
        assert split_mode in ['val', 'test'], 'Invalid split-mode! It should be `val`, `test`!'
        if split_mode == 'val':
            eval_set = self.val_eval_set
        elif split_mode == 'test':
            eval_set = self.test_eval_set
        else:
            raise ValueError("Invalid split-mode!")
        
        if eval_set == None:
            raise ValueError("Evaluation set is not loaded! Please, load the evaluation set first!")
        
        # retrieve the negative sample lists for each positive edge in the `pos_batch`

        # get the information from the batch
        pos_src, pos_dst, pos_timestamp = pos_batch.src.cpu().numpy(), pos_batch.dst.cpu().numpy(), pos_batch.t.cpu().numpy()
        neg_samples = []
        for pos_s, pos_d, pos_t in zip(pos_src, pos_dst, pos_timestamp):
            if (pos_s, pos_d, pos_t) not in eval_set:
                raise ValueError(f"The edge ({pos_s}, {pos_d}, {pos_t}) is not in the '{split_mode}' evaluation set! Please check the implementation.")
            else:
                neg_samples.append(eval_set[(pos_s, pos_d, pos_t)])

        return neg_samples





    




