"""
Sample and Generate negative edges that are going to be used for evaluation of a dynamic graph learning model
Negative samples are generated and saved to files ONLY once; 
    other times, they should be loaded from file with instances of the `negative_sampler.py`.
"""

import torch
from torch import Tensor
import numpy as np
from torch_geometric.data import TemporalData
from tgb.utils.utils import save_pkl, load_pkl
from tgb.utils.info import PROJ_DIR
import os
import time


class NegativeEdgeGenerator(object):
    def __init__(
        self,
        dataset_name: str,
        first_dst_id: int,
        last_dst_id: int,
        device: str,
        num_neg_e: int = 100,  # number of negative edges sampled per positive edges --> make it constant => 1000
        strategy: str = 'rnd',
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
        assert strategy in ['rnd', 'hist_rnd'], "The supported strategies are `rnd` or `hist_rnd`!"
        self.strategy = strategy

    def generate_negative_samples(self, data, split_mode, partial_path):
        if self.strategy == 'rnd':
            self.generate_negative_samples_rnd(data, split_mode, partial_path)
        elif self.strategy == 'hist_rnd':
            self.generate_negative_samples_hist_rnd(data, split_mode, partial_path)
        else:
            raise ValueError("Unsupported negative sample generation strategy!")

    def generate_negative_samples_rnd(self, data, split_mode, partial_path):
        r"""
        Generate negative samples based on the `HIST-RND` strategy:
            - for each positive edge, sample a batch of negative edges from all possible edges with the same source node
            - filter actual positive edges
        """
        print(f"INFO: Negative Sampling Strategy: {self.strategy}")
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

    def generate_negative_samples_hist_rnd(self, data, split_mode, partial_path):
        r"""
        Generate negative samples based on the `HIST-RND` strategy:
            - up to 50% of the negative samples are selected from the set of edges seen during the training with the same source node.
            - the rest of the negative edges are randomly sampled with the fixed source node.
        """
        print(f"INFO: Negative Sampling Strategy: {self.strategy}")



def main():
    r"""
    Generate negative edges for the validation or test phase
    """
    print("*** Negative Sample Generation ***")


if __name__ == '__main__':
    main()
    




