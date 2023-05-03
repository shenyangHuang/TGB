"""
Generative negative edges for the task of link prediction

"""

import torch
from torch import Tensor
import numpy as np



class NegativeEdgeSampler(object):
    def __init__(
        self,
        first_dst_id: int,
        last_dst_id: int,
        device: str,
        nes_mode: str = 'one_vs_one',  # ['one_vs_one', 'one_vs_all', 'any_vs_all']
        rnd_seed: int = 123
    ):
        r"""
        Negative Edge Sampler class
        it is used for generating negative edges for the task of link prediction.
        negative samples can be generated in one of the following three modes:
            - 'one_vs_one': each positive edge is compared against one randomly sampled negative edge
            - 'one_vs_all': each positive edge is compared against all possible edges with the same source node
            - 'any_vs_all': any positive edges with the same source node are compared against all possible negative edges with the same source node
        it is assumed that the destination nodes are indexed sequentially with 'first_dst_id' and 'last_dst_id' being the first and last index, respectively.
        """
        assert nes_mode in ['one_vs_one', 'one_vs_all', 'any_vs_all'], "nes_mode should be in {'one_vs_one', 'one_vs_all', 'any_vs_all'}"
        self.nes_mode = nes_mode
        self.first_dst_id = first_dst_id
        self.last_dst_id = last_dst_id
        self.device = device
        self.rnd_seed = rnd_seed
        torch.manual_seed(self.rnd_seed)

    def generate_one_vs_one_negatives(self, pos_src: Tensor, pos_dst: Tensor):
        r"""
        Select an equal number of random negative edges 
        Args:
            - pos_src: positive source nodes (of a batch)
            - pos_dst: positive destination nodes (of a batch)
        @TODO: add collision check for positive vs. negative edges
        """
        sample_size = pos_src.size(0)
        neg_dst = torch.randint(self.first_dst_id, self.last_dst_id + 1, (sample_size, ), dtype=torch.long, device=self.device)

        return neg_dst

    def generate_one_vs_all_negatives(self, pos_src: Tensor, pos_dst: Tensor):
        r"""
        For each positive edge, all possible negative edges are sampled
        Args:
            - pos_src: positive source nodes (of a batch)
            - pos_dst: positive destination nodes (of a batch)

        Returns:
            - for each positive EDGE:
                - the list of positive edges (which only contain itself)
                - the list of possible negative edges to compare against
        """
        sample_size = pos_src.size(0)
        all_dst = torch.arange(self.first_dst_id, self.last_dst_id + 1)

        edges_per_pos_edge = {}
        
        # positive edges
        for pos_s, pos_d in zip(pos_src, pos_dst):
            pos_s = pos_s.item()
            pos_d = pos_d.item()
            if (pos_s, pos_d) not in edges_per_pos_edge:
                edges_per_pos_edge[(pos_s, pos_d)] = {'pos': [pos_d]}

        # negative edges
        for (pos_s, pos_d) in edges_per_pos_edge:
            edges_per_pos_edge[(pos_s, pos_d)]['neg'] = torch.tensor([neg_d for neg_d in all_dst if neg_d != pos_d], device=self.device)


        return edges_per_pos_edge

    def generate_any_vs_all_negatives(self, pos_src: Tensor, pos_dst: Tensor):
        r"""
        Positive edges in a batch with the same source node are compared agains all possible negative edges with the same source node
        Args:
            - pos_src: positive source nodes (of a batch)
            - pos_dst: positive destination nodes (of a batch)

        Returns:
            - for each positive NODE:
                - the list of positive edges in the batch 
                - the list of possible negative edges to compare against
        """
        sample_size = pos_src.size(0)
        all_dst = torch.arange(self.first_dst_id, self.last_dst_id + 1)

        edges_per_node = {}

        # positive edges
        for pos_s, pos_d in zip(pos_src, pos_dst):
            pos_s = pos_s.item()
            pos_d = pos_d.item()
            if pos_s not in edges_per_node:
                edges_per_node[pos_s] = {'pos': [pos_d]}
            else:
                if pos_d not in edges_per_node[pos_s]['pos']:
                    edges_per_node[pos_s]['pos'].append(pos_d)
        
        # negative edges
        for pos_s in edges_per_node:
            edges_per_node[pos_s]['pos'] = torch.tensor(edges_per_node[pos_s]['pos'], device=self.device)
            edges_per_node[pos_s]['neg'] = torch.tensor([neg_d for neg_d in all_dst if neg_d not in edges_per_node[pos_s]['pos']], device=self.device)

        return edges_per_node



