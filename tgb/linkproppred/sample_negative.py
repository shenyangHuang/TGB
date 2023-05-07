"""
Sample negative edges for evaluation of dynamic link prediction
"""

import torch
from torch import Tensor
import numpy as np
from torch_geometric.data import TemporalData


class NegativeEdgeSampler_RND(object):
    def __init__(
        self,
        first_dst_id: int,
        last_dst_id: int,
        device: str,
        num_neg_e: int = 200,  # number of negative edges sampled per positive edges
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

        self.first_dst_id = first_dst_id
        self.last_dst_id = last_dst_id
        self.num_neg_e = num_neg_e

    def sample(self, pos_batch):
        r"""
        For each positive edge, sample a batch of negative edges
        """
        # retrieve the information from the batch
        pos_src, pos_dst = pos_batch.src.cpu().numpy(), pos_batch.dst.cpu().numpy()

        # all possible destinations
        all_dst = np.arange(self.first_dst_id, self.last_dst_id + 1)

        edges_per_src_node = {}
        # available positive edges
        for pos_s, pos_d in zip(pos_src, pos_dst):
            if pos_s not in edges_per_src_node:
                edges_per_src_node[pos_s] = [pos_d]
            else:
                if pos_d not in edges_per_src_node[pos_s]:
                    edges_per_src_node[pos_s].append(pos_d)
        
        sample_batch_list = []
        # generate new batches of edges; sample negative edges
        for pos_s, pos_d in zip(pos_src, pos_dst):
            filtered_all_dst = list(set(all_dst) - set(edges_per_src_node[pos_s]))
            replace = True if self.num_neg_e > len(filtered_all_dst) else False
            neg_d_arr = np.random.choice(filtered_all_dst, self.num_neg_e, replace=replace)

            new_batch_src = torch.full((1 + self.num_neg_e, ), pos_s, device=self.device)
            new_batch_dst = torch.tensor(np.concatenate(([np.array([pos_d]), neg_d_arr]), axis=0), device=self.device)
            new_batch_y = torch.tensor(np.concatenate((np.array([1]), np.array([0 for _ in range(self.num_neg_e)])), axis=0), device=self.device)

            sample_batch_list.append({'src': new_batch_src,
                                      'dst': new_batch_dst,
                                      'y': new_batch_y
                                      })
        
        return sample_batch_list

        

class NegativeEdgeSampler_HIST_RND(object):
    def __init__(
        self,
        first_dst_id: int,
        last_dst_id: int,
        train_data: TemporalData,
        val_data: TemporalData,
        device: str,
        num_neg_e: int = 200,  # number of negative edges sampled per positive edges
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

        self.first_dst_id = first_dst_id
        self.last_dst_id = last_dst_id
        self.num_neg_e = num_neg_e
        self.historical_edges = self.generate_historical_edge_set(train_data, val_data)
        self.hist_ratio = 0.5  # 50% historical negative and 50% random negative edges

    def generate_historical_edge_set(self, train_data, val_data):
        r"""
        Generate the set of edges seen durign training or validation
        """
        sources = torch.cat([train_data.src, val_data.src]).cpu().numpy()
        destinations = torch.cat([train_data.src, val_data.dst]).cpu().numpy()
        historical_edges = {}
        for src, dst in zip(sources, destinations):
            if (src, dst) not in historical_edges:
                historical_edges[(src, dst)] = 1
        
        return historical_edges

    def get_hist_edges_not_repeat(self, src, dst):
        r"""
        Get those historical edges that their source and destinations are not in 'src' & 'dst'
        """
        current_edges = {}
        for src, dst in zip(src, dst):
            if (src, dst) not in current_edges:
                current_edges[(src, dst)] = 1

        # print("DEBUG: self.historical_edges:", self.historical_edges)
        not_repeat_hist = {}
        for edge_id, _ in self.historical_edges.items():
            if (edge_id[0], edge_id[1]) not in current_edges:
                if (edge_id[0], edge_id[1]) not in not_repeat_hist:
                    not_repeat_hist[(edge_id[0], edge_id[1])] = 1
        
        hist_src, hist_dst = [],[]
        for edge_id, _ in not_repeat_hist.items():
            hist_src.append(edge_id[0])
            hist_dst.append(edge_id[1])

        return np.array(hist_src), np.array(hist_dst)

    def sample_rnd(self, pos_src, pos_dst, sample_size):
        r"""
        For each positive edge with source and destination nodes in pos_src and pos_dst respectively, sample a batch of negative edges
        """
        # all possible destinations
        all_dst = np.arange(self.first_dst_id, self.last_dst_id + 1)

        edges_per_src_node = {}
        # available positive edges
        for pos_s, pos_d in zip(pos_src, pos_dst):
            if pos_s not in edges_per_src_node:
                edges_per_src_node[pos_s] = [pos_d]
            else:
                if pos_d not in edges_per_src_node[pos_s]:
                    edges_per_src_node[pos_s].append(pos_d)
        
        rnd_neg_edges_per_pos_edge = {}
        # generate new batches of edges; sample negative edges
        for pos_s, pos_d in zip(pos_src, pos_dst):
            filtered_all_dst = list(set(all_dst) - set(edges_per_src_node[pos_s]))
            replace = True if sample_size > len(filtered_all_dst) else False
            neg_d_arr = np.random.choice(filtered_all_dst, sample_size, replace=replace)

            rnd_neg_edges_per_pos_edge[(pos_s, pos_d)] = neg_d_arr

        return rnd_neg_edges_per_pos_edge

    def sample(self, pos_batch):
        r"""
        For each positive edge, sample a batch of negative edges including historically and randomly sampled negative edges
        """
        # retrieve the information from the batch
        pos_src, pos_dst = pos_batch.src.cpu().numpy(), pos_batch.dst.cpu().numpy()

        # get possible historical edges 
        all_hist_src, all_hist_dst = self.get_hist_edges_not_repeat(pos_src, pos_dst)

        # get random negative edges
        rnd_neg_edges = self.sample_rnd(pos_src, pos_dst, int((1 - self.hist_ratio) * self.num_neg_e))

        sample_batch_list = []
        # generate new batches of edges; sample negative edges
        for pos_s, pos_d in zip(pos_src, pos_dst):
            pos_s_tensor = torch.tensor(np.array([pos_s]), device=self.device)
            pos_d_tensor = torch.tensor(np.array([pos_d]), device=self.device)

            # historical negative edgs for this positive edge
            hist_idx = np.arange(0, len(all_hist_src))
            hist_idx_selected = np.random.choice(hist_idx, int(self.hist_ratio * self.num_neg_e), replace=True)  # replacement is true, since there might be less number of historical edges than needed.
            hist_src = torch.tensor(np.array(all_hist_src[hist_idx_selected]), device=self.device)
            hist_dst = torch.tensor(np.array(all_hist_dst[hist_idx_selected]), device=self.device)

            # random negative edges for this positive edge 
            rnd_dst = torch.tensor(np.array(rnd_neg_edges[(pos_s, pos_d)]), device=self.device)
            rnd_src = torch.full((len(rnd_dst), ), pos_s, device=self.device)

            # generate a new batch
            # print(f"DEBUG: n_pos:{pos_s_tensor.size()}, hist_src: {hist_src.size()}, rnd_src: {rnd_src.size()}")
            new_batch_src = torch.cat([pos_s_tensor, hist_src, rnd_src]).to(self.device)
            new_batch_dst = torch.cat([pos_d_tensor, hist_dst, rnd_dst]).to(self.device)
            new_batch_y = torch.tensor(np.concatenate((np.array([1]), np.array([0 for _ in range(hist_src.size(0) + rnd_src.size(0))])), axis=0), 
                                        device=self.device)

            sample_batch_list.append({'src': new_batch_src,
                                      'dst': new_batch_dst,
                                      'y': new_batch_y
                                      })
        
        return sample_batch_list





