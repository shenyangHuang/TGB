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
import os.path as osp
import os
import time
from tqdm import tqdm

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset


class NegativeEdgeGenerator(object):
    def __init__(
        self,
        dataset_name: str,
        first_dst_id: int,
        last_dst_id: int,
        num_neg_e: int = 100,  # number of negative edges sampled per positive edges --> make it constant => 1000
        strategy: str = 'rnd',
        rnd_seed: int = 123,
        hist_ratio: float = 0.5,
        historical_data = None
    ):
        r"""
        Negative Edge Sampler class
        it is used for sampling negative edges for the task of link prediction.
        negative edges are sampled with 'oen_vs_many' strategy.
        it is assumed that the destination nodes are indexed sequentially with 'first_dst_id' and 'last_dst_id' being the first and last index, respectively.
        """
        self.rnd_seed = rnd_seed
        np.random.seed(self.rnd_seed)
        self.dataset_name = dataset_name

        self.first_dst_id = first_dst_id
        self.last_dst_id = last_dst_id
        self.num_neg_e = num_neg_e
        assert strategy in ['rnd', 'hist_rnd'], "The supported strategies are `rnd` or `hist_rnd`!"
        self.strategy = strategy
        if self.strategy == 'hist_rnd':
            assert historical_data != None, 'Train data should be passed when `hist_rnd` strategy is selected.'
            self.hist_ratio = hist_ratio
            self.historical_data = historical_data

    def generate_negative_samples(self, data, split_mode, partial_path):
        r"""
        Generate negative samples
        """
        # file name for saving or loading...
        filename = partial_path + '/processed/negative_samples/' + self.dataset_name + '_' + self.strategy + '_' + split_mode + '.pkl'

        if self.strategy == 'rnd':
            self.generate_negative_samples_rnd(data, split_mode, filename)
        elif self.strategy == 'hist_rnd':
            self.generate_negative_samples_hist_rnd(self.historical_data, data, split_mode, filename)
        else:
            raise ValueError("Unsupported negative sample generation strategy!")

    def generate_negative_samples_rnd(self, data, split_mode, filename):
        r"""
        Generate negative samples based on the `HIST-RND` strategy:
            - for each positive edge, sample a batch of negative edges from all possible edges with the same source node
            - filter actual positive edges
        """
        print(f"INFO: Negative Sampling Strategy: {self.strategy}, Data Split: {split_mode}")
        assert split_mode in ['val', 'test'], 'Invalid split-mode! It should be `val` or `test`!'
        
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
            pos_edge_tqdm = tqdm(zip(pos_src, pos_dst, pos_timestamp), total=len(pos_src))
            for pos_s, pos_d, pos_t, in pos_edge_tqdm:
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

    def generate_historical_edge_set(self, historical_data):
        r"""
        Generate the set of edges seen durign training or validation
        NOTE: ONLY `train_data` should be passed as historical data; i.e., the HISTORICAL negative edges should be selected from training data only.
        """
        sources = historical_data.src.cpu().numpy()
        destinations = historical_data.dst.cpu().numpy()
        historical_edges = {}
        hist_e_per_node = {}
        for src, dst in zip(sources, destinations):
            # edge-centric
            if (src, dst) not in historical_edges:
                historical_edges[(src, dst)] = 1

            # node-centric
            if src not in hist_e_per_node:
                hist_e_per_node[src] = [dst]
            else:
                hist_e_per_node[src].append(dst)
        
        hist_edge_set_per_node = {}
        for src, dst_list in hist_e_per_node.items():
            hist_edge_set_per_node[src] = np.array(list(set(dst_list)))
        
        return historical_edges, hist_edge_set_per_node

    def generate_negative_samples_hist_rnd(self, historical_data, data, split_mode, filename):
        r"""
        Generate negative samples based on the `HIST-RND` strategy:
            - up to 50% of the negative samples are selected from the set of edges seen during the training with the same source node.
            - the rest of the negative edges are randomly sampled with the fixed source node.
        """
        print(f"INFO: Negative Sampling Strategy: {self.strategy}, Data Split: {split_mode}")
        assert split_mode in ['val', 'test'], 'Invalid split-mode! It should be `val` or `test`!'
        
        if os.path.exists(filename):
            print(f"INFO: negative samples for '{split_mode}' evaluation are already generated!")
        else:
            print(f"INFO: Generating negative samples for '{split_mode}' evaluation!")
            # retrieve the information from the batch
            pos_src, pos_dst, pos_timestamp = data.src.cpu().numpy(), data.dst.cpu().numpy(), data.t.cpu().numpy()

            # all possible destinations
            all_dst = np.arange(self.first_dst_id, self.last_dst_id + 1)

            # get seen edge history
            historical_edges, hist_edge_set_per_node = self.generate_historical_edge_set(historical_data)

            # sample historical edges
            max_num_hist_neg_e = int(self.num_neg_e * self.hist_ratio)

            evaluation_set = {}
            # generate a list of negative destinations for each positive edge
            pos_edge_tqdm = tqdm(zip(pos_src, pos_dst, pos_timestamp), total=len(pos_src))
            for pos_s, pos_d, pos_t, in pos_edge_tqdm:
                t_mask = pos_timestamp == pos_t
                src_mask = pos_src == pos_s
                fn_mask = np.logical_and(t_mask, src_mask)
                pos_e_dst_same_src = pos_dst[fn_mask]

                # sample historical edges
                num_hist_neg_e = 0
                neg_hist_dsts = np.array([])
                if pos_s in hist_edge_set_per_node:
                    seen_dst = hist_edge_set_per_node[pos_s]
                    if len(seen_dst) >= 1:
                        filtered_all_seen_dst = [dst for dst in seen_dst if dst not in pos_e_dst_same_src]
                        num_hist_neg_e = max_num_hist_neg_e if max_num_hist_neg_e <= len(filtered_all_seen_dst) else len(filtered_all_seen_dst)
                        neg_hist_dsts = np.random.choice(filtered_all_seen_dst, num_hist_neg_e, replace=False)

                # sample random edges
                invalid_dst = np.concatenate((np.array(pos_e_dst_same_src), seen_dst))
                filtered_all_rnd_dst = [dst for dst in all_dst if dst not in invalid_dst]

                num_rnd_neg_e = self.num_neg_e - num_hist_neg_e
                replace = True if num_rnd_neg_e > len(filtered_all_rnd_dst) else False
                neg_rnd_dsts = np.random.choice(filtered_all_rnd_dst, num_rnd_neg_e, replace=replace)

                # concatenate the two sets: historical and random
                neg_dst_arr = np.concatenate((neg_hist_dsts, neg_rnd_dsts))

                evaluation_set[(pos_s, pos_d, pos_t)] = neg_dst_arr

            # save the generated evaluation set to disk            
            save_pkl(evaluation_set, filename)



def main():
    r"""
    Generate negative edges for the validation or test phase
    """
    print("*** Negative Sample Generation ***")
    
    # setting the required parameters
    dataset_name = 'wikipedia'
    num_neg_e_per_pos = 100
    neg_sample_strategy = 'hist_rnd'
    rnd_seed = 42 
    val_ratio = 0.15
    test_ratio = 0.15

    # === wikipedia ===
    # load the original data
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'examples', 'data', 'JODIE')
    dataset = JODIEDataset(path, name=dataset_name)
    data = dataset[0]

    # split the data
    data_splits = {}
    data_splits['train'], data_splits['val'], data_splits['test'] = data.train_val_test_split(val_ratio=val_ratio, test_ratio=test_ratio)

    # # === OpenSky ===
    # name = "opensky"
    # dataset = PyGLinkPropPredDataset(name=name, root="datasets")
    # train_mask = dataset.train_mask
    # val_mask = dataset.val_mask
    # test_mask = dataset.test_mask
    # data = dataset.data[0]

    # data_splits['train'] = data[train_mask]
    # data_splits['val'] = data[val_mask]
    # data_splits['test'] = data[test_mask]


    # Ensure to only sample actual destination nodes as negatives.
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
    
    # After successfully loading the dataset...
    if neg_sample_strategy == 'hist_rnd':
        historical_data = data_splits['train']
    else:
        historical_data = None

    neg_sampler = NegativeEdgeGenerator(dataset_name=dataset_name, first_dst_id=min_dst_idx, last_dst_id=max_dst_idx, 
                                        num_neg_e=num_neg_e_per_pos, strategy=neg_sample_strategy,
                                        rnd_seed=rnd_seed, historical_data=historical_data)

    # generate evaluation set
    partial_path = f'{path}/{dataset_name}/'

    # generate validation negative edge set
    start_time = time.time()
    split_mode = 'val'
    print(f"INFO: Start generating negative samples: {split_mode} --- {neg_sample_strategy}")
    neg_sampler.generate_negative_samples(data=data_splits[split_mode], split_mode=split_mode, partial_path=partial_path)
    print(f"INFO: End of negative samples generation. Elapsed Time (s): {time.time() - start_time: .4f}")

    # generate test negative edge set
    start_time = time.time()
    split_mode = 'test'
    print(f"INFO: Start generating negative samples: {split_mode} --- {neg_sample_strategy}")
    neg_sampler.generate_negative_samples(data=data_splits[split_mode], split_mode=split_mode, partial_path=partial_path)
    print(f"INFO: End of negative samples generation. Elapsed Time (s): {time.time() - start_time: .4f}")




if __name__ == '__main__':
    main()
    




