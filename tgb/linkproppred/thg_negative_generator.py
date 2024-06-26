"""
Sample and Generate negative edges that are going to be used for evaluation of a dynamic graph learning model
Negative samples are generated and saved to files ONLY once; 
    other times, they should be loaded from file with instances of the `negative_sampler.py`.
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import TemporalData
from tgb.utils.utils import save_pkl
from typing import Union


"""
negative sample generator for tkg datasets 
temporal filterted MRR
"""
class THGNegativeEdgeGenerator(object):
    def __init__(
        self,
        dataset_name: str,
        first_node_id: int,
        last_node_id: int,
        node_type: Union[np.ndarray, torch.Tensor],
        strategy: str = "node-type-filtered",
        num_neg_e: int = -1,  # -1 means generate all possible negatives
        rnd_seed: int = 1,
        edge_data: TemporalData = None,
    ) -> None:
        r"""
        Negative Edge Generator class for Temporal Heterogeneous Graphs
        this is a class for generating negative samples for a specific datasets
        the set of the positive samples are provided, the negative samples are generated with specific strategies 
        and are saved for consistent evaluation across different methods

        Parameters:
            dataset_name: name of the dataset
            first_node_id: the first node id
            last_node_id: the last node id
            node_type: the node type of each node
            strategy: the strategy to generate negative samples
            num_neg_e: number of negative samples to generate
            rnd_seed: random seed
            edge_data: the edge data object containing the positive edges
        Returns:
            None
        """
        self.rnd_seed = rnd_seed
        np.random.seed(self.rnd_seed)
        self.dataset_name = dataset_name
        self.first_node_id = first_node_id
        self.last_node_id = last_node_id
        if isinstance(node_type, torch.Tensor):
            node_type = node_type.cpu().numpy()
        self.node_type = node_type
        self.node_type_dict = self.get_destinations_based_on_node_type(first_node_id, last_node_id, self.node_type) # {node_type: {nid:1}}
        assert isinstance(self.node_type, np.ndarray), "node_type should be a numpy array"
        self.num_neg_e = num_neg_e  #-1 means generate all 

        assert strategy in [
            "node-type-filtered",
            "random",
        ], "The supported strategies are `node-type-filtered`"
        self.strategy = strategy
        self.edge_data = edge_data

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

    def generate_negative_samples(self, 
                                  pos_edges: TemporalData,
                                  split_mode: str, 
                                  partial_path: str,
                                  ) -> None:
        r"""
        Generate negative samples

        Parameters:
            pos_edges: positive edges to generate the negatives for
            split_mode: specifies whether to generate negative edges for 'validation' or 'test' splits
            partial_path: in which directory save the generated negatives
        """
        # file name for saving or loading...
        filename = (
            partial_path
            + "/"
            + self.dataset_name
            + "_"
            + split_mode
            + "_"
            + "ns"
            + ".pkl"
        )

        if self.strategy == "node-type-filtered":
            self.generate_negative_samples_nt(pos_edges, split_mode, filename)
        elif self.strategy == "random":
            self.generate_negative_samples_random(pos_edges, split_mode, filename)
        else:
            raise ValueError("Unsupported negative sample generation strategy!")

    def generate_negative_samples_nt(self, 
                                      data: TemporalData, 
                                      split_mode: str, 
                                      filename: str,
                                      ) -> None:
        r"""
        now we consider (s, d, t, edge_type) as a unique edge, also adding the node type info for the destination node for convenience so (s, d, t, edge_type): (conflict_set, d_node_type)
        Generate negative samples based on the random strategy:
            - for each positive edge, retrieve all possible destinations based on the node type of the destination node
            - filter actual positive edges at the same timestamp with the same edge type
        
        Parameters:
            data: an object containing positive edges information
            split_mode: specifies whether to generate negative edges for 'validation' or 'test' splits
            filename: name of the file containing the generated negative edges
        """
        print(
            f"INFO: Negative Sampling Strategy: {self.strategy}, Data Split: {split_mode}"
        )
        assert split_mode in [
            "val",
            "test",
        ], "Invalid split-mode! It should be `val` or `test`!"

        if os.path.exists(filename):
            print(
                f"INFO: negative samples for '{split_mode}' evaluation are already generated!"
            )
        else:
            print(f"INFO: Generating negative samples for '{split_mode}' evaluation!")
            # retrieve the information from the batch
            pos_src, pos_dst, pos_timestamp, edge_type = (
                data.src.cpu().numpy(),
                data.dst.cpu().numpy(),
                data.t.cpu().numpy(),
                data.edge_type.cpu().numpy(),
            )

            # generate a list of negative destinations for each positive edge
            pos_edge_tqdm = tqdm(
                zip(pos_src, pos_dst, pos_timestamp, edge_type), total=len(pos_src)
            )

            edge_t_dict = {} # {(t, u, edge_type): {v_1, v_2, ..} }
            #! iterate once to put all edges into a dictionary for reference
            for (
                pos_s,
                pos_d,
                pos_t,
                edge_type,
            ) in pos_edge_tqdm:
                if (pos_t, pos_s, edge_type) not in edge_t_dict:
                    edge_t_dict[(pos_t, pos_s, edge_type)] = {pos_d:1}
                else:
                    edge_t_dict[(pos_t, pos_s, edge_type)][pos_d] = 1

            out_dict = {}
            for key in tqdm(edge_t_dict):
                conflict_set = np.array(list(edge_t_dict[key].keys()))
                pos_d = conflict_set[0]
                #* retieve the node type of the destination node as well 
                #! assumption, same edge type = same destination node type
                d_node_type = int(self.node_type[pos_d - self.first_node_id])
                all_dst = self.node_type_dict[d_node_type]
                if (self.num_neg_e == -1):
                    filtered_all_dst = np.setdiff1d(all_dst, conflict_set)
                else:
                    #* lazy sampling
                    neg_d_arr = np.random.choice(
                        all_dst, self.num_neg_e, replace=False) #never replace negatives
                    if len(np.setdiff1d(neg_d_arr, conflict_set)) < self.num_neg_e:
                        neg_d_arr = np.random.choice(
                            np.setdiff1d(all_dst, conflict_set), self.num_neg_e, replace=False)
                    filtered_all_dst = neg_d_arr
                out_dict[key] = filtered_all_dst
            print ("ns samples for ", len(out_dict), " positive edges are generated")
            # save the generated evaluation set to disk
            save_pkl(out_dict, filename)

    def generate_negative_samples_random(self, 
                                      data: TemporalData, 
                                      split_mode: str, 
                                      filename: str,
                                      ) -> None:
        r"""
        generate random negative edges for ablation study
        
        Parameters:
            data: an object containing positive edges information
            split_mode: specifies whether to generate negative edges for 'validation' or 'test' splits
            filename: name of the file containing the generated negative edges
        """
        print(
            f"INFO: Negative Sampling Strategy: {self.strategy}, Data Split: {split_mode}"
        )
        assert split_mode in [
            "val",
            "test",
        ], "Invalid split-mode! It should be `val` or `test`!"

        if os.path.exists(filename):
            print(
                f"INFO: negative samples for '{split_mode}' evaluation are already generated!"
            )
        else:
            print(f"INFO: Generating negative samples for '{split_mode}' evaluation!")
            # retrieve the information from the batch
            pos_src, pos_dst, pos_timestamp, edge_type = (
                data.src.cpu().numpy(),
                data.dst.cpu().numpy(),
                data.t.cpu().numpy(),
                data.edge_type.cpu().numpy(),
            )
            first_dst_id = self.edge_data.dst.min()
            last_dst_id = self.edge_data.dst.max()
            all_dst = np.arange(first_dst_id, last_dst_id + 1)
            evaluation_set = {}
            # generate a list of negative destinations for each positive edge
            pos_edge_tqdm = tqdm(
                zip(pos_src, pos_dst, pos_timestamp, edge_type), total=len(pos_src)
            )

            for (
                pos_s,
                pos_d,
                pos_t,
                edge_type,
            ) in pos_edge_tqdm:
                t_mask = pos_timestamp == pos_t
                src_mask = pos_src == pos_s
                fn_mask = np.logical_and(t_mask, src_mask)
                pos_e_dst_same_src = pos_dst[fn_mask]
                filtered_all_dst = np.setdiff1d(all_dst, pos_e_dst_same_src)
                if (self.num_neg_e > len(filtered_all_dst)):
                    neg_d_arr = filtered_all_dst
                else:
                    neg_d_arr = np.random.choice(
                    filtered_all_dst, self.num_neg_e, replace=False) #never replace negatives
                evaluation_set[(pos_t, pos_s, edge_type)] = neg_d_arr
            save_pkl(evaluation_set, filename)









