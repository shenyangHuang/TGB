"""
Sample and Generate negative edges that are going to be used for evaluation of a dynamic graph learning model
Negative samples are generated and saved to files ONLY once; 
    other times, they should be loaded from file with instances of the `negative_sampler.py`.
"""
import numpy as np
from torch_geometric.data import TemporalData
import matplotlib.pyplot as plt
from tgb.utils.utils import save_pkl
import os
from tqdm import tqdm


"""
negative sample generator for tkg datasets 
temporal filterted MRR
"""
class TKGNegativeEdgeGenerator(object):
    def __init__(
        self,
        dataset_name: str,
        first_dst_id: int,
        last_dst_id: int,
        strategy: str = "time-filtered",
        num_neg_e: int = -1,  # -1 means generate all possible negatives
        rnd_seed: int = 1,
        partial_path: str = None,
        edge_data: TemporalData = None,
    ) -> None:
        r"""
        Negative Edge Generator class for Temporal Knowledge Graphs
        constructor for the negative edge generator class

        Parameters:
            dataset_name: name of the dataset
            first_dst_id: identity of the first destination node
            last_dst_id: indentity of the last destination node
            num_neg_e: number of negative edges being generated per each positive edge
            strategy: specifies which strategy should be used for generating the negatives
            rnd_seed: random seed for reproducibility
            edge_data: the positive edges to generate the negatives for, assuming sorted temporally
        
        Returns:
            None
        """
        self.rnd_seed = rnd_seed
        np.random.seed(self.rnd_seed)
        self.dataset_name = dataset_name
        self.first_dst_id = first_dst_id
        self.last_dst_id = last_dst_id      
        self.num_neg_e = num_neg_e  #-1 means generate all 
        assert strategy in [
            "time-filtered",
            "dst-time-filtered",
            "random"
        ], "The supported strategies are `time-filtered`, dst-time-filtered, random"
        self.strategy = strategy
        self.dst_dict = None
        if self.strategy == "dst-time-filtered":
            if partial_path is None:
                raise ValueError(
                    "The partial path to the directory where the dst_dict is stored is required")
            else:
                self.dst_dict_name = (
                    partial_path
                    + "/"
                    + self.dataset_name
                    + "_"
                    + "dst_dict"
                    + ".pkl"
                )
                self.dst_dict = self.generate_dst_dict(edge_data=edge_data, dst_name=self.dst_dict_name)
        self.edge_data = edge_data

    def generate_dst_dict(self, edge_data: TemporalData, dst_name: str) -> dict:
        r"""
        Generate a dictionary of destination nodes for each type of edge

        Parameters:
            edge_data: an object containing positive edges information
            dst_name: name of the file to save the generated dictionary of destination nodes
        
        Returns:
            dst_dict: a dictionary of destination nodes for each type of edge
        """

        min_dst_idx, max_dst_idx = int(edge_data.dst.min()), int(edge_data.dst.max())

        pos_src, pos_dst, pos_timestamp, edge_type = (
            edge_data.src.cpu().numpy(),
            edge_data.dst.cpu().numpy(),
            edge_data.t.cpu().numpy(),
            edge_data.edge_type.cpu().numpy(),
        )



        dst_track_dict = {} # {edge_type: {dst_1, dst_2, ..} }

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
            if edge_type not in dst_track_dict:
                dst_track_dict[edge_type] = {pos_d:1}
            else:
                dst_track_dict[edge_type][pos_d] = 1
        dst_dict = {}
        edge_type_size = []
        for key in dst_track_dict:
            dst = np.array(list(dst_track_dict[key].keys()))
            edge_type_size.append(len(dst))
            dst_dict[key] = dst
        print ('destination candidates generated for all edge types ', len(dst_dict))
        return dst_dict

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

        if self.strategy == "time-filtered":
            self.generate_negative_samples_ftr(pos_edges, split_mode, filename)
        elif self.strategy == "dst-time-filtered":
            self.generate_negative_samples_dst(pos_edges, split_mode, filename)
        elif self.strategy == "random":
            self.generate_negative_samples_random(pos_edges, split_mode, filename)
        else:
            raise ValueError("Unsupported negative sample generation strategy!")
        
    def generate_negative_samples_ftr(self, 
                                      data: TemporalData, 
                                      split_mode: str, 
                                      filename: str,
                                      ) -> None:
        r"""
        now we consider (s, d, t, edge_type) as a unique edge
        Generate negative samples based on the random strategy:
            - for each positive edge, sample a batch of negative edges from all possible edges with the same source node
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

            conflict_dict = {}
            for key in edge_t_dict:
                conflict_dict[key] = np.array(list(edge_t_dict[key].keys()))
            
            print ("conflict sets for ns samples for ", len(conflict_dict), " positive edges are generated")
            # save the generated evaluation set to disk
            save_pkl(conflict_dict, filename)


    def generate_negative_samples_dst(self, 
                                      data: TemporalData, 
                                      split_mode: str, 
                                      filename: str,
                                      ) -> None:
        r"""
        now we consider (s, d, t, edge_type) as a unique edge
        Generate negative samples based on the random strategy:
            - for each positive edge, sample a batch of negative edges from all possible edges with the same source node
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
            if self.dst_dict is None:
                raise ValueError("The dst_dict is not generated!")

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
            out_dict = {}
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


            pos_src, pos_dst, pos_timestamp, edge_type = (
                data.src.cpu().numpy(),
                data.dst.cpu().numpy(),
                data.t.cpu().numpy(),
                data.edge_type.cpu().numpy(),
            )

            new_pos_edge_tqdm = tqdm(
                zip(pos_src, pos_dst, pos_timestamp, edge_type), total=len(pos_src)
            )

            min_dst_idx, max_dst_idx = int(self.edge_data.dst.min()), int(self.edge_data.dst.max())


            for (
                pos_s,
                pos_d,
                pos_t,
                edge_type,
            ) in new_pos_edge_tqdm:
                #* generate based on # of ns samples
                conflict_set = np.array(list(edge_t_dict[(pos_t, pos_s, edge_type)].keys()))
                dst_set = self.dst_dict[edge_type]  #dst_set contains conflict set
                sample_num = self.num_neg_e
                filtered_dst_set = np.setdiff1d(dst_set, conflict_set) #more efficient
                dst_sampled = None
                all_dst = np.arange(min_dst_idx, max_dst_idx+1)
                if len(filtered_dst_set) < (sample_num):
                    #* with collision check
                    filtered_sample_set = np.setdiff1d(all_dst, filtered_dst_set)
                    dst_sampled = np.random.choice(filtered_sample_set, sample_num, replace=False)
                    # #* remove the conflict set from dst set
                    dst_sampled[0:len(filtered_dst_set)] = filtered_dst_set[:]
                else:
                    # dst_sampled = rng.choice(max_dst_idx+1, sample_num, replace=False)
                    dst_sampled = np.random.choice(filtered_dst_set, sample_num, replace=False)
                

                if (dst_sampled.shape[0] > sample_num):
                    print ("I am the bug that Julia worries about")
                    print ("dst_sampled shape is ", dst_sampled.shape)
                out_dict[(pos_t, pos_s, edge_type)] = dst_sampled
            
            print ("negative samples for ", len(out_dict), " positive edges are generated")
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

    