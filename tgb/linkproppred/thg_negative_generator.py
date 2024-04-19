"""
Sample and Generate negative edges that are going to be used for evaluation of a dynamic graph learning model
Negative samples are generated and saved to files ONLY once; 
    other times, they should be loaded from file with instances of the `negative_sampler.py`.
"""
import numpy as np
from torch_geometric.data import TemporalData
from tgb.utils.utils import save_pkl
import os
from tqdm import tqdm

#! require node types


"""
negative sample generator for tkg datasets 
temporal filterted MRR
"""
class THGNegativeEdgeGenerator(object):
    def __init__(
        self,
        dataset_name: str,
        first_dst_id: int,
        last_dst_id: int,
        strategy: str = "time-filtered",
        num_neg_e: int = -1,  # -1 means generate all possible negatives
        rnd_seed: int = 1,
        edge_data: TemporalData = None,
    ) -> None:
        r"""
        Negative Edge Sampler class
        this is a class for generating negative samples for a specific datasets
        the set of the positive samples are provided, the negative samples are generated with specific strategies 
        and are saved for consistent evaluation across different methods
        negative edges are sampled with 'oen_vs_many' strategy.
        it is assumed that the destination nodes are indexed sequentially with 'first_dst_id' 
        and 'last_dst_id' being the first and last index, respectively.

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
        ], "The supported strategies are `time-filtered`"
        self.strategy = strategy
        self.edge_data = edge_data

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

            # all possible destinations
            all_dst = np.arange(self.first_dst_id, self.last_dst_id + 1)
            evaluation_set = {}
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

            pos_src, pos_dst, pos_timestamp, edge_type = (
                data.src.cpu().numpy(),
                data.dst.cpu().numpy(),
                data.t.cpu().numpy(),
                data.edge_type.cpu().numpy(),
            )
            
            conflict_dict = {}
            for key in edge_t_dict:
                conflict_dict[key] = np.array(list(edge_t_dict[key].keys()))
            
            print ("ns samples for ", len(conflict_dict), " positive edges are generated")

            # save the generated evaluation set to disk
            save_pkl(conflict_dict, filename)

            # # generate a list of negative destinations for each positive edge
            # pos_edge_tqdm = tqdm(
            #     zip(pos_src, pos_dst, pos_timestamp, edge_type), total=len(pos_src)
            # )

            
            # for (
            #     pos_s,
            #     pos_d,
            #     pos_t,
            #     edge_type,
            # ) in pos_edge_tqdm:
                
            # #! generate all negatives unless restricted
            # conflict_set = list(edge_t_dict[(pos_t, pos_s, edge_type)].keys())

            # # filter out positive destination
            # conflict_set = np.array(conflict_set)
            # filtered_all_dst = np.setdiff1d(all_dst, conflict_set)

            # '''
            # when num_neg_e is larger than all possible destinations simple return all possible destinations
            # '''
            # if (self.num_neg_e < 0):
            #     neg_d_arr = filtered_all_dst
            # elif (self.num_neg_e > len(filtered_all_dst)):
            #     neg_d_arr = filtered_all_dst
            # else:
            #     neg_d_arr = np.random.choice(
            #     filtered_all_dst, self.num_neg_e, replace=False) #never replace negatives

            # evaluation_set[(pos_s, pos_d, pos_t, edge_type)] = neg_d_arr

            # # save the generated evaluation set to disk
            # save_pkl(evaluation_set, filename)
