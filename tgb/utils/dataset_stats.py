"""
Dataset statistics
"""

import numpy as np
import pandas as pd
import networkx as nx

from torch_geometric.loader import TemporalDataLoader
from tgb.linkproppred.dataset import LinkPropPredDataset




def get_unique_edges(sources, destination):
    r"""
    return unique edges
    """
    unique_e = {}
    for src, dst in zip(sources, destination):
        if (src, dst) not in unique_e:
            unique_e[(src, dst)] = True
    return unique_e


def get_avg_e_per_ts(edgelist_df):
    r"""
    get the average number of edges per each timestamp
    """
    sum_num_e_per_ts = 0
    unique_ts = np.unique(np.array(edgelist_df['ts'].tolist()))
    for ts in unique_ts:
        num_e_at_this_ts = len(edgelist_df.loc[edgelist_df['ts'] == ts])
        sum_num_e_per_ts += num_e_at_this_ts
    avg_num_e_per_ts = (sum_num_e_per_ts * 1.0) / len(unique_ts)
    
    # print(f"INFO: avg_num_e_per_ts: {avg_num_e_per_ts}")
    return avg_num_e_per_ts


def get_avg_degree(edgelist_df):
    r"""
    get average degree over the timestamps
    """
    degree_avg_at_ts_list = []
    unique_ts = np.unique(np.array(edgelist_df['ts'].tolist()))
    for ts in unique_ts:  
        e_at_this_ts = edgelist_df.loc[edgelist_df['ts'] == ts]
        G = nx.MultiGraph()
        for idx, e_row in e_at_this_ts.iterrows():
            G.add_edge(e_row['src'], e_row['dst'], weight=e_row['ts'])
        nodes = G.nodes()
        degrees = [G.degree[n] for n in nodes]
        degree_avg_at_ts_list.append(np.mean(degrees))

    # print(f"INFO: avg_degree: {np.mean(degree_avg_at_ts_list)}")
    
    return np.mean(degree_avg_at_ts_list)


def get_dataset_stats(data):
    r"""
    returns simple stats based on counts
    """
    sources, destinations, timestamps = data['sources'], data['destinations'], data['timestamps']
    edgelist_df = pd.DataFrame(zip(sources, destinations, timestamps), columns=['src', 'dst', 'ts'])
    num_nodes = len(np.unique(np.concatenate((sources, destinations), axis=0)))
    num_edges = len(sources)  # = len(destinations) = len(timestamps)
    num_unique_ts = len(np.unique(timestamps))
    unique_e = get_unique_edges(sources, destinations)
    num_unique_e = len(unique_e)
    avg_e_per_ts = get_avg_e_per_ts(edgelist_df)
    avg_degree_per_ts = get_avg_degree(edgelist_df)

    stats_dict = {'num_nodes': num_nodes,
                  'num_edges': num_edges,
                  'num_unique_ts': num_unique_ts,
                  'num_unique_e': num_unique_e,
                  'avg_e_per_ts': avg_e_per_ts,
                  'avg_degree_per_ts': avg_degree_per_ts,
                  }
    return stats_dict


def main():
    r"""
    Generate dateset statistics
    """
    DATA = "stablecoin"

    # load data
    dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)
    data = dataset.full_data   

    dataset_stats = get_dataset_stats(data)
    print(f"DATA: {DATA}")
    for k, v in dataset_stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()