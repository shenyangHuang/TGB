"""
Dataset statistics
"""

import numpy as np
import pandas as pd
import networkx as nx
import argparse

from torch_geometric.loader import TemporalDataLoader
from tgb.nodeproppred.dataset_pyg import PyGNodePropertyDataset
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


def get_index_stats(train_val_data, test_data):
    r"""
    compute `surprise` and `recurrence` indices
    """
    train_val_e_set = {}
    for src, dst in zip(train_val_data['sources'], train_val_data['destinations']):
        if (src, dst) not in train_val_e_set:
            train_val_e_set[(src, dst)] = True
    
    test_e_set = {}
    for src, dst in zip(test_data['sources'], test_data['destinations']):
        if (src, dst) not in test_e_set:
            test_e_set[(src, dst)] = True
    
    train_val_size = len(train_val_data['sources'])
    test_size = len(test_data['sources'])

    intersect = difference = 0
    for e in test_e_set:
        if e in train_val_e_set:
            intersect += 1
        else:
            difference += 1

    surprise = float(difference * 1.0 / test_size)
    recurrence = float(intersect * 1.0 / train_val_size)
    return surprise, recurrence


def get_dataset_stats(data, task='linkproppred'):
    r"""
    returns simple stats based on counts
    """
    if task == 'linkproppred':
        sources, destinations, timestamps = data['sources'], data['destinations'], data['timestamps']
    elif task == 'nodeproppred':
        sources, destinations, timestamps = data.src, data.dst, data.t
    else:
        raise ValueError(f"Task can either be `linkproppred` or `nodeproppred`! {task} not acceptable.")
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
                  'surprise': 0,
                  'reocurrence': 0,
                  }
    return stats_dict


def main():
    r"""
    Generate dateset statistics
    """
    parser = argparse.ArgumentParser(description='Dataset statistics')
    parser.add_argument('-d', '--data', type=str, default='wikipedia', help='random seed to use')
    parser.parse_args()
    args = parser.parse_args()

    DATA = args.data
    task = ''

    if DATA in ['wikipedia', 'amazonreview', 'opensky', 'redditcomments', 'stablecoin']:
        task = 'linkproppred'

        # load data: link prop. pred
        dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)
        data = dataset.full_data   

        # split data
        train_mask = dataset.train_mask
        val_mask = dataset.val_mask
        test_mask = dataset.test_mask
        temporal_data = dataset.get_TemporalData()
        
        train_val_mask = np.logical_or(train_mask, val_mask)
        train_val_data = temporal_data[train_val_mask]
        test_data = temporal_data[test_mask]

    elif DATA in ['un_trade', 'lastfmgenre', 'subreddits']:
        task = 'nodeproppred'

        # load data: node prop. pred.
        dataset = PyGNodePropertyDataset(name=DATA, root="datasets")
        data = dataset.get_TemporalData()
        
        # split data
        train_mask = dataset.train_mask
        val_mask = dataset.val_mask
        test_mask = dataset.test_mask

        train_val_mask = np.logical_or(train_mask, val_mask)
        train_val_data = data[train_val_mask]
        test_data = data[test_mask]

    else:
        raise ValueError("Unsupported data!")

    dataset_stats = get_dataset_stats(data, train_val_data, test_data, task)
    print(f"DATA: {DATA}")
    for k, v in dataset_stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()