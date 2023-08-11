"""
Dataset statistics
"""

import numpy as np
import pandas as pd
import networkx as nx
import argparse

from torch_geometric.loader import TemporalDataLoader
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
# from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
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


def get_index_metrics(train_val_data, test_data):
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
    reoccurrence = float(intersect * 1.0 / train_val_size)
    return surprise, reoccurrence


def get_node_ratio(history_data, eval_data):
    r"""
    compute the ratio of new nodes
    """
    eval_uniq_nodes = set(eval_data['sources']).union(set(eval_data['destinations'])) 
    hist_uniq_nodes = set(history_data['sources']).union(set(history_data['destinations'])) 
    new_nodes = []
    for node in eval_uniq_nodes:
        if node not in hist_uniq_nodes:
            new_nodes.append(node)
    new_nodes = set(new_nodes)
    new_node_ratio = float(len(new_nodes) * 1.0 / len(eval_uniq_nodes))

    return new_node_ratio


def get_dataset_stats(data, temporal_stats=False):
    r"""
    returns simple stats based on counts
    """
    # simple stats
    sources, destinations, timestamps = data['full']['sources'], data['full']['destinations'], data['full']['timestamps']
    edgelist_df = pd.DataFrame(zip(sources, destinations, timestamps), columns=['src', 'dst', 'ts'])
    num_nodes = len(np.unique(np.concatenate((sources, destinations), axis=0)))
    num_edges = len(sources)  # = len(destinations) = len(timestamps)
    num_unique_ts = len(np.unique(timestamps))
    unique_e = get_unique_edges(sources, destinations)
    num_unique_e = len(unique_e)

    # compute temporal stats
    if temporal_stats:  # because it takes so long for large datasets...
        avg_e_per_ts = get_avg_e_per_ts(edgelist_df)
        avg_degree_per_ts = get_avg_degree(edgelist_df)
    else:
        avg_e_per_ts = -1
        avg_degree_per_ts = -1
    
    # compute reoccurrence & surprise
    surprise, reoccurrence = get_index_metrics(data['train_val'], data['test'])

    # compute new node ratio 
    val_nn_ratio = get_node_ratio(data['train'], data['val'])
    #test_nn_ratio = get_node_ratio(data['train_val'], data['test'])
    test_nn_ratio = get_node_ratio(data['train'], data['test'])


    stats_dict = {
                  'num_nodes': num_nodes,
                  'num_edges': num_edges,
                  'num_unique_ts': num_unique_ts,
                  'num_unique_e': num_unique_e,
                  'avg_e_per_ts': avg_e_per_ts,
                  'avg_degree_per_ts': avg_degree_per_ts,
                  'surprise': surprise,
                  'reocurrence': reoccurrence,
                  'val_nn_ratio': val_nn_ratio,
                  'test_nn_ratio': test_nn_ratio,
                  }
    return stats_dict


def main():
    r"""
    Generate dateset statistics
    """
    parser = argparse.ArgumentParser(description='Dataset statistics')
    parser.add_argument('-d', '--data', type=str, default='tgbl-wiki', help='random seed to use')
    parser.add_argument('--tempstats', action='store_true', default=False, help='whether compute temporal statistics')
    parser.parse_args()
    args = parser.parse_args()

    DATA = args.data
    temporal_stats = args.tempstats

    # data loading ...
    if DATA in ['tgbl-wiki', 'tgbl-review', 'tgbl-flight', 'tgbl-comment', 'tgbl-coin']:
        # load data: link prop. pred. with `numpy`
        dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)
        data = dataset.full_data  

        # get masks
        train_mask = dataset.train_mask
        val_mask = dataset.val_mask
        test_mask = dataset.test_mask
        train_data = {'sources': data['sources'][train_mask],
                      'destinations': data['destinations'][train_mask],
                      }
        val_data = {'sources': data['sources'][val_mask],
                      'destinations': data['destinations'][val_mask],
                      }
        train_val_data = {'sources': np.concatenate([data['sources'][train_mask], data['sources'][val_mask]]),
                      'destinations': np.concatenate([data['destinations'][train_mask], data['destinations'][val_mask]]),
                      }
        test_data = {'sources': data['sources'][test_mask],
                      'destinations': data['destinations'][test_mask],
                      }
        full_data = {'sources': data['sources'], 
                     'destinations': data['destinations'], 
                     'timestamps': data['timestamps'],
                     }

    elif DATA in ['tgbn-trade', 'tgbn-genre', 'tgbn-reddit', 'tgbn-token']:
        # load data: node prop. pred.
        dataset = PyGNodePropPredDataset(name=DATA, root="datasets")
        data = dataset.get_TemporalData()
        
        # split data
        train_mask = dataset.train_mask
        val_mask = dataset.val_mask
        test_mask = dataset.test_mask
        train_val_mask = np.logical_or(np.array(train_mask), np.array(val_mask))

        train_data = {'sources': np.array(data[train_mask].src),
                      'destinations': np.array(data[train_mask].dst),
                      }
        val_data = {'sources': np.array(data[val_mask].src),
                    'destinations': np.array(data[val_mask].dst),
                    }
        train_val_data = {'sources': np.concatenate([np.array(data[train_mask].src), np.array(data[val_mask].src)]),
                          'destinations': np.concatenate([np.array(data[train_mask].dst), np.array(data[val_mask].dst)]),
                          }
        test_data = {'sources': np.array(data[test_mask].src),
                     'destinations': np.array(data[test_mask].dst),
                     } 
        full_data = {'sources': np.array(data.src), 
                     'destinations': np.array(data.dst), 
                     'timestamps': np.array(data.t),
                     }

    else:
        raise ValueError("Unsupported data!")

    split_data = {'train': train_data,
                  'val': val_data,
                  'train_val': train_val_data,
                  'test': test_data,
                  'full': full_data,
                  }
    # compute dataset statistics...
    print("=============================")
    print(f">>> DATA: {DATA}")
    dataset_stats = get_dataset_stats(split_data, temporal_stats)
    for k, v in dataset_stats.items():
        print(f"{k}: {v}")
    print("=============================")


if __name__ == "__main__":
    main()