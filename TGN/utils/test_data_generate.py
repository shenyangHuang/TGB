"""
Generate test set data for the test phase according to EGO-SNAPSHOT evaluation protocol

Date:
    - March 1, 2023
"""


import math
import time
import numpy as np
import pandas as pd

from utils.data_load import Data




def generate_pos_graph_snapshots(split_data, num_snapshots):
    """
    divide the given data into snapshots
    """
    timestamps = split_data.timestamps
    unique_timestamps = np.unique(timestamps)
    num_unique_timestamps = len(unique_timestamps)
    print(f"INFO: Number of unique timestamps: {num_unique_timestamps}.")
    snapshot_size = int(math.ceil((max(unique_timestamps) - min(unique_timestamps)) / num_snapshots))

    snapshot_list = []
    for snap_index in range(num_snapshots):
        start_ts = min(unique_timestamps) + snap_index * snapshot_size
        end_ts = start_ts + snapshot_size + 1
        current_masks = np.logical_and(start_ts <= timestamps, timestamps < end_ts)
        if len(split_data.timestamps[current_masks]) > 0:
            current_snapshot = Data(split_data.sources[current_masks], split_data.destinations[current_masks],
                                    split_data.timestamps[current_masks], split_data.edge_idxs[current_masks],
                                    split_data.true_y[current_masks])
            print(
                f"INFO: {snap_index}: start_ts: {min(split_data.timestamps[current_masks])}, end_ts: {max(split_data.timestamps[current_masks])}: Number of edges: {len(split_data.timestamps[current_masks])}")
            snapshot_list.append(current_snapshot)
        else:
            print(f"INFO: {snap_index}: start_ts: {start_ts}, end_ts: {end_ts}: NO EDGES!")
    return snapshot_list


def get_unique_edges(sources, destinations):
    """
    return a dictionary of unique edges
    """
    unique_e_dict = {}
    for src, dst in zip(sources, destinations):
        if (src, dst) not in unique_e_dict:
            unique_e_dict[(src, dst)] = 1
    return unique_e_dict


def generate_all_negative_edges(pos_data_split, all_dsts):
    """
    generate all negative edges for the positive edge set
    """
    pos_edges = get_unique_edges(pos_data_split.sources, pos_data_split.destinations)
    source_unique = np.unique(pos_data_split.sources)
    all_destinations_unique = np.unique(all_dsts)
    neg_srcs, neg_dsts = [], []
    for src in source_unique:
        for dst in all_destinations_unique:
            if (src, dst) not in pos_edges:  # Collision check for positive vs. negative edges
                neg_srcs.append(src)
                neg_dsts.append(dst)

    return np.array(neg_srcs), np.array(neg_dsts)


def get_last_positive_edge_info(pos_e_data):
    """
    return the last latest positive edge that has been observed if there are repetition of the same positive edge
    """

    sources = pos_e_data.sources
    destinations = pos_e_data.destinations
    timestamps = pos_e_data.timestamps
    e_idxs = pos_e_data.edge_idxs
    last_pos_info = {}
    for idx, src in enumerate(sources):
        if src not in last_pos_info:
            last_pos_info[src] = {'timestamp': timestamps[idx],
                                   'edge_idx': e_idxs[idx],
                                              }
        else:
            if last_pos_info[src]['timestamp'] < timestamps[idx]:
                last_pos_info[src]['timestamp'] = timestamps[idx]
                last_pos_info[src]['edge_idx'] = e_idxs[idx]
    return last_pos_info


def generate_test_edge_for_one_snapshot(pos_snap, full_data):
    """
    generate the test edges only for one snapshot
    """
    all_neg_srcs, all_neg_dsts = generate_all_negative_edges(pos_snap, full_data.destinations)
    last_pos_info = get_last_positive_edge_info(pos_snap)

    neg_ts_l, neg_e_idx_l = [], []
    for src in all_neg_srcs:
        neg_ts_l.append(last_pos_info[src]['timestamp'])
        neg_e_idx_l.append(last_pos_info[src]['edge_idx'])

    snap_data = {'pos_e': pos_snap,
                 'neg_e': Data(all_neg_srcs, all_neg_dsts, np.array(neg_ts_l), 
                 np.array(neg_e_idx_l), np.array([0 for _ in range(len(all_neg_srcs))]))}

    return snap_data