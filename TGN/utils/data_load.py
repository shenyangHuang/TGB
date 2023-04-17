"""
Loading the data for the model training

"""

import numpy as np
import pandas as pd
import random

INIT_FEAT_DIM = 172  # pass the memory_dimension!!! It does not make sense to have this value hard-codded!

class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, true_y):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.true_y= true_y
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def get_data(dataset_name, args, logger=None, verbose=True):
    random.seed(args.seed)
    ### Load data and generate training, validation, and test split
    graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
    if args.data_usage < 1:
        g_df = g_df.iloc[:int(args.data_usage * g_df.shape[0])]
        logger.info('Use partial data, ratio: {}'.format(args.data_usage))

    edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
    node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    if dataset_name in ['enron', 'socialevolve', 'uci']:
        node_zero_padding = np.zeros((node_features.shape[0], INIT_FEAT_DIM - node_features.shape[1]))
        node_features = np.concatenate([node_features, node_zero_padding], axis=1)
        edge_zero_padding = np.zeros((edge_features.shape[0], INIT_FEAT_DIM - edge_features.shape[1]))
        edge_features = np.concatenate([edge_features, edge_zero_padding], axis=1)

    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - args.val_ratio - args.test_ratio), (1 - args.test_ratio)]))

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    # Compute nodes which appear at test time
    test_node_set = set(sources[timestamps > val_time]).union(
        set(destinations[timestamps > val_time]))
    # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
    # their edges from training
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))  # @TODO: make a variable/argument

    # Mask saying for each source and destination whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # Mask which is true for edges with both destination and source not being new test nodes (because
    # we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # For train we keep edges happening before the validation time which do not involve any new node
    # used for inductiveness
    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.sources).union(train_data.destinations)
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

    if args.different_new_nodes:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
        test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

        edge_contains_new_val_node_mask = np.array(
            [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
        edge_contains_new_test_node_mask = np.array(
            [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)

    else:
        edge_contains_new_node_mask = np.array(
            [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test with all edges
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask],
                             edge_idxs[new_node_val_mask], labels[new_node_val_mask])

    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                              labels[new_node_test_mask])

    if verbose:
        logger.info("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                    full_data.n_unique_nodes))
        logger.info("The training dataset has {} interactions, involving {} different nodes".format(
            train_data.n_interactions, train_data.n_unique_nodes))
        logger.info("The validation dataset has {} interactions, involving {} different nodes".format(
            val_data.n_interactions, val_data.n_unique_nodes))
        logger.info("The test dataset has {} interactions, involving {} different nodes".format(
            test_data.n_interactions, test_data.n_unique_nodes))
        logger.info("The new node validation dataset has {} interactions, involving {} different nodes".format(
            new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
        logger.info("The new node test dataset has {} interactions, involving {} different nodes".format(
            new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
        logger.info("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
            len(new_test_node_set)))

    return node_features, edge_features, full_data, train_data, val_data, test_data, \
           new_node_val_data, new_node_test_data



def get_data_LR(dataset_name, seed, use_validation=True, val_ratio=0.15, test_ratio=0.15):
    """
    genrate data splits for the Link Regression task
    NOTE:
        - For now, only "Transductive" setting is considered.
        - For the edge regression task, 'labels' field contains the true labels (i.e., the edge weights) that we are interested to predict
    """
    partial_path = f"./data/"
    # load the data
    edgelist_df = pd.read_csv(f"{partial_path}/ml_{dataset_name}.csv")
    edge_feats = np.load(f"{partial_path}/ml_{dataset_name}.npy")
    node_feats = np.load(f"{partial_path}/ml_{dataset_name}_node.npy")

    # split the data into train, validation, and test set
    val_time, test_time = list(np.quantile(edgelist_df['ts'], [1 - (val_ratio + test_ratio), 1 - test_ratio]))

    sources = edgelist_df['src'].values
    destinations = edgelist_df['dst'].values
    timestamps = edgelist_df['ts'].values
    edge_idxs = edgelist_df['idx'].values
    labels = edgelist_df['w'].values

    random.seed(seed)

    train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    test_mask = timestamps > test_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask], edge_idxs[train_mask], labels[train_mask])
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask], edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask], edge_idxs[test_mask], labels[test_mask])

    return node_feats, edge_feats, full_data, train_data, val_data, test_data

