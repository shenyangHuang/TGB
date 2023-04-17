""""
Parsing the arguments

Date: 
    - Feb. 28, 2023
"""
import argparse
import sys


def get_args():
    ### Argument and global variables
    parser = argparse.ArgumentParser('TGN Training Arguments')
    # data parameters
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='wikipedia')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='Whether to use disjoint set of new nodes for train and val')
    parser.add_argument('--data_usage', default=1.0, type=float, help='Use a fraction of the data (0-1)')
    parser.add_argument('--randomize_features', action='store_true', help='Whether to randomize node features')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of the validation data')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Ratio of the test data')
    # model parameters
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
    parser.add_argument('--use_memory', action='store_true', help='Whether to augment the model with a node memory')
    parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=["graph_attention", "graph_sum", "identity", "time"], 
                        help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity", choices=["mlp", "identity"], help='Type of message function')
    parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn"], help='Type of memory updater')
    parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
    parser.add_argument('--memory_update_at_end', action='store_true',
                        help='Whether to update memory at the end or at the start of the batch')
    parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
    parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for each user')
    parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the destination node as part of the message')
    parser.add_argument('--use_source_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the source node as part of the message')
    parser.add_argument('--dyrep', action='store_true', help='Whether to run the dyrep model')
    # training parameters
    parser.add_argument('--bs', type=int, default=200, help='Batch_size')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # setup parameters
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
    # negative sampling parameters
    parser.add_argument('--tr_rnd_ne_ratio', type=float, default=1.0,
                        help='Ratio of RANDOM negative edges sampled during training.')
    parser.add_argument('--ts_rnd_ne_ratio', type=float, default=1.0,
                        help='Ratio of RANDOM negative edges sampled during TEST phase.')
    parser.add_argument('--tr_neg_sample', type=str, default='haphaz_rnd', choices=['rnd', 'hist', 'induc', 'haphaz_rnd'],
                        help='Strategy for the edge negative sampling at training.')
    parser.add_argument('--ts_neg_sample', type=str, default='haphaz_rnd', choices=['rnd', 'hist', 'induc', 'haphaz_rnd'],
                        help='Strategy for the edge negative sampling at test.')
    parser.add_argument('--ego_snap', action='store_true', help='Whether to evaluate in EGO-SNAPSHOT manner')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args, sys.argv