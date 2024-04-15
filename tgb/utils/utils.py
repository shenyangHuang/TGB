import random
import os
import pickle
import sys
import argparse
import json
import torch
from typing import Any
import numpy as np
from torch_geometric.data import TemporalData
import pandas as pd
import dgl
import torch
from itertools import groupby
from operator import itemgetter
from collections import defaultdict


def add_inverse_quadruples(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    adds the inverse relations required for the model to the dataframe
    """
    if ("edge_type" not in df):
        raise ValueError("edge_type is required to invert relation in TKG")
    
    sources = np.array(df["u"])
    destinations = np.array(df["i"])
    timestamps = np.array(df["ts"])
    edge_idxs = np.array(df["idx"])
    weights = np.array(df["w"])
    edge_type = np.array(df["edge_type"])

    num_rels = np.unique(edge_type).shape[0]
    inv_edge_type = edge_type + num_rels

    all_sources = np.concatenate([sources, destinations])
    all_destinations = np.concatenate([destinations, sources])
    all_timestamps = np.concatenate([timestamps, timestamps])
    all_edge_idxs = np.concatenate([edge_idxs, edge_idxs+edge_idxs.max()+1])
    all_weights = np.concatenate([weights, weights])
    all_edge_types = np.concatenate([edge_type, inv_edge_type])

    return pd.DataFrame(
            {
                "u": all_sources,
                "i": all_destinations,
                "ts": all_timestamps,
                "label": np.ones(all_timestamps.shape[0]),
                "idx": all_edge_idxs,
                "w": all_weights,
                "edge_type": all_edge_types,
            }
        )



def add_inverse_quadruples_np(quadruples: np.array, 
                              num_rels:int) -> np.array:
    """
    creates an inverse quadruple for each quadruple in quadruples. inverse quadruple swaps subject and objsect, and increases 
    relation id by num_rels
    :param quadruples: [np.array] dataset quadruples, [src, relation_id, dst, timestamp ]
    :param num_rels: [int] number of relations that we have originally
    returns all_quadruples: [np.array] quadruples including inverse quadruples
    """
    inverse_quadruples = quadruples[:, [2, 1, 0, 3]]
    inverse_quadruples[:, 1] = inverse_quadruples[:, 1] + num_rels  # we also need inverse quadruples
    all_quadruples = np.concatenate((quadruples[:,0:4], inverse_quadruples))
    return all_quadruples


def add_inverse_quadruples_pyg(data: TemporalData, num_rels:int=-1) -> list:
    r"""
    creates an inverse quadruple from PyG TemporalData object, returns both the original and inverse quadruples
    """
    timestamp = data.t
    head = data.src
    tail = data.dst
    msg = data.msg
    edge_type = data.edge_type #relation
    num_rels = torch.max(edge_type).item() + 1
    inv_type = edge_type + num_rels
    all_data = TemporalData(src=torch.cat([head, tail]), 
                            dst=torch.cat([tail, head]), 
                            t=torch.cat([timestamp, timestamp.clone()]), 
                            edge_type=torch.cat([edge_type, inv_type]), 
                            msg=torch.cat([msg, msg.clone()]),
                            y = torch.cat([data.y, data.y.clone()]),)
    return all_data



# import torch
def save_pkl(obj: Any, fname: str) -> None:
    r"""
    save a python object as a pickle file
    """
    with open(fname, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(fname: str) -> Any:
    r"""
    load a python object from a pickle file
    """
    with open(fname, "rb") as handle:
        return pickle.load(handle)


def set_random_seed(seed: int):
    r"""
    setting random seed for reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_args():
    parser = argparse.ArgumentParser('*** TGB ***')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='tgbl-wiki')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--bs', type=int, help='Batch size', default=200)
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)
    parser.add_argument('--num_epoch', type=int, help='Number of epochs', default=50)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--mem_dim', type=int, help='Memory dimension', default=100)
    parser.add_argument('--time_dim', type=int, help='Time dimension', default=100)
    parser.add_argument('--emb_dim', type=int, help='Embedding dimension', default=100)
    parser.add_argument('--tolerance', type=float, help='Early stopper tolerance', default=1e-6)
    parser.add_argument('--patience', type=float, help='Early stopper patience', default=5)
    parser.add_argument('--num_run', type=int, help='Number of iteration runs', default=1)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv

def get_args_cen():
    parser = argparse.ArgumentParser(description='CEN')
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, default='tkgl-yago',
                        help="dataset to use")
    parser.add_argument("--test", type=int, default=0,
                        help="1: formal test 2: continual test")
  
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")

    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    
    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=1,
                        help="weight of entity prediction task")
    parser.add_argument("--kl-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
   
    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")

    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=True,
                        help="add entity prediction loss")


    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=30,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--ft_epochs", type=int, default=30,
                        help="number of minimum fine-tuning epoch")
    parser.add_argument("--ft_lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--norm_weight", type=float, default=1,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=10,
                        help="history length for test")
    parser.add_argument("--test-history-len-2", type=int, default=3,
                        help="history length for test")
    parser.add_argument("--start-history-len", type=int, default=3,
                    help="start history length")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--run-nr', type=int, help='Run Number', default=1)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv 

def get_args_regcn():
    parser = argparse.ArgumentParser(description='REGCN')

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, default='tkgl-yago',
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=0.5,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=True,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=1,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=1,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--run-nr', type=int, help='Run Number', default=1)
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv 

def save_results(new_results: dict, filename: str):
    r"""
    save (new) results into a json file
    :param: new_results (dictionary): a dictionary of new results to be saved
    :filename: the name of the file to save the (new) results
    """
    if os.path.isfile(filename):
        # append to the file
        with open(filename, 'r+') as json_file:
            file_data = json.load(json_file)
            # convert file_data to list if not
            if type(file_data) is dict:
                file_data = [file_data]
            file_data.append(new_results)
            json_file.seek(0)
            json.dump(file_data, json_file, indent=4)
    else:
        # dump the results
        with open(filename, 'w') as json_file:
            json.dump(new_results, json_file, indent=4)


def split_by_time(data):
    """
    https://github.com/Lee-zix/CEN/blob/main/rgcn/utils.py
    create list where each entry has an entry with all triples for this timestep
    """
    timesteps = list(set(data[:,3]))
    timesteps.sort()
    snapshot_list = [None] * len(timesteps)

    for index, ts in enumerate(timesteps):
        mask = np.where(data[:, 3] == ts)[0]
        snapshot_list[index] = data[mask,:3]

    return snapshot_list


def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    # uniq_r = np.concatenate((uniq_r, uniq_r+num_rels)) #we already have the inverse triples
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        r_to_e[rel+num_rels].add(src)
        r_to_e[rel+num_rels].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx

def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu):
    """
    https://github.com/Lee-zix/CEN/blob/main/rgcn/utils.py
    :param node_id: node id in the large graph
    :param num_rels: number of relation
    :param src: relabeled src id
    :param rel: original rel id
    :param dst: relabeled dst id
    :param use_cuda:
    :return:
    """
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    src, rel, dst = triples.transpose()
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)


    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r = uniq_r
    g.r_to_e = r_to_e
    g.r_len = r_len

    if use_cuda:
        g = g.to(gpu)
        g.r_to_e = torch.from_numpy(np.array(r_to_e))
    return g



def group_by(data: np.array, key_idx: int) -> dict:
    """
    group data in an np array to dict; where key is specified by key_idx. for example groups elements of array by relations
    :param data: [np.array] data to be grouped
    :param key_idx: [int] index for element of interest
    returns data_dict: dict with key: values of element at index key_idx, values: all elements in data that have that value
    """
    data_dict = {}
    data_sorted = sorted(data, key=itemgetter(key_idx))
    for key, group in groupby(data_sorted, key=itemgetter(key_idx)):
        data_dict[key] = np.array(list(group))
    return data_dict


def reformat_ts(timestamps):
    """ reformat timestamps s.t. they start with 0, and have stepsize 1.
    :param timestamps: np.array() with timestamps
    returns: np.array(ts_new)
    """
    all_ts = list(set(timestamps))
    all_ts.sort()
    ts_min = np.min(all_ts)
    ts_dist = all_ts[1] - all_ts[0]

    ts_new = []
    timestamps2 = timestamps - ts_min
    for timestamp in timestamps2:
        timestamp = int(timestamp/ts_dist)
        ts_new.append(timestamp)
    return np.array(ts_new)

## preprocess: define rules
def create_basis_dict(data):
    ""
    """
    data: concatenated train and vali data, INCLUDING INVERSE QUADRUPLES. we need it for the relation ids.
    """
    rels = list(set(data[:,1]))
    basis_dict = {}
    for rel in rels:
        basis_id_new = []
        rule_dict = {}
        rule_dict["head_rel"] = int(rel)
        rule_dict["body_rels"] = [int(rel)] #same body and head relation -> what happened before happens again
        rule_dict["conf"] = 1 #same confidence for every rule
        rule_new = rule_dict
        basis_id_new.append(rule_new)
        basis_dict[str(rel)] = basis_id_new
    return basis_dict


def get_inv_relation_id(num_rels):
    """
    Get inverse relation id.
    parameters:
        num_rels (int): number of relations
    returns:
        inv_relation_id (dict): mapping of relation to inverse relation
    """
    inv_relation_id = dict()
    for i in range(int(num_rels / 2)):
        inv_relation_id[i] = i + int(num_rels / 2)
    for i in range(int(num_rels / 2), num_rels):
        inv_relation_id[i] = i % int(num_rels / 2)
    return inv_relation_id


def create_scores_array(predictions_dict, num_nodes):
    # predictions_dict is a dictionary mapping indices to values
    # num_nodes is the size of the array

    # Convert keys and values of the predictions_dict into NumPy arrays
    keys_array = np.array(list(predictions_dict.keys()))
    values_array = np.array(list(predictions_dict.values()))

    # Create an array of zeros with the desired shape
    predictions = np.zeros(num_nodes)

    # Use advanced indexing to scatter values into predictions array
    predictions[keys_array.astype(int)] = values_array.astype(float)
    return predictions

