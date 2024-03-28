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
