from datetime import datetime
from itertools import groupby
from operator import itemgetter
import numpy as np


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
