import csv
import datetime
import os.path as osp
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

"""
function to process node type for thg datasets
"""


def process_node_type(
    fname: str,
    node_ids,
):
    """
    1. process the node type into integer
    3. return a numpy array of node types with index corresponding to node id
    """
    node_feat = np.zeros(len(node_ids))
    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # node_id,type
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                nid = int(row[0])
                try:
                    node_type = int(row[1])
                except:
                    raise ValueError(
                        row[1],
                        " is not an integer thus can't be a node type for thg dataset",
                    )
                try:
                    node_id = node_ids[nid]
                except:
                    raise ValueError(nid, " is not a valid node id")
                node_feat[node_id] = node_type
    return node_feat


"""
functions for thgl-forum dataset
"""


def csv_to_forum_data(
    fname: str,
) -> pd.DataFrame:
    r"""
    used by thgl-forum dataset
    convert the raw .csv data to pandas dataframe and numpy array
    input .csv file format should be: timestamp, head, tail, relation type
    Args:
        fname: the path to the raw data
    """
    feat_size = 2
    num_lines = sum(1 for line in open(fname)) - 1
    print("number of lines counted", num_lines)
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    label_list = np.zeros(num_lines)
    edge_type = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)
    node_ids = {}
    unique_id = 0

    word_max = 10000
    score_max = 10000

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # timestamp, head, tail, relation type
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                #! ts,src,dst,relation_type,num_words,score
                ts = int(row[0])  # converted to UNIX timestamp already
                src = int(row[1])
                dst = int(row[2])
                relation = int(row[3])
                num_words = int(row[4])
                score = int(row[5])
                if src not in node_ids:
                    node_ids[src] = unique_id
                    unique_id += 1
                if dst not in node_ids:
                    node_ids[dst] = unique_id
                    unique_id += 1
                u = node_ids[src]
                i = node_ids[dst]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = float(1)
                edge_type[idx - 1] = relation
                feat_l[idx - 1] = np.array([num_words / word_max, score / score_max])
                idx += 1
    return (
        pd.DataFrame(
            {
                "u": u_list,
                "i": i_list,
                "ts": ts_list,
                "label": label_list,
                "idx": idx_list,
                "w": w_list,
                "edge_type": edge_type,
            }
        ),
        feat_l,
        node_ids,
    )


"""
functions for thg dataset
"""


def csv_to_thg_data(
    fname: str,
) -> pd.DataFrame:
    r"""
    used by thgl-myket dataset
    convert the raw .csv data to pandas dataframe and numpy array
    input .csv file format should be: timestamp, head, tail, relation type
    Args:
        fname: the path to the raw data
    """
    feat_size = 1
    num_lines = sum(1 for line in open(fname)) - 1
    print("number of lines counted", num_lines)
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    label_list = np.zeros(num_lines)
    edge_type = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)
    node_ids = {}
    unique_id = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # timestamp, head, tail, relation type
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                ts = int(row[0])  # converted to UNIX timestamp already
                src = int(row[1])
                dst = int(row[2])
                relation = int(row[3])
                if src not in node_ids:
                    node_ids[src] = unique_id
                    unique_id += 1
                if dst not in node_ids:
                    node_ids[dst] = unique_id
                    unique_id += 1
                u = node_ids[src]
                i = node_ids[dst]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = float(1)
                edge_type[idx - 1] = relation
                idx += 1
    return (
        pd.DataFrame(
            {
                "u": u_list,
                "i": i_list,
                "ts": ts_list,
                "label": label_list,
                "idx": idx_list,
                "w": w_list,
                "edge_type": edge_type,
            }
        ),
        feat_l,
        node_ids,
    )


"""
functions for tkgl-wikidata dataset
"""


def csv_to_wikidata(
    fname: str,
) -> pd.DataFrame:
    r"""
    used by tkgl-wikidata and tkgl-smallpedia
    convert the raw .csv data to pandas dataframe and numpy array
    input .csv file format should be: timestamp, head, tail, relation type
    Args:
        fname: the path to the raw data
    """
    feat_size = 1
    num_lines = sum(1 for line in open(fname)) - 1
    print("number of lines counted", num_lines)
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    label_list = np.zeros(num_lines)
    edge_type = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)
    node_ids = {}
    edge_type_ids = {}
    unique_id = 0
    et_id = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # timestamp, head, tail, relation type
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                ts = int(row[0])  # converted to year already
                src = row[1]
                dst = row[2]
                relation = row[3]
                if src not in node_ids:
                    node_ids[src] = unique_id
                    unique_id += 1
                if dst not in node_ids:
                    node_ids[dst] = unique_id
                    unique_id += 1
                if relation not in edge_type_ids:
                    edge_type_ids[relation] = et_id
                    et_id += 1
                u = node_ids[src]
                i = node_ids[dst]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = float(1)
                edge_type[idx - 1] = edge_type_ids[relation]
                idx += 1
    return (
        pd.DataFrame(
            {
                "u": u_list,
                "i": i_list,
                "ts": ts_list,
                "label": label_list,
                "idx": idx_list,
                "w": w_list,
                "edge_type": edge_type,
            }
        ),
        feat_l,
        node_ids,
    )


def csv_to_staticdata(
    fname: str,
    node_ids: dict,
) -> pd.DataFrame:
    r"""
    used by tkgl-wikidata and tkgl-smallpedia
    convert the raw .csv data to pandas dataframe and numpy array for static knowledge edges
    input .csv file format should be: head, tail, relation type
    Args:
        fname: the path to the raw data
        node_ids: dictionary of node names mapped to integer node ids
    """
    num_lines = sum(1 for line in open(fname)) - 1
    print("number of lines counted", num_lines)
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    edge_type = np.zeros(num_lines)
    edge_type_ids = {}
    out_dict = {}

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # timestamp, head, tail, relation type
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                src = row[0]
                dst = row[1]
                relation = row[2]
                if src not in node_ids:
                    node_ids[src] = len(node_ids)
                if dst not in node_ids:
                    node_ids[dst] = len(node_ids)
                if relation not in edge_type_ids:
                    edge_type_ids[relation] = len(edge_type_ids)
                u = node_ids[src]
                i = node_ids[dst]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                edge_type[idx - 1] = edge_type_ids[relation]
                idx += 1

    out_dict["head"] = u_list
    out_dict["tail"] = i_list
    out_dict["edge_type"] = edge_type
    return out_dict, node_ids


"""
functions for tkgl-polecat, tkgl-icews dataset
"""


def csv_to_tkg_data(
    fname: str,
) -> pd.DataFrame:
    r"""
    used by tkgl-polecat
    convert the raw .csv data to pandas dataframe and numpy array
    input .csv file format should be: timestamp, head, tail, relation type
    Args:
        fname: the path to the raw data
    """
    feat_size = 1
    num_lines = sum(1 for line in open(fname)) - 1
    print("number of lines counted", num_lines)
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    label_list = np.zeros(num_lines)
    edge_type = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)
    node_ids = {}
    unique_id = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # timestamp, head, tail, relation type
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                ts = int(row[0])  # converted to UNIX timestamp already
                src = int(row[1])
                dst = int(row[2])
                relation = int(row[3])
                if src not in node_ids:
                    node_ids[src] = unique_id
                    unique_id += 1
                if dst not in node_ids:
                    node_ids[dst] = unique_id
                    unique_id += 1
                u = node_ids[src]
                i = node_ids[dst]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = float(1)
                edge_type[idx - 1] = relation
                idx += 1
    return (
        pd.DataFrame(
            {
                "u": u_list,
                "i": i_list,
                "ts": ts_list,
                "label": label_list,
                "idx": idx_list,
                "w": w_list,
                "edge_type": edge_type,
            }
        ),
        feat_l,
        node_ids,
    )


"""
functions for wikipedia dataset
---------------------------------------
"""


def load_edgelist_wiki(fname: str) -> pd.DataFrame:
    """
    loading wikipedia dataset into pandas dataframe
    similar processing to
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/jodie.html

    Parameters:
        fname: str, name of the input file
    Returns:
        df: a pandas dataframe containing the edgelist data
    """
    df = pd.read_csv(fname, skiprows=1, header=None)
    src = df.iloc[:, 0].values
    dst = df.iloc[:, 1].values
    dst += int(src.max()) + 1
    t = df.iloc[:, 2].values
    msg = df.iloc[:, 4:].values
    idx = np.arange(t.shape[0])
    w = np.ones(t.shape[0])

    return pd.DataFrame({"u": src, "i": dst, "ts": t, "idx": idx, "w": w}), msg, None


"""
functions for un_trade dataset
---------------------------------------
"""


def load_edgelist_trade(fname: str, label_size=255):
    """
    load the edgelist into pandas dataframe
    """
    feat_size = 1
    num_lines = sum(1 for line in open(fname)) - 1
    print("number of lines counted", num_lines)
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)
    # print("numpy allocated")
    node_ids = {}  # dictionary for node ids
    node_uid = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
            else:
                ts = int(row[0])
                u = row[1]
                v = row[2]
                w = float(row[3])
                if u not in node_ids:
                    node_ids[u] = node_uid
                    node_uid += 1

                if v not in node_ids:
                    node_ids[v] = node_uid
                    node_uid += 1

                u = node_ids[u]
                i = node_ids[v]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = w
                feat_l[idx - 1] = np.array([w])
                idx += 1

    return (
        pd.DataFrame(
            {"u": u_list, "i": i_list, "ts": ts_list, "idx": idx_list, "w": w_list}
        ),
        feat_l,
        node_ids,
    )


def load_trade_label_dict(
    fname: str,
    node_ids: dict,
) -> dict:
    """
    load node labels into a nested dictionary instead of pandas dataobject
    {ts: {node_id: label_vec}}
    Parameters:
        fname: str, name of the input file
        node_ids: dictionary of user names mapped to integer node ids
    Returns:
        node_label_dict: a nested dictionary of node labels
    """
    if not osp.exists(fname):
        raise FileNotFoundError(f"File not found at {fname}")

    label_size = len(node_ids)
    # label_vec = np.zeros(label_size)

    node_label_dict = {}  # {ts: {node_id: label_vec}}

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
            else:
                ts = int(row[0])
                u = node_ids[row[1]]
                v = node_ids[row[2]]
                weight = float(row[3])

                if ts not in node_label_dict:
                    node_label_dict[ts] = {u: np.zeros(label_size)}

                if u not in node_label_dict[ts]:
                    node_label_dict[ts][u] = np.zeros(label_size)

                node_label_dict[ts][u][v] = weight
                idx += 1
        return node_label_dict


"""
functions for tgbn-token
---------------------------------------
"""


def load_edgelist_token(
    fname: str,
    label_size: int = 1001,
) -> pd.DataFrame:
    """
    load the edgelist into pandas dataframe
    also outputs index for the user nodes and genre nodes
    Parameters:
        fname: str, name of the input file
        label_size: int, number of genres
    Returns:
        df: a pandas dataframe containing the edgelist data
    """
    feat_size = 2
    num_lines = sum(1 for line in open(fname)) - 1
    # print("number of lines counted", num_lines)
    print("there are ", num_lines, " lines in the raw data")
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    label_list = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)

    node_ids = {}
    rd_dict = {}
    node_uid = label_size  # node ids start after all the genres
    sr_uid = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # [timestamp,user_address,token_address,value,IsSender]
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
            else:
                ts = row[0]
                src = row[1]
                token = row[2]
                w = float(row[3])
                attr = float(row[4])
                if src not in node_ids:
                    node_ids[src] = node_uid
                    node_uid += 1
                if token not in rd_dict:
                    rd_dict[token] = sr_uid
                    sr_uid += 1
                u = node_ids[src]
                i = rd_dict[token]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = w
                feat_l[idx - 1] = np.array([w, attr])
                idx += 1

        return (
            pd.DataFrame(
                {
                    "u": u_list,
                    "i": i_list,
                    "ts": ts_list,
                    "label": label_list,
                    "idx": idx_list,
                    "w": w_list,
                }
            ),
            feat_l,
            node_ids,
            rd_dict,
        )


"""
functions for subreddits dataset
---------------------------------------
"""


def load_edgelist_sr(
    fname: str,
    label_size: int = 2221,
) -> pd.DataFrame:
    """
    load the edgelist into pandas dataframe
    also outputs index for the user nodes and genre nodes
    Parameters:
        fname: str, name of the input file
        label_size: int, number of genres
    Returns:
        df: a pandas dataframe containing the edgelist data
    """
    feat_size = 1  # 2
    num_lines = sum(1 for line in open(fname)) - 1
    # print("number of lines counted", num_lines)
    print("there are ", num_lines, " lines in the raw data")
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    label_list = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)

    node_ids = {}
    rd_dict = {}
    node_uid = label_size  # node ids start after all the genres
    sr_uid = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # ['ts', 'src', 'subreddit', 'num_words', 'score']
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
            else:
                ts = row[0]
                src = row[1]
                subreddit = row[2]
                # num_words = int(row[3])
                score = int(row[4])
                if src not in node_ids:
                    node_ids[src] = node_uid
                    node_uid += 1
                if subreddit not in rd_dict:
                    rd_dict[subreddit] = sr_uid
                    sr_uid += 1
                w = float(score)
                u = node_ids[src]
                i = rd_dict[subreddit]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = w
                feat_l[idx - 1] = np.array([w])
                idx += 1

        return (
            pd.DataFrame(
                {
                    "u": u_list,
                    "i": i_list,
                    "ts": ts_list,
                    "label": label_list,
                    "idx": idx_list,
                    "w": w_list,
                }
            ),
            feat_l,
            node_ids,
            rd_dict,
        )


def load_labels_sr(
    fname,
    node_ids,
    rd_dict,
):
    """
    load the node labels for subreddit dataset
    """
    if not osp.exists(fname):
        raise FileNotFoundError(f"File not found at {fname}")

    # day, user_idx, label_vec
    label_size = len(rd_dict)
    label_vec = np.zeros(label_size)
    ts_prev = 0
    prev_user = 0

    ts_list = []
    node_id_list = []
    y_list = []

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # ['ts', 'src', 'subreddit', 'num_words', 'score']
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
            else:
                user_id = node_ids[int(row[1])]
                ts = int(row[0])
                sr_id = int(rd_dict[row[2]])
                weight = float(row[3])
                if idx == 1:
                    ts_prev = ts
                    prev_user = user_id
                # the next day
                if ts != ts_prev:
                    ts_list.append(ts_prev)
                    node_id_list.append(prev_user)
                    y_list.append(label_vec)
                    label_vec = np.zeros(label_size)
                    ts_prev = ts
                    prev_user = user_id
                else:
                    label_vec[sr_id] = weight

                if user_id != prev_user:
                    ts_list.append(ts_prev)
                    node_id_list.append(prev_user)
                    y_list.append(label_vec)
                    prev_user = user_id
                    label_vec = np.zeros(label_size)
                idx += 1
        return pd.DataFrame({"ts": ts_list, "node_id": node_id_list, "y": y_list})


def load_label_dict(fname: str, node_ids: dict, rd_dict: dict) -> dict:
    """
    load node labels into a nested dictionary instead of pandas dataobject
    {ts: {node_id: label_vec}}
    Parameters:
        fname: str, name of the input file
        node_ids: dictionary of user names mapped to integer node ids
        rd_dict: dictionary of subreddit names mapped to integer node ids
    """
    if not osp.exists(fname):
        raise FileNotFoundError(f"File not found at {fname}")

    # day, user_idx, label_vec
    label_size = len(rd_dict)
    node_label_dict = {}  # {ts: {node_id: label_vec}}

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # ['ts', 'src', 'dst', 'w']
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
            else:
                u = node_ids[row[1]]
                ts = int(row[0])
                v = int(rd_dict[row[2]])
                weight = float(row[3])
                if ts not in node_label_dict:
                    node_label_dict[ts] = {u: np.zeros(label_size)}

                if u not in node_label_dict[ts]:
                    node_label_dict[ts][u] = np.zeros(label_size)

                node_label_dict[ts][u][v] = weight
                idx += 1
        return node_label_dict


"""
functions for redditcomments
-------------------------------------------
"""


def csv_to_pd_data_rc(
    fname: str,
) -> pd.DataFrame:
    r"""
    currently used by redditcomments dataset
    convert the raw .csv data to pandas dataframe and numpy array
    input .csv file format should be: timestamp, node u, node v, attributes
    Args:
        fname: the path to the raw data
    """
    feat_size = 2  # 1 for subreddit, 1 for num words
    num_lines = sum(1 for line in open(fname)) - 1
    # print("number of lines counted", num_lines)
    print("there are ", num_lines, " lines in the raw data")
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    label_list = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)
    node_ids = {}

    unique_id = 0
    max_words = 5000  # counted form statistics

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # ['ts', 'src', 'dst', 'subreddit', 'num_words', 'score']
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                ts = int(row[0])
                src = row[1]
                dst = row[2]
                num_words = int(row[3]) / max_words  # int number, normalize to [0,1]
                score = int(row[4])  # int number

                # reindexing node and subreddits
                if src not in node_ids:
                    node_ids[src] = unique_id
                    unique_id += 1
                if dst not in node_ids:
                    node_ids[dst] = unique_id
                    unique_id += 1
                w = float(score)
                u = node_ids[src]
                i = node_ids[dst]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = w
                feat_l[idx - 1] = np.array([num_words])
                idx += 1
    print("there are ", len(node_ids), " unique nodes")

    return (
        pd.DataFrame(
            {
                "u": u_list,
                "i": i_list,
                "ts": ts_list,
                "label": label_list,
                "idx": idx_list,
                "w": w_list,
            }
        ),
        feat_l,
        node_ids,
    )


"""
functions for stablecoin
-------------------------------------------
"""


def csv_to_pd_data_sc(
    fname: str,
) -> pd.DataFrame:
    r"""
    currently used by stablecoin dataset
    convert the raw .csv data to pandas dataframe and numpy array
    input .csv file format should be: timestamp, node u, node v, attributes
    Parameters:
        fname: the path to the raw data
    Returns:
        df: a pandas dataframe containing the edgelist data
        feat_l: a numpy array containing the node features
        node_ids: a dictionary mapping node id to integer
    """
    feat_size = 1
    num_lines = sum(1 for line in open(fname)) - 1
    print("number of lines counted", num_lines)
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    label_list = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)
    print("numpy allocated")
    node_ids = {}
    unique_id = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # time,src,dst,weight
        # 1648811421,0x27cbb0e6885ccb1db2dab7c2314131c94795fbef,0x8426a27add8dca73548f012d92c7f8f4bbd42a3e,800.0
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                ts = int(row[0])
                src = row[1]
                dst = row[2]

                if src not in node_ids:
                    node_ids[src] = unique_id
                    unique_id += 1
                if dst not in node_ids:
                    node_ids[dst] = unique_id
                    unique_id += 1

                w = float(row[3])
                if w == 0:
                    w = 1

                u = node_ids[src]
                i = node_ids[dst]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = w
                feat_l[idx - 1] = np.zeros(feat_size)
                idx += 1

    #! normalize by log 2 for stablecoin
    w_list = np.log2(w_list)

    return (
        pd.DataFrame(
            {
                "u": u_list,
                "i": i_list,
                "ts": ts_list,
                "label": label_list,
                "idx": idx_list,
                "w": w_list,
            }
        ),
        feat_l,
        node_ids,
    )


"""
functions for opensky
-------------------------------------------
"""


def convert_str2int(
    in_str: str,
) -> np.ndarray:
    """
    convert strings to vectors of integers based on individual character
    each letter is converted as follows, a=10, b=11
    numbers are still int
    Parameters:
        in_str: an input string to parse
    Returns:
        out: a numpy integer array
    """
    out = []
    for element in in_str:
        if element.isnumeric():
            out.append(element)
        elif element == "!":
            out.append(-1)
        else:
            out.append(ord(element.upper()) - 44 + 9)
    out = np.array(out, dtype=np.float32)
    return out


def csv_to_pd_data(
    fname: str,
) -> pd.DataFrame:
    r"""
    currently used by tgbl-flight dataset
    convert the raw .csv data to pandas dataframe and numpy array
    input .csv file format should be: timestamp, node u, node v, attributes
    Args:
        fname: the path to the raw data
    """
    feat_size = 16
    num_lines = sum(1 for line in open(fname)) - 1
    print("number of lines counted", num_lines)
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    label_list = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)
    print("numpy allocated")
    node_ids = {}
    unique_id = 0
    ts_format = None

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        #'day','src','dst','callsign','typecode'
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                ts = row[0]
                if ts_format is None:
                    if ts.isdigit():
                        ts_format = True
                    else:
                        ts_format = False

                if ts_format:
                    ts = float(int(ts))  # unix timestamp already
                else:
                    # convert to unix timestamp
                    TIME_FORMAT = "%Y-%m-%d"
                    date_cur = datetime.datetime.strptime(ts, TIME_FORMAT)
                    ts = float(date_cur.timestamp())
                    # TIME_FORMAT = "%Y-%m-%d" # 2019-01-01
                    # date_cur  = date.fromisoformat(ts)
                    # dt = datetime.datetime.combine(date_cur, datetime.datetime.min.time())
                    # dt = dt.replace(tzinfo=datetime.timezone.edt)
                    # ts = float(dt.timestamp())

                src = row[1]
                dst = row[2]

                # 'callsign' has max size 8, can be 4, 5, 6, or 7
                # 'typecode' has max size 8
                # use ! as padding

                # pad row[3] to size 7
                if len(row[3]) == 0:
                    row[3] = "!!!!!!!!"
                while len(row[3]) < 8:
                    row[3] += "!"

                # pad row[4] to size 4
                if len(row[4]) == 0:
                    row[4] = "!!!!!!!!"
                while len(row[4]) < 8:
                    row[4] += "!"
                if len(row[4]) > 8:
                    row[4] = "!!!!!!!!"

                feat_str = row[3] + row[4]

                if src not in node_ids:
                    node_ids[src] = unique_id
                    unique_id += 1
                if dst not in node_ids:
                    node_ids[dst] = unique_id
                    unique_id += 1
                u = node_ids[src]
                i = node_ids[dst]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = float(1)
                feat_l[idx - 1] = convert_str2int(feat_str)
                idx += 1
    return (
        pd.DataFrame(
            {
                "u": u_list,
                "i": i_list,
                "ts": ts_list,
                "label": label_list,
                "idx": idx_list,
                "w": w_list,
            }
        ),
        feat_l,
        node_ids,
    )


def csv_to_pd_data_ln(
    fname: str,
) -> pd.DataFrame:
    r"""
    currently used by tgbl-flight dataset
    convert the raw .csv data to pandas dataframe and numpy array
    input .csv file format should be: timestamp, node u, node v, attributes
    Args:
        fname: the path to the raw data
    """
    # Define the future time window (7 days in milliseconds)
    time_window = 15 * 24 * 60 * 60 * 1000

    df = pd.read_csv(fname, true_values=["True"], false_values=["False"])
    unique_nodes = pd.concat([df["src"], df["dst"]]).unique()
    node_ids = {node: idx for idx, node in enumerate(unique_nodes)}
    df["u"] = df["src"].map(node_ids)
    df["i"] = df["dst"].map(node_ids)

    label_map = {"OPEN": 0, "FORCED": 1, "PENALTY": 1, "MUTUAL": 2}
    # df["label"] = df["closing_info"].map(label_map)

    df["status"] = (df["channel_status"] == "OPEN").astype(int)
    df["w"] = df["capacity"].astype(float)
    df["idx"] = np.arange(1, len(df) + 1)

    # Compute the final label from the future:
    # First, select only the rows that correspond to a closing event.
    df_closing = df[df["closing_info"] != "OPEN"][["src", "dst", "gossip_ts", "closing_info"]].copy()
    df_closing = df_closing.sort_values("gossip_ts")
    # Use merge_asof to join each row with the first future closing event (if any) within the time_window.
    # Note: merge_asof matches on keys that are >= the left's key. (If needed, you can add a tiny epsilon to ensure strictly >.)
    df_merged = pd.merge_asof(
        df,
        df_closing,
        on="gossip_ts",
        by=["src", "dst"],
        direction="forward",
        tolerance=time_window,
        suffixes=("", "_future")
    )
    # Map the future closing event to a label (if exists); otherwise, default to OPEN (0).
    df_merged["final_label"] = df_merged["closing_info_future"].map(label_map).fillna(0).astype(int)

    feat_cols = [
        "status",
        "src_implementation",
        "dst_implementation",
        "capacity",
        "transaction_vout",
        "src_time_lock_delta",
        "dst_time_lock_delta",
        "src_min_htlc",
        "dst_min_htlc",
        "src_fee_base_msat",
        "dst_fee_base_msat",
        "src_fee_rate_milli_msat",
        "dst_fee_rate_milli_msat",
        "src_disabled",
        "dst_disabled",
        "src_last_update",
        "dst_last_update",
        "ts",
        "block_avg_fee_rate",
        "height",
    ]

    feat_l = df_merged[feat_cols].astype(float).to_numpy()
    result_df = df_merged[["u", "i", "gossip_ts", "final_label", "status", "idx", "w"]].rename(
        columns={"gossip_ts": "ts", "final_label": "label"}
    )
    return result_df, feat_l, node_ids


def process_node_feat(
    fname: str,
    node_ids,
):
    """
    1. need to have the same node id as csv_to_pd_data
    2. process the various node features into a vector
    3. return a numpy array of node features with index corresponding to node id

    airport_code,type,continent,iso_region,longitude,latitude
    type: onehot encoding
    continent: onehot encoding
    iso_region: alphabet encoding same as edge feat
    longitude: float divide by 180
    latitude: float divide by 90
    """
    feat_size = 20
    node_feat = np.zeros((len(node_ids), feat_size))
    type_dict = {}
    type_idx = 0
    continent_dict = {}
    cont_idx = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # airport_code,type,continent,iso_region,longitude,latitude
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                code = row[0]
                if code not in node_ids:
                    continue
                else:
                    node_id = node_ids[code]
                    airport_type = row[1]
                    if airport_type not in type_dict:
                        type_dict[airport_type] = type_idx
                        type_idx += 1
                    continent = row[2]
                    if continent not in continent_dict:
                        continent_dict[continent] = cont_idx
                        cont_idx += 1

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # airport_code,type,continent,iso_region,longitude,latitude
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                code = row[0]
                if code not in node_ids:
                    continue
                else:
                    node_id = node_ids[code]
                    airport_type = type_dict[row[1]]
                    type_vec = np.zeros(type_idx)
                    type_vec[airport_type] = 1
                    continent = continent_dict[row[2]]
                    cont_vec = np.zeros(cont_idx)
                    cont_vec[continent] = 1
                    while len(row[3]) < 7:
                        row[3] += "!"
                    iso_region = convert_str2int(row[3])  # numpy float array
                    lng = float(row[4])
                    lat = float(row[5])
                    coor_vec = np.array([lng, lat])
                    final = np.concatenate(
                        (type_vec, cont_vec, iso_region, coor_vec), axis=0
                    )
                    node_feat[node_id] = final
    return node_feat


"""
functions for un trade
-------------------------------------------
"""


#! these are helper functions
# TODO cleaning the un trade csv with countries with comma in the name, to remove this function
def clean_rows(
    fname: str,
    outname: str,
):
    r"""
    clean the rows with comma in the name
    args:
        fname: the path to the raw data
        outname: the path to the cleaned data
    """

    outf = open(outname, "w")

    with open(fname) as f:
        s = next(f)
        outf.write(s)
        for idx, line in enumerate(f):
            strs = ["China, Taiwan Province of", "China, mainland"]
            for str in strs:
                line = line.replace(
                    "China, Taiwan Province of", "Taiwan Province of China"
                )
                line = line.replace("China, mainland", "China mainland")
                line = line.replace("China, Hong Kong SAR", "China Hong Kong SAR")
                line = line.replace("China, Macao SAR", "China Macao SAR")
                line = line.replace(
                    "Saint Helena, Ascension and Tristan da Cunha",
                    "Saint Helena Ascension and Tristan da Cunha",
                )

            e = line.strip().split(",")
            if len(e) > 4:
                print(e)
                raise ValueError("line has more than 4 elements")
            outf.write(line)

    outf.close()


"""
functions for last fm genre
-------------------------------------------
"""


def load_edgelist_datetime(fname, label_size=514):
    """
    load the edgelist into a pandas dataframe
    use numpy array instead of list for faster processing
    assume all edges are already sorted by time
    convert all time unit to unix time

    time, user_id, genre, weight
    """
    feat_size = 1
    num_lines = sum(1 for line in open(fname)) - 1
    print("number of lines counted", num_lines)
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)
    # print("numpy allocated")
    node_ids = {}  # dictionary for node ids
    label_ids = {}  # dictionary for label ids
    node_uid = label_size  # node ids start after the genre nodes
    label_uid = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
            else:
                ts = int(row[0])
                user_id = row[1]
                genre = row[2]
                w = float(row[3])

                if user_id not in node_ids:
                    node_ids[user_id] = node_uid
                    node_uid += 1

                if genre not in label_ids:
                    label_ids[genre] = label_uid
                    if label_uid >= label_size:
                        print("id overlap, terminate")
                    label_uid += 1

                u = node_ids[user_id]
                i = label_ids[genre]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = w
                feat_l[idx - 1] = np.asarray([w])
                idx += 1

    return (
        pd.DataFrame(
            {"u": u_list, "i": i_list, "ts": ts_list, "idx": idx_list, "w": w_list}
        ),
        feat_l,
        node_ids,
        label_ids,
    )


def load_genre_list(fname):
    """
    load the list of genres
    """
    if not osp.exists(fname):
        raise FileNotFoundError(f"File not found at {fname}")

    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    genre_index = {}
    ctr = 0
    for i in range(1, len(lines)):
        vals = lines[i].split(",")
        genre = vals[0]
        if genre not in genre_index:
            genre_index[genre] = ctr
            ctr += 1
        else:
            raise ValueError("duplicate in genre_index")
    return genre_index


"""
functions for wikipedia and un_trade
-------------------------------------------
"""


def reindex(
    df: pd.DataFrame,
    bipartite: Optional[bool] = False,
):
    r"""
    reindex the nodes especially if the node ids are not integers
    Args:
        df: the pandas dataframe containing the graph
        bipartite: whether the graph is bipartite
    """
    new_df = df.copy()
    if bipartite:
        assert df.u.max() - df.u.min() + 1 == len(df.u.unique())
        assert df.i.max() - df.i.min() + 1 == len(df.i.unique())

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df


if __name__ == "__main__":
    # """
    # clean rows for un trade dataset
    # """
    # fname = "/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/un_trade/un_trade.csv"
    # outname = "/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/un_trade/un_trade_cleaned.csv"
    # clean_rows(fname, outname)
    print("hi")
