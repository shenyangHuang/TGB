"""
utils for TKG Forecasting

"""
import numpy as np
import pandas as pd
from collections import Counter
import json
import torch 
import logging
from operator import itemgetter
def match_body_relations(rule, edges, test_query_sub):
    """
    for rules of length 1
    Find quadruples that match the rule (starting from the test query subject)
    Find edges whose subject match the query subject and the relation matches
    the relation in the rule body. 
    Memory-efficient implementation.

    modified from Tlogic rule_application.py https://github.com/liu-yushan/TLogic/blob/main/mycode/rule_application.py
    shortened because we only have rules of length one 

    Parameters:
        rule (dict): rule from rules_dict
        edges (dict): edges for rule application
        test_query_sub (int): test query subject
    Returns:
        walk_edges (list of np.ndarrays): edges that could constitute rule walks
    """

    rels = rule["body_rels"]
    # Match query subject and first body relation
    try:
        rel_edges = edges[rels[0]]
        mask = rel_edges[:, 0] == test_query_sub
        new_edges = rel_edges[mask]
        walk_edges = [np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))]  # [sub, obj, ts]

    except KeyError:
        walk_edges = [[]]

    return walk_edges #subject object timestamp


def score_delta(cands_ts, test_query_ts, lmbda):
    """ deta function to score a given candidate based on its distance to current timestep and based on param lambda
    Parameters:
        cands_ts (int): timestep of candidate(s)
        test_query_ts (int): timestep of current test quadruple
        lmbda (float): param to specify how steep decay is
    Returns:
        score (float): score for a given candicate
    """
    score = pow(2, lmbda * (cands_ts - test_query_ts))


    return score


def get_window_edges(all_data, test_query_ts, window=-2, first_test_query_ts=0): #modified eval_paper_authors: added first_test_query_ts for validation set usage
    """
    modified from Tlogic rule_application.py https://github.com/liu-yushan/TLogic/blob/main/mycode/rule_application.py
    introduce window -2 

    Get the edges in the data (for rule application) that occur in the specified time window.
    If window is 0, all edges before the test query timestamp are included.
    If window is -2, all edges from train and validation set are used. as long as they are < first_test_query_ts
    If window is an integer n > 0, all edges within n timestamps before the test query
    timestamp are included.

    Parameters:
        all_data (np.ndarray): complete dataset (train/valid/test)
        test_query_ts (np.ndarray): test query timestamp
        window (int): time window used for rule application
        first_test_query_ts (int): smallest timestamp from test set (eval_paper_authors)

    Returns:
        window_edges (dict): edges in the window for rule application
    """

    if window > 0:
        mask = (all_data[:, 3] < test_query_ts) * (
            all_data[:, 3] >= test_query_ts - window 
        )
        window_edges = quads_per_rel(all_data[mask]) # quadruples per relation that fullfill the time constraints 
    elif window == 0:
        mask = all_data[:, 3] < test_query_ts #!!! 
        window_edges = quads_per_rel(all_data[mask]) 
    elif window == -2: #modified eval_paper_authors: added this option
        mask = all_data[:, 3] < first_test_query_ts # all edges at timestep smaller then the test queries. meaning all from train and valid set
        window_edges = quads_per_rel(all_data[mask])  
    elif window == -200: #modified eval_paper_authors: added this option
        abswindow = 200
        mask = (all_data[:, 3] < first_test_query_ts) * (
            all_data[:, 3] >= first_test_query_ts - abswindow  # all edges at timestep smaller than the test queries - 200
        )
        window_edges = quads_per_rel(all_data[mask])
    all_data_ts = all_data[mask]
    return window_edges, all_data_ts


def quads_per_rel(quads):
    """
    modified from Tlogic rule_application.py https://github.com/liu-yushan/TLogic/blob/main/mycode/rule_application.py
    Store all edges for each relation.

    Parameters:
        quads (np.ndarray): indices of quadruples

    Returns:
        edges (dict): edges for each relation
    """

    edges = dict()
    relations = list(set(quads[:, 1]))
    for rel in relations:
        edges[rel] = quads[quads[:, 1] == rel]

    return edges


    
def compute_mrr(scores_dict, test_data, timesteps_test):
    """ compute time-aware filtered MRR and hits
    scores_dict: dicht with keys: query in string format, e.g. '13_0_xxx0_305' -> in this case xxx0 is to be predicted.
    values: list with two elements. element 0: tensor of shape num_nodes,1 -> scores for each node. elemen 1: array with 
    query test triple. for subject prediction we have inverse relation ids
    """
    # sort the data based on timesteps
    sorted_indices = np.argsort(test_data[:, 3])
    sorted_test_data = test_data[sorted_indices]

    all_ans_list_test, test_data_snaps = load_all_answers_for_time_filter(sorted_test_data)
    assert len(all_ans_list_test) == len(timesteps_test)
    scores_t_filter = compute_testscores(timesteps_test, test_data_snaps, scores_dict, all_ans_list_test)

    return scores_t_filter

    

#### compute MRRs
## all methods below taken or modified from https://github.com/Lee-zix/RE-GCN
# Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang and Xueqi Cheng. 
# Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning. SIGIR 2021.
def load_all_answers_for_filter(total_data):
    # taken or modified from https://github.com/Lee-zix/RE-GCN
    # from RE-GCN
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    all_ans = {}
    try: total_data.shape[1]
    except: total_data = np.expand_dims(total_data, axis=0) # if we only have one triple

    for line in total_data:
        s, r, o = line[: 3]
        add_node(s, o, r, all_ans) # both directions

    return all_ans

def load_all_answers_for_time_filter(total_data):
    # taken or modified from https://github.com/Lee-zix/RE-GCN
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap)
        all_ans_list.append(all_ans_t)

    return all_ans_list, all_snap

def add_node(e1, e2, r, d):
    # taken or modified from https://github.com/Lee-zix/RE-GCN
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)

def split_by_time(data):
    # taken or modified from https://github.com/Lee-zix/RE-GCN
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        if latest_t != t:  
            # show snapshot
            latest_t = t
            if len(snapshot):  # appends in the list lazily i.e. when new timestamp is observed
                # load the previous batch and empty the cache
                snapshot_list.append(np.array(snapshot).copy().squeeze())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy().squeeze())
        snapshots_num += 1

    return snapshot_list

def compute_testscores(timesteps_test, test_data, scores_dict, all_ans_list_test):
    # taken or modified from https://github.com/Lee-zix/RE-GCN
    ranks_t_filter, mrr_t_filter_list= [], []
    # assert len(timesteps_test) == len(test_data) == len(list(scores_dict.keys())) == len(all_ans_list_test)

    timesteps_idx = list(range(len(timesteps_test)))  # rename to match the standard of all_and_list_test
    for time_idx, test_triple, test_ts in zip(timesteps_idx, test_data, timesteps_test):
        try: test_triple.shape[1]
        except: test_triple= np.expand_dims(test_triple, axis=0) # if we only have one triple
        b =  np.ones((len(test_triple),1), dtype=int)*(test_ts) 
        test_quad = np.concatenate([test_triple, b], axis=1)
        string_array = ['[' + ', '.join(map(str, inner_array)) + ']' for inner_array in test_quad]
        if len(string_array) ==1:
            final_score = scores_dict[string_array[0]].unsqueeze(0)
        else:
            final_score = torch.stack(itemgetter(*string_array)(scores_dict))
        mrr_t_filter_snap,rank_t_filter = get_total_rank(
            test_triple, final_score,
            all_ans_list_test[time_idx],
            eval_bz=300)
        ranks_t_filter.append(rank_t_filter)
        mrr_t_filter_list.append(mrr_t_filter_snap)

    mode = 'valid'
    scores_t_filter = stat_ranks(ranks_t_filter, "Entity TimeAware Prediction Filter", mode, mrr_t_filter_list) 
    return scores_t_filter


def get_total_rank(test_triples, score, all_ans, eval_bz=1000):
    '''
    :param test_triples: triples with inverse relationship.
    :param score:
    :param all_ans: dict with [s,o]:rel:[o,s] or [s,o]:[o,s]:rel per timestamp.
    :param all_ans_static: dict with [s,o]:rel:[o,s] or [s,o]:[o,s]:rel, timestep independent
    :param eval_bz: evaluation batch size
    :param rel_predict: if 1 predicts relations/link prediction otherwise entity prediction.
    :return:
    '''
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_t_rank = []
    
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = torch.tensor(test_triples[batch_start:batch_end, :], device = score.device)
        score_batch = score[batch_start:batch_end, :]
        target = test_triples[batch_start:batch_end, 2]
        # time aware filter

        filter_score_batch_t = filter_score(triples_batch, score_batch, all_ans)
        filter_t_rank.append(sort_and_rank(filter_score_batch_t, target))
    filter_t_rank = torch.cat(filter_t_rank)
    filter_t_rank += 1
    filter_t_mrr = torch.mean(1.0 / filter_t_rank.float())
    return filter_t_mrr.item(), filter_t_rank

def filter_score(test_triples, score, all_ans):
    # taken or modified from https://github.com/Lee-zix/RE-GCN
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score

def sort_and_rank(score, target): # in case of ties: random rank selection
    # taken or modified from https://github.com/Lee-zix/RE-GCN
    score = score*np.max([10000, 10000*score.max()])
    random_values = torch.rand_like(score)
    score = score + random_values   # to add some randomness in case of ties. the random values are significantly 
                                    # smaller to ensure that we only permute for ties
    _, indices = torch.sort(score, dim=1, descending=True) # with default: stable=False; pytorch docu: 
            #"If stable is True then the sorting routine becomes stable, preserving the order of equivalent elements."
    target = torch.tensor(target)
    indices = torch.nonzero(indices == target.view(-1, 1), as_tuple=False)
    indices = indices[:, 1].view(-1)
    return indices


def stat_ranks(rank_list, method, mode, mrr_snapshot_list):
    # taken or modified from https://github.com/Lee-zix/RE-GCN
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)
    mr = torch.mean(total_rank.float())
    mrr = torch.mean(1.0 / total_rank.float())
    # print("MR ({}): {:.6f}".format(method, mr.item()))
    # print("MRR ({}): {:.6f}".format(method, mrr.item()))

    # if mode == 'test':
        # logging.debug("MR ({}): {:.6f}".format(method, mr.item()))
        # logging.debug("MRR ({}): {:.6f}".format(method, mrr.item()))
        # logging.debug("MRR over time ({}): {:.6f}".format(method, mrr_snapshot_list))
    hit_scores = []
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        # print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
        if mode == 'test':
            logging.debug("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
            hit_scores.append(avg_count.item())
    return (mr.item(), mrr.item(), hit_scores, mrr_snapshot_list)


