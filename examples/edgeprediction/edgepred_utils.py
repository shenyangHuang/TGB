
"""
Utility functions required for the task of link prediction

"""


import numpy as np
import torch



# general parameters 
K = 10  # for computing metrics@k



def gen_eval_set_for_batch(src, dst, min_dst_idx, max_dst_idx):
    """
    generate the evaluation set of edges for a batch of positive edges
    """
    pos_src = src.cpu().numpy()
    pos_dst = dst.cpu().numpy()

    batch_size = len(pos_src)

    all_dst = np.arange(min_dst_idx, max_dst_idx + 1)

    edges_per_node = {}
    # positive edges
    for pos_s, pos_d in zip(pos_src, pos_dst):
        if pos_s not in edges_per_node:
            edges_per_node[pos_s] = {'pos': [pos_d]}
        else:
            if pos_d not in edges_per_node[pos_s]['pos']:
                edges_per_node[pos_s]['pos'].append(pos_d)

    # negative edges
    for pos_s in edges_per_node:
        edges_per_node[pos_s]['neg'] = [neg_dst for neg_dst in all_dst if neg_dst not in edges_per_node[pos_s]['pos']]

    return edges_per_node

def eval_hits(y_pred_pos, y_pred_neg, type_info, K):
    '''
        source: https://github.com/snap-stanford/ogb/blob/d5c11d91c9e1c22ed090a2e0bbda3fe357de66e7/ogb/linkproppred/evaluate.py#L214
        compute Hits@K
        For each positive target node, the negative target nodes are the same.
        y_pred_neg is an array.
        rank y_pred_pos[i] against y_pred_neg for each i
    '''

    if len(y_pred_neg) < K:
        return {'hits@{}'.format(K): 1.}

    if type_info == 'torch':
        kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
        hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

    # type_info is numpy
    else:
        kth_score_in_negative_edges = np.sort(y_pred_neg)[-K]
        hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

    return hitsK


def metric_at_k_score(y_true, y_pred_proba, k=K, pos_label=1):
    """
        reference: 
            - https://subscription.packtpub.com/book/data/9781838826048/11/ch11lvl1sec70/calculating-the-precision-at-k
            - https://insidelearningmachines.com/precisionk_and_recallk/
    """

    topk = [
        y_true_ == pos_label 
        for y_true_, y_pred_proba_ 
        in sorted(
            zip(y_true, y_pred_proba), 
            key=lambda y: y[1], 
            reverse=True
        )[:k]
    ]
    precision_at_k = sum(topk) / len(topk) 
    recall_at_k = sum(topk) / sum(y_true == pos_label)
    
    if precision_at_k + recall_at_k != 0:
        f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
    else:
        f1_at_k = 0
    return precision_at_k, recall_at_k, f1_at_k