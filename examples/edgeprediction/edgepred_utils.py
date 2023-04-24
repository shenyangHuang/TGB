
"""
Utility functions required for the task of link prediction

"""


import numpy as np
import torch
import random
import os


# general parameters 
K = 10  # for computing metrics@k


def set_random_seed(seed):
  """
  set random seed
  """
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)


def gen_eval_set_for_batch_multi_pos(src, dst, min_dst_idx, max_dst_idx):
    """
    generate the evaluation set of edges for a batch of positive edges
    in this setting, if there are more than one positive edges with the same source node, both will be considered together
    and they are evaluated against all possible relevant negative edges
    NOTE: the key of the returned dictionary is the source node of the positive edges
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


def gen_eval_set_for_batch_one_pos(src, dst, min_dst_idx, max_dst_idx):
    """
    generate the evaluation set of edges for a batch of positive edges
    in this setting, each positive edge is evaluated agains all possible relevant negative edges
    thus, if there are multiple positive edges with the same source node, each one is evaluated individually
    NOTE: the key of the returned dictionary is a positive edge
    """
    pos_src = src.cpu().numpy()
    pos_dst = dst.cpu().numpy()

    batch_size = len(pos_src)

    all_dst = np.arange(min_dst_idx, max_dst_idx + 1)

    edges_per_pos_edge = {}
    # positive edges
    for pos_s, pos_d in zip(pos_src, pos_dst):
        if (pos_s, pos_d) not in edges_per_pos_edge:
            edges_per_pos_edge[(pos_s, pos_d)] = {'pos': [pos_d]}

    # negative edges
    for (pos_s, pos_d) in edges_per_pos_edge:
        edges_per_pos_edge[(pos_s, pos_d)]['neg'] = [neg_dst for neg_dst in all_dst if neg_dst != pos_d]

    return edges_per_pos_edge


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

    else:
        kth_score_in_negative_edges = np.sort(y_pred_neg)[-K]
        hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

    return hitsK


def eval_mrr(y_pred_pos, y_pred_neg, type_info, K):
    """
    source: https://github.com/snap-stanford/ogb/blob/d5c11d91c9e1c22ed090a2e0bbda3fe357de66e7/ogb/linkproppred/evaluate.py#L214
    """
    if type_info == 'torch':
        # calculate ranks
        y_pred_pos = y_pred_pos.view(-1, 1)
        # optimistic rank: "how many negatives have a larger score than the positive?"
        # ~> the positive is ranked first among those with equal score
        optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
        # pessimistic rank: "how many negatives have at least the positive score?"
        # ~> the positive is ranked last among those with equal score
        pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        # hits1_list = (ranking_list <= 1).to(torch.float)
        # hits3_list = (ranking_list <= 3).to(torch.float)
        # hits10_list = (ranking_list <= 10).to(torch.float)
        hitsK_list = (ranking_list <= K).to(torch.float)
        mrr_list = 1./ranking_list.to(torch.float)

        return {
                #  'hits@1_list': hits1_list,
                #  'hits@3_list': hits3_list,
                #  'hits@10_list': hits10_list,
                'hits@k': hitsK_list.mean(),
                'mrr_list': mrr_list.mean()
                }

    else:
        y_pred_pos = y_pred_pos.reshape(-1, 1)
        optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
        pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        # hits1_list = (ranking_list <= 1).astype(np.float32)
        # hits3_list = (ranking_list <= 3).astype(np.float32)
        # hits10_list = (ranking_list <= 10).astype(np.float32)
        hitsK_list = (ranking_list <= K).astype(np.float32)
        mrr_list = 1./ranking_list.astype(np.float32)

        return {
                # 'hits@1_list': hits1_list,
                # 'hits@3_list': hits3_list,
                # 'hits@10_list': hits10_list,
                'hits@k': hitsK_list.mean(),
                'mrr_list': mrr_list.mean()
                }



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
    metrics_at_k = {'precision@k': sum(topk) / len(topk) ,
                    'recall@k': sum(topk) / sum(y_true == pos_label),
                    }
    if metrics_at_k['precision@k'] + metrics_at_k['recall@k'] != 0:
        f1_at_k = 2 * (metrics_at_k['precision@k'] * metrics_at_k['recall@k']) / (metrics_at_k['precision@k'] + metrics_at_k['recall@k'])
    else:
        f1_at_k = 0
    metrics_at_k ['f1@k'] = f1_at_k

    return metrics_at_k