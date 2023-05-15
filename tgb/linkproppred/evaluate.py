"""
Evaluator Module for Link Prediction Task
"""

import numpy as np
from sklearn.metrics import *
import math


try:
    import torch
except ImportError:
    torch = None


class Evaluator(object):
    """Evaluator for Link Prediction Task"""

    def __init__(self, name: str):
        r"""
        Parameters:
            name: name of the dataset
        """
        self.name = name
        if self.name not in ["wikipedia"]:
            raise NotImplementedError("Dataset not supported")

    def eval_rnk_metrics(self, y_pred_pos, y_pred_neg, type_info, k):
        """
        reference:
            - https://github.com/snap-stanford/ogb/blob/d5c11d91c9e1c22ed090a2e0bbda3fe357de66e7/ogb/linkproppred/evaluate.py#L214
        """
        if type_info == "torch":
            # calculate ranks
            y_pred_pos = y_pred_pos.view(-1, 1)
            # optimistic rank: "how many negatives have a larger score than the positive?"
            # ~> the positive is ranked first among those with equal score
            optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
            # pessimistic rank: "how many negatives have at least the positive score?"
            # ~> the positive is ranked last among those with equal score
            pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
            ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
            hitsK_list = (ranking_list <= k).to(torch.float)
            mrr_list = 1.0 / ranking_list.to(torch.float)

            return {f"hits@{k}": hitsK_list.mean(), "mrr": mrr_list.mean()}

        else:
            y_pred_pos = y_pred_pos.reshape(-1, 1)
            optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
            pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
            ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
            hitsK_list = (ranking_list <= K).astype(np.float32)
            mrr_list = 1.0 / ranking_list.astype(np.float32)

            return {f"hits@{k}": hitsK_list.mean(), "mrr": mrr_list.mean()}
