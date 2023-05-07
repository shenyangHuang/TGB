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
        self.valid_metric_list = ['ap', 'auc', 'prec@k', 'rec@k', 'f1@k', 'hits@k', 'mrr']
        if self.name not in ["wikipedia"]:
            raise NotImplementedError("Dataset not supported")

    def _parse_and_check_input(self, input_dict):
        """
        check whether the input has the required format
        Parametrers:
            - input_dict: a dictionary containing "y_true", "y_pred", and "eval_metric"
            
            note: "eval_metric" specifies the evaluation metric used for evaluation (its a 'str')
        """

        if 'eval_metric' not in input_dict:
            raise RuntimeError("Missing key of eval_metric")

        eval_metric = input_dict['eval_metric'] 
        if eval_metric in self.valid_metric_list:
            if 'y_true' not in input_dict:
                raise RuntimeError('Missing key of y_true')
            if 'y_pred' not in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            # converting to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()
            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            # check type and shape
            if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
                raise RuntimeError("Arguments to Evaluator need to be either numpy ndarray or torch tensor!")

            if not y_true.shape == y_pred.shape:
                raise RuntimeError("Shape of y_true and y_pred must be the same!")

        else:
            print("ERROR: The evaluation metric should be in:", self.valid_metric_list)
            raise ValueError('Undefined eval metric %s ' % (eval_metric))

        return y_true, y_pred, eval_metric


    # def eval_cls(self, input_dict, verbose: bool=False):
    #     """
    #     evaluation for the link prediction task
    #     """
    #     y_true, y_pred, eval_metric = self._parse_and_check_input(input_dict)
    #     performace = self._compute_metrics_cls(y_true, y_pred, eval_metric)

    #     if verbose:
    #         print(f"INFO: Evaluation Results in terms of {input_dict['eval_metric']}: {performace: .4f}")
    #     return performace

    def eval_metrics_cls(self, y_true, y_pred, eval_metric: str):
        """
        compute the performance metrics for the given true labels and prediction probabilities
        Parameters:
            -y_true: actual true labels
            -y_pred: predicted probabilities
        """
        assert eval_metric in ['ap', 'auc'], 'Evaluation metric is undefined for this mode!'
        if eval_metric == 'ap':
            performace = average_precision_score(y_true, y_pred)
        elif eval_metric == 'auc':
            performace = roc_auc_score(y_true, y_pred)
        return performace

    def eval_metrics_cls_rnk(self, y_true, y_pred, k, pos_label=1):
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
        metrics_at_k = {'prec@k': sum(topk) / len(topk) ,
                        'rec@k': sum(topk) / sum(y_true == pos_label),
                        }
        if metrics_at_k['prec@k'] + metrics_at_k['rec@k'] != 0:
            f1_at_k = 2 * (metrics_at_k['prec@k'] * metrics_at_k['rec@k']) / (metrics_at_k['prec@k'] + metrics_at_k['rec@k'])
        else:
            f1_at_k = 0
        metrics_at_k ['f1@k'] = f1_at_k

        return metrics_at_k

    def eval_metrics_mrr_rnk(self, y_pred_pos, y_pred_neg, type_info, k):
        """
        reference:
            - https://github.com/snap-stanford/ogb/blob/d5c11d91c9e1c22ed090a2e0bbda3fe357de66e7/ogb/linkproppred/evaluate.py#L214
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
            hitsK_list = (ranking_list <= k).to(torch.float)
            mrr_list = 1./ranking_list.to(torch.float)

            return {
                    'hits@k': hitsK_list.mean(),
                    'mrr': mrr_list.mean()
                    }

        else:
            y_pred_pos = y_pred_pos.reshape(-1, 1)
            optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
            pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
            ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
            hitsK_list = (ranking_list <= K).astype(np.float32)
            mrr_list = 1./ranking_list.astype(np.float32)

            return {
                    'hits@k': hitsK_list.mean(),
                    'mrr': mrr_list.mean()
                    }

    