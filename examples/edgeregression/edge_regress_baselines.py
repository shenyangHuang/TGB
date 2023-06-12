"""
EdgeRegress: A simple baseline for the dynamic edge regression task

Date:
    - March 08, 2023
"""
import numpy as np
import pandas as pd
import time
import math
from sklearn.metrics import *


class PersistentForcaster:
    def __init__(self, train_data, val_data, memory_update_enable=False):
        self.model_name = "PersistentForcaster"
        self.memory = {}
        self.unseen_edge_weight = 0
        self.memory_update_enable = memory_update_enable
        # update memory with the train edges
        self.update_memory_by_edges(train_data.src, train_data.dst, train_data.y)
        # update the memory with the validation edges
        self.update_memory_by_edges(val_data.src, val_data.dst, val_data.dst)

    def update_memory_by_edges(self, sources, destinations, edge_weights):
        """
        Update the memory with the given edges
        The memory always has the most recent weights stored for each edge
        """
        for src, dst, weight in zip(sources, destinations, edge_weights):
            self.memory[(src, dst)] = weight

    def compute_edge_weights(
        self, sources, destinations, pos_e=False, pos_edge_weights=None
    ):
        """
        predict the edge weights based on the information stored in the memory
        """
        pred_value = []
        for idx, (src, dst) in enumerate(zip(sources, destinations)):
            if (src, dst) in self.memory:
                pred_value.append(self.memory[(src, dst)])
            else:
                pred_value.append(self.unseen_edge_weight)

            # update the memory with the recently observed positive edges if memory update is enabled
            if self.memory_update_enable and pos_e:
                self.memory[(src, dst)] = pos_edge_weights[idx]

        return np.array(pred_value)


class HistoricalMeanTeller:
    def __init__(self, train_data, val_data, memory_update_enable=False):
        self.model_name = "HistoricalMeanTeller"
        self.memory = {}
        self.unseen_edge_weight = 0
        self.memory_update_enable = memory_update_enable
        # update memory with the train edges
        self.update_memory_by_edges(train_data.src, train_data.dst, train_data.y)
        # update the memory with the validation edges
        self.update_memory_by_edges(val_data.src, val_data.dst, val_data.y)

    def update_memory_by_edges(self, sources, destinations, edge_weights):
        """
        Update the memory with the given edges
        """
        for src, dst, weight in zip(sources, destinations, edge_weights):
            if (src, dst) in self.memory:
                self.memory[(src, dst)].append(weight)
            else:
                self.memory[(src, dst)] = [weight]

    def compute_edge_weights(
        self, sources, destinations, pos_e=False, pos_edge_weights=None
    ):
        """
        predict the edge weights based on the information stored in the memory
        """
        pred_value = []
        for idx, (src, dst) in enumerate(zip(sources, destinations)):
            if (src, dst) in self.memory:
                pred_value.append(np.mean(self.memory[(src, dst)]))
            else:
                pred_value.append(self.unseen_edge_weight)

            # update the memory with the recently observed positive edges if memory update is enabled
            if self.memory_update_enable and pos_e:
                if (src, dst) in self.memory:
                    self.memory[(src, dst)].append(pos_edge_weights[idx])
                else:
                    self.memory[(src, dst)] = [pos_edge_weights[idx]]

        return np.array(pred_value)


class AbsoluteMeanTeller:
    def __init__(self, train_data, val_data, memory_update_enable=False):
        self.model_name = "AbsoluteMeanTeller"
        self.memory = []
        self.unseen_edge_weight = 0
        self.memory_update_enable = memory_update_enable
        # update memory with the train edges
        self.update_memory_by_edges(train_data.src, train_data.dst, train_data.y)
        # update the memory with the validation edges
        self.update_memory_by_edges(val_data.src, val_data.dst, val_data.y)

    def update_memory_by_edges(self, sources, destinations, edge_weights):
        """
        Update the memory with the given edges
        """
        for src, dst, weight in zip(sources, destinations, edge_weights):
            self.memory.append(weight)

    def compute_edge_weights(
        self, sources, destinations, pos_e=False, pos_edge_weights=None
    ):
        """
        predict the edge weights based on the information stored in the memory
        """
        pred_value = []
        absolute_mean_value = np.mean(self.memory)
        for idx, (src, dst) in enumerate(zip(sources, destinations)):
            pred_value.append(absolute_mean_value)

            # update the memory with the recently observed positive edges if memory update is enabled
            if self.memory_update_enable and pos_e:
                self.memory.append(weight)

        return np.array(pred_value)


class SnapshotMeanTeller:
    def __init__(self, data):
        self.model_name = "SnapshotMeanTeller"
        self.memory = {}
        self.unseen_edge_weight = 0
        # generate the memory content with the full data
        self.generate_memory(data)

    def generate_memory(self, data):
        """
        Update the memory with the given edges
        """
        unique_timestamps = np.unique(data.t)
        data.y = np.array(data.y.float())
        for ts in unique_timestamps:
            ts_idx = [int(idx) for idx, t in enumerate(data.t) if t == ts]
            self.memory[ts] = np.mean(data.y[ts_idx])

    def compute_edge_weights(self, e_timestamps):
        """
        predict the edge weights based on the information stored in the memory
        """
        pred_value = []
        for ts in e_timestamps:
            if ts in self.memory:
                pred_value.append(self.memory[ts])
            else:
                pred_value.append(self.unseen_edge_weight)

        return np.array(pred_value)


def compute_perf_metrics(y_true, y_pred_score):
    """
    compute extra performance measures
    """
    perf_dict = {
        "MSE": mean_squared_error(y_true, y_pred_score),  # Lower is better
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred_score)),  # Lower is better
        #  'KL_div': sum(kl_div(y_true, y_pred_score)),  # Lower is better
        #  'PCC': stats.pearsonr(y_true, y_pred_score).statistic,  # Higher is better
    }

    return perf_dict


def eval_link_reg_only_pos_e_baseline(model, data):
    """
    Evaluate the performance of edge regression ONLY for the positive edges
    """
    total_start_time = time.time()

    if model.model_name == "SnapshotMeanTeller":
        pos_prob = model.compute_edge_weights(data.ts)
    else:
        # memoy is ONLY updated if model.memory_update_enable = True
        pos_prob = model.compute_edge_weights(
            data.src, data.dst, pos_e=True, pos_edge_weights=data.y
        )

    perf_dict = compute_perf_metrics(data.y, pos_prob)

    print(
        f"INFO: Total one snapshot evaluation elapsed time (in seconds): {time.time() - total_start_time}"
    )

    return perf_dict
