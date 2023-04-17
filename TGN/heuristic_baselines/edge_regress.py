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
    def __init__(self, train_data, val_data):
        self.model_name = 'PersistentForcaster'
        self.memory = {}
        self.unseen_edge_weight = 0
        # update memory with the train edges
        self.update_memory_by_edges(train_data.sources, train_data.destinations, train_data.true_y)
        # update the memory with the validation edges
        self.update_memory_by_edges(val_data.sources, val_data.destinations, val_data.true_y)

    
    def update_memory_by_edges(self, sources, destinations, edge_weights):
        """
        Update the memory with the given edges
        The memory always has the most recent weights stored for each edge
        """
        for src, dst, weight in zip(sources, destinations, edge_weights):
            self.memory[(src, dst)] = weight

    def compute_edge_weights(self, sources, destinations, pos_e=False, pos_edge_weights=None):
        """
        predict the edge weights based on the information stored in the memory
        """
        pred_value = []
        for idx, (src, dst) in enumerate(zip(sources, destinations)):
            if (src, dst) in self.memory:
                pred_value.append(self.memory[(src, dst)])
            else:
                pred_value.append(self.unseen_edge_weight)

            # update the memory with the recently observed positive edges
            if pos_e:
                self.memory[(src, dst)] = pos_edge_weights[idx]

        return np.array(pred_value)


class HistoricalMeanTeller:
    def __init__(self, train_data, val_data):
        self.model_name = 'HistoricalMeanTeller'
        self.memory = {}
        self.unseen_edge_weight = 0
        # update memory with the train edges
        self.update_memory_by_edges(train_data.sources, train_data.destinations, train_data.true_y)
        # update the memory with the validation edges
        self.update_memory_by_edges(val_data.sources, val_data.destinations, val_data.true_y)

    
    def update_memory_by_edges(self, sources, destinations, edge_weights):
        """
        Update the memory with the given edges
        """
        for src, dst, weight in zip(sources, destinations, edge_weights):
            if (src, dst) in self.memory:
                self.memory[(src, dst)].append(weight)
            else:
                self.memory[(src, dst)] = [weight]

    def compute_edge_weights(self, sources, destinations, pos_e=False, pos_edge_weights=None):
        """
        predict the edge weights based on the information stored in the memory
        """
        pred_value = []
        for idx, (src, dst) in enumerate(zip(sources, destinations)):
            if (src, dst) in self.memory:
                pred_value.append(np.mean(self.memory[(src, dst)]))
            else:
                pred_value.append(self.unseen_edge_weight)

            # update the memory with the recently observed positive edges
            if pos_e:
                if (src, dst) in self.memory:
                    self.memory[(src, dst)].append(pos_edge_weights[idx])
                else:
                    self.memory[(src, dst)] = [pos_edge_weights[idx]]

        return np.array(pred_value)


class AbsoluteMeanTeller:
    def __init__(self, train_data, val_data):
        self.model_name = 'AbsoluteMeanTeller'
        self.memory = []
        self.unseen_edge_weight = 0
        # update memory with the train edges
        self.update_memory_by_edges(train_data.sources, train_data.destinations, train_data.true_y)
        # update the memory with the validation edges
        self.update_memory_by_edges(val_data.sources, val_data.destinations, val_data.true_y)

    
    def update_memory_by_edges(self, sources, destinations, edge_weights):
        """
        Update the memory with the given edges
        """
        for src, dst, weight in zip(sources, destinations, edge_weights):
            self.memory.append(weight)
            

    def compute_edge_weights(self, sources, destinations, pos_e=False, pos_edge_weights=None):
        """
        predict the edge weights based on the information stored in the memory
        """
        pred_value = []
        absolute_mean_value = np.mean(self.memory)
        for idx, (src, dst) in enumerate(zip(sources, destinations)):
            pred_value.append(absolute_mean_value)

        return np.array(pred_value)


class SnapshotMeanTeller:
    def __init__(self, full_data):
        self.model_name = 'SnapshotMeanTeller'
        self.memory = {}
        self.unseen_edge_weight = 0
        # generate the memory content with the full data
        self.generate_memory(full_data)

    
    def generate_memory(self, full_data):
        """
        Update the memory with the given edges
        """
        unique_timestamps = np.unique(full_data.timestamps)
        for ts in unique_timestamps:
            ts_idx = full_data.timestamps == ts
            self.memory[ts] = np.mean(full_data.true_y[ts_idx])
            

    def compute_edge_weights(self, e_timestamps, pos_e=False, pos_edge_weights=None):
        """
        predict the edge weights based on the information stored in the memory
        """
        pred_value = []
        for ts in e_timestamps:
            pred_value.append(self.memory[ts])

        return np.array(pred_value)


def compute_perf_metrics(y_true, y_pred_score):
    """
    compute extra performance measures
    """
    perf_dict = {'MSE': mean_squared_error(y_true, y_pred_score), # Lower is better
                 'RMSE': math.sqrt(mean_squared_error(y_true, y_pred_score)),  # Lower is better
                #  'KL_div': sum(kl_div(y_true, y_pred_score)),  # Lower is better
                #  'PCC': stats.pearsonr(y_true, y_pred_score).statistic,  # Higher is better
                 }

    return perf_dict


def eval_link_reg_only_pos_e_baseline(model, data, logger=None, snapshot_ts=False, ):
    """
    Evaluate the performance of edge regression ONLY for the positive edges
    """
    total_start_time = time.time()

    if snapshot_ts:
        # This is the "SnapshotMeanTeller"; the only baseline that is different
        pos_prob = model.compute_edge_weights(data.timestamps, pos_e=False, pos_edge_weights=None)
    else:
        pos_prob = model.compute_edge_weights(data.sources, data.destinations, pos_e=True, pos_edge_weights=data.true_y)

    perf_dict = compute_perf_metrics(data.true_y, pos_prob)

    logger.info(f"INFO: Total one snapshot evaluation elapsed time (in seconds): {time.time() - total_start_time}")

    return perf_dict



def eval_link_reg_one_snapshot_EdgeRegress(model, snap_data, logger, stats_filename=None, batch_size=32):
    """
    Evaluate the link prediction task
    """
    total_start_time = time.time()
    if stats_filename is not None:
        logger.info("Test edge evaluation statistics are saved at {}".format(stats_filename))

    pos_data = snap_data['pos_e']
    neg_data = snap_data['neg_e']

    logger.info("INFO: Number of positive edges: {}".format(len(pos_data.sources)))
    logger.info("INFO: Number of negative edges: {}".format(len(neg_data.sources)))

    pred_score_agg, true_label_agg = [], []
    src_agg, dst_agg, ts_agg = [], [], []


    TEST_BATCH_SIZE = batch_size
    NUM_TEST_BATCH_POS = math.ceil(len(pos_data.sources) / TEST_BATCH_SIZE)

    NUM_TEST_BATCH_NEG = math.ceil(len(neg_data.sources) / TEST_BATCH_SIZE)
    NUM_NEG_BATCH_PER_POS_BATCH = math.ceil(NUM_TEST_BATCH_NEG / NUM_TEST_BATCH_POS)

    logger.info("INFO: NUM_TEST_BATCH_POS: {}".format(NUM_TEST_BATCH_POS))
    logger.info("INFO: NUM_NEG_BATCH_PER_POS_BATCH: {}".format(NUM_NEG_BATCH_PER_POS_BATCH))

    for p_b_idx in range(NUM_TEST_BATCH_POS):
        start_p_b_time = time.time()
        # ========== positive edges ==========
        pos_s_idx = p_b_idx * TEST_BATCH_SIZE
        pos_e_idx = min(len(pos_data.sources) - 1, pos_s_idx + TEST_BATCH_SIZE)          

        pos_src_batch = pos_data.sources[pos_s_idx: pos_e_idx]
        pos_dst_batch = pos_data.destinations[pos_s_idx: pos_e_idx]
        pos_ts_batch = pos_data.timestamps[pos_s_idx: pos_e_idx]
        pos_e_weight_batch = pos_data.true_y[pos_s_idx: pos_e_idx]

        if len(pos_src_batch) > 1:
            pos_prob = model.compute_edge_weights(pos_src_batch, pos_dst_batch, pos_e=True, pos_edge_weights=pos_e_weight_batch)

            if stats_filename is not None:
                src_agg.append(pos_src_batch)
                dst_agg.append(pos_dst_batch)
                ts_agg.append(pos_ts_batch)

                pred_score_agg.append(pos_prob)
                true_label_agg.append(pos_e_weight_batch)
            
        else:
            logger.info(f"DEBUG: no Positive edges in batch P-{p_b_idx}! [len(pos_src_batch): {len(pos_src_batch)}]")
        
        # logger.info(f"INFO: Positive batch {p_b_idx} evaluation elapsed time (in seconds): {time.time() - start_p_b_time}")

        # ========== negative edges ==========
        for n_b_idx in range(NUM_NEG_BATCH_PER_POS_BATCH):
            start_n_b_time = time.time()

            neg_s_idx = (p_b_idx * NUM_NEG_BATCH_PER_POS_BATCH + n_b_idx) * TEST_BATCH_SIZE
            neg_e_idx = min(len(neg_data.sources) - 1, neg_s_idx + TEST_BATCH_SIZE)

            neg_src_batch = neg_data.sources[neg_s_idx: neg_e_idx]
            neg_dst_batch = neg_data.destinations[neg_s_idx: neg_e_idx]
            neg_ts_batch = neg_data.timestamps[neg_s_idx: neg_e_idx]

            if len(neg_src_batch) > 1:
                neg_prob = model.compute_edge_weights(neg_src_batch, neg_dst_batch, pos_e=False, pos_edge_weights=None)
                neg_e_weight_batch = np.zeros(len(neg_src_batch))

                if stats_filename is not None:
                    src_agg.append(neg_src_batch)
                    dst_agg.append(neg_dst_batch)
                    ts_agg.append(neg_ts_batch)

                    pred_score_agg.append(neg_prob)
                    true_label_agg.append(neg_e_weight_batch)

            else:
                logger.info(f"DEBUG: no Negative edges in batch P-{p_b_idx}_N-{n_b_idx}!")
            
            # logger.info(f"INFO: Negative batch {n_b_idx} evaluation elapsed time (in seconds): {time.time() - start_n_b_time}")

    if stats_filename is not None:
        src_agg = np.concatenate(src_agg, axis=0)
        dst_agg = np.concatenate(dst_agg, axis=0)
        ts_agg = np.concatenate(ts_agg, axis=0)
        pred_score_agg = np.concatenate(pred_score_agg, axis=0)
        true_label_agg = np.concatenate(true_label_agg, axis=0)
        # save to file 
        np.save(stats_filename + '_src.npy', src_agg)
        np.save(stats_filename + '_dst.npy', dst_agg)
        np.save(stats_filename + '_ts.npy', ts_agg)
        np.save(stats_filename + '_pred_score.npy', pred_score_agg)
        np.save(stats_filename + '_label.npy', true_label_agg)

    logger.info(f"INFO: Total one snapshot evaluation elapsed time (in seconds): {time.time() - total_start_time}")

    return model




