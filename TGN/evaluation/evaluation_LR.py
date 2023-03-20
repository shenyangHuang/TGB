"""
Evaluation of Dynamic Link Regression

Date:
    - Mar. 5, 2023
"""

import time
import torch
import math
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd 
from tqdm import tqdm
from sklearn.metrics import *
from scipy.special import kl_div
from scipy import stats



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


def eval_link_reg_rnd_neg(model, sampler, data, logger, batch_size=32, n_neighbors=20):
    """
    Evaluate the link regression tasks when negative edges are randomly selected & 
    the number of positive and negative edges are equal.
    This function is mainly used for the validation set of the link regression task.
    """
    pred_score_list, true_label_list = [], []

    with torch.no_grad():
        model = model.eval()
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            if s_idx == e_idx:
                logger.info(f"DEBUG: Not enough Positive edges in the batch {k}! [len(data.sources[s_idx:e_idx]): {len(data.sources[s_idx:e_idx])}]")
                continue
        
            # ========== positive edges
            src_l_cut = data.sources[s_idx: e_idx]
            dst_l_cut = data.destinations[s_idx: e_idx]
            ts_l_cut = data.timestamps[s_idx: e_idx]
            e_l_cut = data.edge_idxs[s_idx: e_idx]
            pos_e_weight_l_cut = data.true_y[s_idx: e_idx]

            # ========== negative edges
            size = len(src_l_cut)
            neg_hist_ne_source, neg_hist_ne_dest, neg_rnd_source, neg_rnd_dest = sampler.sample(size, ts_l_cut[0], ts_l_cut[-1])

            src_l_fake = np.concatenate([neg_hist_ne_source, neg_rnd_source], axis=0)
            dst_l_fake = np.concatenate([neg_hist_ne_dest, neg_rnd_dest], axis=0)

            if sampler.neg_sample == 'haphaz_rnd':
                src_l_fake = src_l_cut

            neg_e_weights_l = np.zeros(size)

            # edge regression prediction
            if len(dst_l_fake) > 1:
                pos_pred_score = model.compute_edge_weights(src_l_cut, dst_l_cut, ts_l_cut, e_l_cut, True, n_neighbors)
                neg_pred_score = model.compute_edge_weights(src_l_fake, dst_l_fake, ts_l_cut, e_l_cut, False, n_neighbors)

                pred_scores = np.concatenate([(pos_pred_score).cpu().numpy(), (neg_pred_score).cpu().numpy()])
                true_labels = np.concatenate([pos_e_weight_l_cut, neg_e_weights_l])

                pred_score_list.append(pred_scores)
                true_label_list.append(true_labels)
            else:
                logger.info(f"DEBUG: Not enough Negative edges in the batch {k}! [len(dst_l_fake): {len(dst_l_fake)}]")

        pred_score_list = np.concatenate(pred_score_list, axis=0)
        true_label_list = np.concatenate(true_label_list, axis=0)
        perf_dict = compute_perf_metrics(true_label_list, pred_score_list)

    return perf_dict


def eval_link_reg_one_snapshot(model, snap_data, logger, stats_filename=None, batch_size=32, n_neighbors=20):
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

    with torch.no_grad():
        model = model.eval()

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
            pos_e_idx_batch = pos_data.edge_idxs[pos_s_idx: pos_e_idx]
            pos_e_weight_batch = pos_data.true_y[pos_s_idx: pos_e_idx]

            if len(pos_src_batch) > 1:
                pos_prob = model.compute_edge_weights(pos_src_batch, pos_dst_batch, pos_ts_batch, 
                                                        pos_e_idx_batch, True, n_neighbors)

                if stats_filename is not None:
                    src_agg.append(pos_src_batch)
                    dst_agg.append(pos_dst_batch)
                    ts_agg.append(pos_ts_batch)

                    pred_score_agg.append(pos_prob.cpu().numpy())
                    true_label_agg.append(pos_e_weight_batch)

                
            else:
                logger.info(f"DEBUG: no Positive edges in batch P-{p_b_idx}! [len(pos_src_batch): {len(pos_src_batch)}]")
            
            logger.info(f"INFO: Positive batch {p_b_idx} evaluation elapsed time (in seconds): {time.time() - start_p_b_time}")

            # ========== negative edges ==========
            for n_b_idx in range(NUM_NEG_BATCH_PER_POS_BATCH):
                start_n_b_time = time.time()

                neg_s_idx = (p_b_idx * NUM_NEG_BATCH_PER_POS_BATCH + n_b_idx) * TEST_BATCH_SIZE
                neg_e_idx = min(len(neg_data.sources) - 1, neg_s_idx + TEST_BATCH_SIZE)

                neg_src_batch = neg_data.sources[neg_s_idx: neg_e_idx]
                neg_dst_batch = neg_data.destinations[neg_s_idx: neg_e_idx]
                neg_ts_batch = neg_data.timestamps[neg_s_idx: neg_e_idx]
                neg_e_idx_batch = neg_data.edge_idxs[neg_s_idx: neg_e_idx]

                if len(neg_src_batch) > 1:
                    # logger.info(f"DEBUG: Number of negative edges in batch P-{p_b_idx}-N-{n_b_idx}: {len(neg_src_batch)}")
                    neg_prob = model.compute_edge_weights(neg_src_batch, neg_dst_batch, neg_ts_batch, 
                                                        neg_e_idx_batch, False,  n_neighbors)
                    neg_e_weight_batch = np.zeros(len(neg_src_batch))

                    if stats_filename is not None:
                        src_agg.append(neg_src_batch)
                        dst_agg.append(neg_dst_batch)
                        ts_agg.append(neg_ts_batch)

                        pred_score_agg.append(neg_prob.cpu().numpy())
                        true_label_agg.append(neg_e_weight_batch)

                # else:
                #     logger.info(f"DEBUG: no Negative edges in batch P-{p_b_idx}_N-{n_b_idx}!")
                
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



def eval_link_reg_only_pos_e(model, data, logger, batch_size=32, n_neighbors=20):
    """
    Evaluate the link regression tasks when negative edges are randomly selected & 
    the number of positive and negative edges are equal.
    This function is mainly used for the validation set of the link regression task.
    """
    pred_score_list, true_label_list = [], []

    with torch.no_grad():
        model = model.eval()
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            if s_idx == e_idx:
                logger.info(f"DEBUG: Not enough Positive edges in the batch {k}! [len(data.sources[s_idx:e_idx]): {len(data.sources[s_idx:e_idx])}]")
                continue
        
            # ========== positive edges
            src_l_cut = data.sources[s_idx: e_idx]
            dst_l_cut = data.destinations[s_idx: e_idx]
            ts_l_cut = data.timestamps[s_idx: e_idx]
            e_l_cut = data.edge_idxs[s_idx: e_idx]
            pos_e_weight_l_cut = data.true_y[s_idx: e_idx]

            # edge regression prediction
            if len(src_l_cut) > 1:
                pos_pred_score = model.compute_edge_weights(src_l_cut, dst_l_cut, ts_l_cut, e_l_cut, True, n_neighbors)

                pred_scores = pos_pred_score.cpu().numpy()
                true_labels = np.array(pos_e_weight_l_cut)

                pred_score_list.append(pred_scores)
                true_label_list.append(true_labels)
            else:
                logger.info(f"DEBUG: Not enough Positive edges in the batch {k}!")

        pred_score_list = np.concatenate(pred_score_list, axis=0)
        true_label_list = np.concatenate(true_label_list, axis=0)
        perf_dict = compute_perf_metrics(true_label_list, pred_score_list)

    return perf_dict