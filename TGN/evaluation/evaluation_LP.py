"""
Evaluation of Dynamic Link Prediction

Date:
  - March 1, 2023
"""

import torch
import math
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd 
from tqdm import tqdm
from sklearn.metrics import *



def get_metric_for_threshold(y_true, y_pred_score, threshold):
    """
    compute measures for a specific threshold
    """
    perf_measures = {}
    y_pred_label = y_pred_score > threshold
    perf_measures['acc'] = accuracy_score(y_true, y_pred_label)
    prec, rec, f1, num = precision_recall_fscore_support(y_true, y_pred_label, average='binary', zero_division=1)
    perf_measures['prec'] = prec
    perf_measures['rec'] = rec
    perf_measures['f1'] = f1
    return perf_measures


def compute_perf_metrics(y_true, y_pred_score):
    """
    compute extra performance measures
    """
    perf_dict = {}
    # find optimal threshold of au-roc
    perf_dict['ap'] = average_precision_score(y_true, y_pred_score)

    perf_dict['auc'] = roc_auc_score(y_true, y_pred_score)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_score)
    opt_idx = np.argmax(tpr - fpr)
    opt_thr_auroc = roc_thresholds[opt_idx]
    perf_dict['opt_thr_auc'] = opt_thr_auroc

    prec_pr_curve, rec_pr_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_score)
    perf_dict['aupr'] = auc(rec_pr_curve, prec_pr_curve)
    fscore = (2 * prec_pr_curve * rec_pr_curve) / (prec_pr_curve + rec_pr_curve)
    opt_idx = np.argmax(fscore)
    opt_thr_aupr = pr_thresholds[opt_idx]
    perf_dict['opt_thr_aupr'] = opt_thr_aupr

    # threshold = 0.5: it is assumed that the threshold should be set before the test phase
    perf_half_dict = get_metric_for_threshold(y_true, y_pred_score, 0.5)
    perf_dict['acc'] = perf_half_dict['acc']
    perf_dict['prec'] = perf_half_dict['prec']
    perf_dict['rec'] = perf_half_dict['rec']
    perf_dict['f1'] = perf_half_dict['f1']

    return perf_dict


def eval_link_pred(model, sampler, data, logger, stats_filename=None, batch_size=32, n_neighbors=20):
    """
    Evaluate the link prediction task
    """
    if stats_filename is not None:
        logger.info("Test edge evaluation statistics are saved at {}".format(stats_filename))
    pred_score_list, true_label_list = [], []
    src_agg, dst_agg, ts_agg = [], [], []
    # e_idx_agg = []
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
            src_l_cut = data.sources[s_idx:e_idx]
            dst_l_cut = data.destinations[s_idx:e_idx]
            ts_l_cut = data.timestamps[s_idx:e_idx]
            e_l_cut = data.edge_idxs[s_idx:e_idx]

            # ========== negative edges
            size = len(src_l_cut)
            neg_hist_ne_source, neg_hist_ne_dest, neg_rnd_source, neg_rnd_dest = sampler.sample(size, ts_l_cut[0], ts_l_cut[-1])

            src_l_fake = np.concatenate([neg_hist_ne_source, neg_rnd_source], axis=0)
            dst_l_fake = np.concatenate([neg_hist_ne_dest, neg_rnd_dest], axis=0)

            if sampler.neg_sample == 'haphaz_rnd':
                src_l_fake = src_l_cut

            # edge prediction
            if len(dst_l_fake) > 1:
                pos_prob = model.compute_edge_probabilities_modified(src_l_cut, dst_l_cut, ts_l_cut, e_l_cut, True, n_neighbors)
                neg_prob = model.compute_edge_probabilities_modified(src_l_fake, dst_l_fake, ts_l_cut, e_l_cut, False, n_neighbors)

                pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
                true_label = np.concatenate([np.ones(size), np.zeros(size)])

                pred_score_list.append(pred_score)
                true_label_list.append(true_label)

                if stats_filename is not None:
                    # positive edges
                    src_agg.append(src_l_cut)
                    dst_agg.append(dst_l_cut)
                    ts_agg.append(ts_l_cut)
                    # e_idx_agg.append(e_l_cut)
                    
                    # negative edges
                    src_agg.append(src_l_fake)
                    dst_agg.append(dst_l_fake)
                    ts_agg.append(ts_l_cut)
                    # e_idx_agg.append(e_l_cut)
            else:
                logger.info(f"DEBUG: Not enough Negative edges in the batch {k}! [len(dst_l_fake): {len(dst_l_fake)}]")

        pred_score_list = np.concatenate(pred_score_list, axis=0)
        true_label_list = np.concatenate(true_label_list, axis=0)
        perf_dict = compute_perf_metrics(true_label_list, pred_score_list)

        if stats_filename is not None:
            src_agg = np.concatenate(src_agg, axis=0)
            dst_agg = np.concatenate(dst_agg, axis=0)
            ts_agg = np.concatenate(ts_agg, axis=0)
            # e_idx_agg = np.concatenate(e_idx_agg, axis=0)
            # save to file 
            np.save(stats_filename + '_src.npy', src_agg)
            np.save(stats_filename + '_dst.npy', dst_agg)
            np.save(stats_filename + '_ts.npy', ts_agg)
            # np.save(stats_filename + '_e_idx.npy', e_idx_agg)
            np.save(stats_filename + '_pred_score.npy', pred_score_list)
            np.save(stats_filename + '_label.npy', true_label_list)

    return perf_dict


def eval_link_pred_one_snapshot(model, snap_data, logger, stats_filename=None, batch_size=32, n_neighbors=20):
    """
    Evaluate the link prediction task
    """
    if stats_filename is not None:
        logger.info("Test edge evaluation statistics are saved at {}".format(stats_filename))

    pos_data = snap_data['pos_e']
    neg_data = snap_data['neg_e']

    logger.info("INFO: Number of positive edges: {}".format(len(pos_data.sources)))
    logger.info("INFO: Number of negative edges: {}".format(len(neg_data.sources)))

    pred_score_agg, true_label_agg = [], []
    src_agg, dst_agg, ts_agg, e_idx_agg = [], [], [], []

    with torch.no_grad():
        model = model.eval()

        TEST_BATCH_SIZE = batch_size
        NUM_TEST_BATCH_POS = math.ceil(len(pos_data.sources) / TEST_BATCH_SIZE)

        NUM_TEST_BATCH_NEG = math.ceil(len(neg_data.sources) / TEST_BATCH_SIZE)
        NUM_NEG_BATCH_PER_POS_BATCH = math.ceil(NUM_TEST_BATCH_NEG / NUM_TEST_BATCH_POS)

        logger.info("INFO: NUM_TEST_BATCH_POS: {}".format(NUM_TEST_BATCH_POS))
        logger.info("INFO: NUM_NEG_BATCH_PER_POS_BATCH: {}".format(NUM_NEG_BATCH_PER_POS_BATCH))

        for p_b_idx in range(NUM_TEST_BATCH_POS):
            
            # ========== positive edges ==========
            pos_s_idx = p_b_idx * TEST_BATCH_SIZE
            pos_e_idx = min(len(pos_data.sources), pos_s_idx + TEST_BATCH_SIZE)                

            pos_src_batch = pos_data.sources[pos_s_idx: pos_e_idx]
            pos_dst_batch = pos_data.destinations[pos_s_idx: pos_e_idx]
            pos_ts_batch = pos_data.timestamps[pos_s_idx: pos_e_idx]
            pos_e_idx_batch = pos_data.edge_idxs[pos_s_idx: pos_e_idx]

            if len(pos_src_batch) > 1:
                pos_prob = model.compute_edge_probabilities_modified(pos_src_batch, pos_dst_batch, pos_ts_batch, 
                                                                    pos_e_idx_batch, True, n_neighbors)
                pos_true_label = np.ones(len(pos_src_batch))

                if stats_filename is not None:
                    src_agg.append(pos_src_batch)
                    dst_agg.append(pos_dst_batch)
                    ts_agg.append(pos_ts_batch)
                    # e_idx_agg.append(pos_e_idx_batch)

                    pred_score_agg.append(pos_prob.cpu().numpy())
                    true_label_agg.append(pos_true_label)
            else:
                logger.info(f"DEBUG: no Positive edges in batch P-{p_b_idx}! [len(pos_src_batch): {len(pos_src_batch)}]")

            # ========== negative edges ==========
            for n_b_idx in range(NUM_NEG_BATCH_PER_POS_BATCH):
                neg_s_idx = (p_b_idx * NUM_NEG_BATCH_PER_POS_BATCH + n_b_idx) * TEST_BATCH_SIZE
                neg_e_idx = min(len(neg_data.sources), neg_s_idx + TEST_BATCH_SIZE)

                neg_src_batch = neg_data.sources[neg_s_idx: neg_e_idx]
                neg_dst_batch = neg_data.destinations[neg_s_idx: neg_e_idx]
                neg_ts_batch = neg_data.timestamps[neg_s_idx: neg_e_idx]
                neg_e_idx_batch = neg_data.edge_idxs[neg_s_idx: neg_e_idx]

                if len(neg_src_batch) > 1:
                    neg_prob = model.compute_edge_probabilities_modified(neg_src_batch, neg_dst_batch, neg_ts_batch, 
                                                                        neg_e_idx_batch, False,  n_neighbors)
                    neg_true_label = np.zeros(len(neg_src_batch))

                    if stats_filename is not None:
                        src_agg.append(neg_src_batch)
                        dst_agg.append(neg_dst_batch)
                        ts_agg.append(neg_ts_batch)
                        # e_idx_agg.append(neg_e_idx_batch)

                        pred_score_agg.append(neg_prob.cpu().numpy())
                        true_label_agg.append(neg_true_label)
                else:
                    logger.info(f"DEBUG: no Negative edges in batch P-{p_b_idx}_N-{n_b_idx}! [len(neg_src_batch): {len(neg_src_batch)}]")


    if stats_filename is not None:
        src_agg = np.concatenate(src_agg, axis=0)
        dst_agg = np.concatenate(dst_agg, axis=0)
        ts_agg = np.concatenate(ts_agg, axis=0)
        # e_idx_agg = np.concatenate(e_idx_agg, axis=0)
        pred_score_agg = np.concatenate(pred_score_agg, axis=0)
        true_label_agg = np.concatenate(true_label_agg, axis=0)
        # save to file 
        np.save(stats_filename + '_src.npy', src_agg)
        np.save(stats_filename + '_dst.npy', dst_agg)
        np.save(stats_filename + '_ts.npy', ts_agg)
        # np.save(stats_filename + '_e_idx.npy', e_idx_agg)
        np.save(stats_filename + '_pred_score.npy', pred_score_agg)
        np.save(stats_filename + '_label.npy', true_label_agg)

