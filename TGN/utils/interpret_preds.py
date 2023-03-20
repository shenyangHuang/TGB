"""
Interpretation of prediction of the link regression task for
the EGO-SNAPSHOT setting

Date:
- March 05, 2023
"""

import numpy as np
import pandas as pd
import time
import argparse
import sys
import os
import math
from scipy.special import kl_div
from scipy import stats
from sklearn.metrics import *



NUM_SNAPSHOTS_TEST = {
                    'UNtrade': 3,
                      }



def get_args():
    parser = argparse.ArgumentParser('*** DLP Results Interpretation ***')
    # Related to process_stats
    parser.add_argument('--prefix', type=str, default='tgn_attn', choices=['tgn_attn', 'jodie_rnn', 'dyrep_rnn', 'EdgeRegress'], help='Model Prefix')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='wikipedia')
    parser.add_argument('--tr_neg_sample', type=str, default='haphaz_rnd', choices=['rnd', 'haphaz_rnd', 'hist', 'induc'],
                        help='Strategy for the negative sampling at the training phase.')
    parser.add_argument('--ts_neg_sample', type=str, default='haphaz_rnd', choices=['rnd', 'haphaz_rnd', 'hist', 'induc'],
                        help='Strategy for the negative edge sampling at the test phase.')
    parser.add_argument('--n_runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--avg_res', action='store_true', help='Compute and return the average performance results.')
    parser.add_argument('--eval_mode', type=str, default='STD', choices=['std', 'snapshot'], help='Evaluation mode.')
    parser.add_argument('--lp_mode', type=str, default='trans', choices=['trans', 'induc'],
                        help="Link prediction mode: transductive or inductive")
    parser.add_argument('--opt', type=str, default='gen', choices=['gen', 'avg', 'log'], 
                        help='Generate new statistics or report average of the existing statistics.')
    # Required for the data_loading
    parser.add_argument('--seed', type=int, default=123, help='random seed for all randomized algorithms')
    parser.add_argument('--data_usage', default=1.0, type=float, help='fraction of data to use (0-1)')
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='Whether to use disjoint set of new nodes for validation and test.')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of the validation data.')
    parser.add_argument('--test_ratio', type=float, default=0.15, help="Ratio of the test data.")
    parser.add_argument('--tr_rnd_ne_ratio', type=float, default=1.0,
                        help='Ratio of random negative edges sampled during TRAINING phase.')
    parser.add_argument('--ts_rnd_ne_ratio', type=float, default=1.0,
                        help='Ratio of random negative edges sampled during TEST phase.')

    try:
        args = parser.parse_args()
        print("INFO: args:", args)
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv


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



def generate_stats(partial_path, partial_name, stats_filename, DATA, 
                    EVAL_MODE, LP_MODE, N_RUNS, TR_NEG_SAMPLE, TS_NEG_SAMPLE, NUM_SNAPSHOTS):

    stats_list = []
    for run_idx in range(N_RUNS):
        for snap_idx in range(NUM_SNAPSHOTS_TEST[DATA]):
            iter_partial_filename = f"{partial_path}/snapshots/{partial_name}_{run_idx}_{LP_MODE}_{snap_idx}"
            if os.path.isfile(f"{iter_partial_filename}_src.npy"):  # if the current snapshot info exists...
                pred_dict = {
                            # 'sources': np.load(f"{iter_partial_filename}_src.npy"),
                            # 'destinations': np.load(f"{iter_partial_filename}_dst.npy"),
                            # 'timestamps': np.load(f"{iter_partial_filename}_ts.npy"),
                            'pred_scores': np.load(f"{iter_partial_filename}_pred_score.npy"),
                            'labels': np.load(f"{iter_partial_filename}_label.npy"),
                            }
                stats_snapshot_dict = compute_perf_metrics(pred_dict['labels'], pred_dict['pred_scores'])
                stats_snapshot_dict['DATA'] = DATA
                stats_snapshot_dict['EVAL_MODE'] = EVAL_MODE
                stats_snapshot_dict['LP_MODE'] = LP_MODE
                stats_snapshot_dict['TR_NEG_SAMPLE'] = TR_NEG_SAMPLE
                stats_snapshot_dict['TS_NEG_SAMPLE'] = TS_NEG_SAMPLE
                stats_list.append(stats_snapshot_dict)

                if run_idx == 0:
                    stats_header = ""
                    for key in list(stats_snapshot_dict.keys()):
                        stats_header += key + ","
                    if not os.path.isfile(stats_filename):
                        with open(stats_filename, 'w') as writer:
                            writer.write(stats_header)
            else:
                print(f"INFO: DATA: {DATA}, EVAL_MODE: {EVAL_MODE}, LR_MODE: {LP_MODE}: Run {run_idx}, Snapshot {snap_idx} does not exist!")
        
    stats_df = pd.read_csv(stats_filename)
    stats_df = pd.concat([stats_df, pd.DataFrame(stats_list)])
    stats_df.to_csv(stats_filename, index=False)


def gen_avg_perf(all_stats_df_filename, avg_stats_df_filename, DATA, EVAL_MODE, LP_MODE, TR_NEG_SAMPLE, TS_NEG_SAMPLE):
    
    all_cols_w_values = ['MSE', 'RMSE', 
                        # 'KL_div', 'PCC'
                        ]
    stats_df = pd.read_csv(all_stats_df_filename)
    setting_res = stats_df.loc[((stats_df['DATA'] == DATA) & (stats_df['LP_MODE'] == LP_MODE) & (stats_df['EVAL_MODE'] == EVAL_MODE) &
                                (stats_df['TR_NEG_SAMPLE'] == TR_NEG_SAMPLE) & (stats_df['TS_NEG_SAMPLE'] == TS_NEG_SAMPLE)),
                                all_cols_w_values]
    setting_avg_dict = dict(setting_res.mean())
    setting_avg_dict['DATA'] = DATA
    setting_avg_dict['LP_MODE'] = LP_MODE
    setting_avg_dict['EVAL_MODE'] = EVAL_MODE
    setting_avg_dict['TR_NEG_SAMPLE'] = TR_NEG_SAMPLE
    setting_avg_dict['TS_NEG_SAMPLE'] = TS_NEG_SAMPLE

    if not os.path.isfile(avg_stats_df_filename):
        avg_stats_df = pd.DataFrame([setting_avg_dict])
        avg_stats_df.to_csv(avg_stats_df_filename, index=False)
    else:
        avg_stats_df = pd.read_csv(avg_stats_df_filename)
        avg_stats_df = pd.concat([avg_stats_df, pd.DataFrame([setting_avg_dict])])
        avg_stats_df.to_csv(avg_stats_df_filename, index=False)


def get_avg_based_on_log(stats_filename, avg_stats_filename, model, data, tr_neg_sample, ts_neg_sample):
    """
    get the average of the results generated besides the log files
    """
    metrics = ['MSE', 'RMSE']
    stats_df = pd.read_csv(stats_filename)
    selected_rows = stats_df.loc[(stats_df['model'] == model) & (stats_df['data'] == data) 
                                 & (stats_df['tr_neg_sample'] == tr_neg_sample) & (stats_df['ts_neg_sample'] == ts_neg_sample), metrics]
    avg_res = dict(selected_rows.mean())
    avg_res['MODEL'] = model
    avg_res['DATA'] = data
    avg_res['TR_NEG_SAMPLE'] = tr_neg_sample
    avg_res['TS_NEG_SAMPLE'] = ts_neg_sample

    if not os.path.isfile(avg_stats_filename):
        avg_stats_df = pd.DataFrame([avg_res])
        avg_stats_df.to_csv(avg_stats_filename, index=False)
    else:
        avg_stats_df = pd.read_csv(avg_stats_filename)
        avg_stats_df = pd.concat([avg_stats_df, pd.DataFrame([avg_res])])
        avg_stats_df.to_csv(avg_stats_filename, index=False)



def main():
    """
    to generate performance metrics based on the prediction scores of the dynamic link regression task

    Command:
    python utils/interpret_preds.py -d "UNtrade" --prefix "tgn_attn" --n_runs 5 --tr_rnd_ne_ratio 1.0 --ts_rnd_ne_ratio 1.0 --tr_neg_sample "rnd"  --ts_neg_samp "rnd"
    """
    args, _ = get_args()
    DATA = args.data
    N_RUNS = args.n_runs
    OPT = args.opt
    # EVAL_MODE = args.eval_mode  # 'STD' or 'SNAPSHOT'
    EVAL_MODE = 'SNAPSHOT'  # 'STD' or 'SNAPSHOT'
    # LP_MODE = args.lp_mode  # 'trans' or 'induc'
    LP_MODE = 'trans'  # 'trans' or 'induc'
    TR_NEG_SAMPLE = args.tr_neg_sample
    TS_NEG_SAMPLE = args.ts_neg_sample
    TR_RND_NE_RATIO = args.tr_rnd_ne_ratio
    TS_RND_NE_RATIO = args.ts_rnd_ne_ratio
    MODEL_NAME = args.prefix
    TASK = 'LR'

    partial_path = f"{TASK}_stats/{DATA}"
    partial_name = f"{MODEL_NAME}_{DATA}_TR_{TR_NEG_SAMPLE}_TS_{TS_NEG_SAMPLE}"

    stats_filename = f"{TASK}_stats/Interpreted_Stats/{TASK}_pred_scores_{MODEL_NAME}_{EVAL_MODE}.csv"
    avg_stats_df_filename = f"{TASK}_stats/Interpreted_Stats/{TASK}_pred_scores_{MODEL_NAME}_{EVAL_MODE}_avg.csv"

    print("="*150)
    print(f"INFO: TASK: {TASK}, METHOD: {MODEL_NAME}, DATA: {DATA}, OPT: {OPT}, TR_NEG_SAMPLE: {TR_NEG_SAMPLE}, TS_NEG_SAMPLE: {TS_NEG_SAMPLE}, N_RUNS: {N_RUNS}, EVAL_MODE: {EVAL_MODE}, LP_MODE: {LP_MODE}")
    print("="*150)

    if OPT == 'avg':
        print("INFO: *** Reporting the average statistics ***")
        gen_avg_perf(stats_filename, avg_stats_df_filename, DATA, EVAL_MODE, LP_MODE, TR_NEG_SAMPLE, TS_NEG_SAMPLE)

    elif OPT == 'gen':
        print("INFO: *** Generating statistics ***")
        generate_stats(partial_path, partial_name, stats_filename, DATA, 
                    EVAL_MODE, LP_MODE, N_RUNS, TR_NEG_SAMPLE, TS_NEG_SAMPLE, NUM_SNAPSHOTS_TEST)

    elif OPT == 'log':
        print("INFO: *** Generating based on information in the log file! ***")
        if LP_MODE == 'induc':
            keyword = 'INDUC'
        elif LP_MODE == 'trans':
            keyword = 'TRANS'
        stats_filename = f'{TASK}_stats/STD_pred_{keyword}.csv'
        avg_stats_filename = f'{TASK}_stats/Interpreted_Stats/STD_pred_{keyword}_avg.csv'
        get_avg_based_on_log(stats_filename, avg_stats_filename, MODEL_NAME, DATA, TR_NEG_SAMPLE, TS_NEG_SAMPLE)


    else:
        raise ValueError("INFO: Invalid option!!!")



if __name__ == '__main__':
    main()